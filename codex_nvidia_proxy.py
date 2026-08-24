
import sys
import os
import json
import uuid
import getpass
import re
import hmac
import secrets
import time
from collections import deque
from datetime import datetime
from threading import Condition, RLock

from flask import Flask, jsonify, render_template, request, Response
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_THREADS_PER_KEY = 4
MAX_THREADS_PER_KEY = 64
# Waitress 的线程数在启动后无法动态调整。代理最多占用 64 个长连接工作线程，
# 另外固定预留 4 个线程供控制台和管理 API 使用。
MAX_TOTAL_UPSTREAM_SLOTS = 64
MANAGEMENT_WORKER_RESERVE = 4
WEB_SERVER_THREADS = MAX_TOTAL_UPSTREAM_SLOTS + MANAGEMENT_WORKER_RESERVE
# 模型列表是控制台辅助请求，不应继承 OpenAI SDK 的长默认超时。
# 同时限制单个 Key 和整个聚合请求的耗时，避免一个不可达的上游拖住后续 Key。
MODEL_LIST_REQUEST_TIMEOUT = 10.0
MODEL_LIST_TOTAL_TIMEOUT = 30.0


def _load_dotenv():
    """加载 .env 文件到 os.environ（不覆盖已有的系统环境变量）"""
    env_file = os.path.join(BASE_DIR, ".nvidia_env")
    if not os.path.exists(env_file):
        return
    key_names = ("NVIDIA_API_KEY", "NVIDIA_API_KEYS")
    had_key = any(k in os.environ for k in key_names)
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip("\"'")
            # 单 Key 和多 Key 是同一项配置的两个别名。系统环境中任一别名存在时，
            # 都不能再从文件加载另一个别名，否则文件中的旧配置会反向覆盖系统配置。
            if had_key and key in key_names:
                continue
            if key and key not in os.environ:
                os.environ[key] = val
    if any(k in os.environ for k in key_names):
        os.environ["_NVIDIA_KEY_SOURCE"] = "sys" if had_key else "dotenv"


def _parse_api_keys(value):
    """解析 NVIDIA_API_KEYS，支持逗号、分号或换行分隔。"""
    if not value:
        return []
    return [key.strip().strip("\"'") for key in re.split(r"[,;\r\n]+", value) if key.strip()]


def _configured_api_keys():
    """读取多 Key 配置；未配置 NVIDIA_API_KEYS 时兼容单 Key 配置。"""
    keys = _parse_api_keys(os.environ.get("NVIDIA_API_KEYS", ""))
    if not keys:
        keys = _parse_api_keys(os.environ.get("NVIDIA_API_KEY", ""))
    # 防止同一个 Key 被重复配置，避免重复占用并发槽位。
    return list(dict.fromkeys(keys))


def _parse_threads_per_key(value):
    """解析环境变量，并将结果收敛到后端允许的安全范围。"""
    try:
        return min(MAX_THREADS_PER_KEY, max(1, int(value)))
    except (TypeError, ValueError):
        return DEFAULT_THREADS_PER_KEY


def _validated_threads_per_key(value):
    """严格校验 API 输入，避免静默接受越界或非整数值。"""
    if isinstance(value, bool):
        raise ValueError("每 Key 并发必须是整数")
    try:
        text = str(value).strip()
        if not text or not re.fullmatch(r"[+-]?\d+", text):
            raise ValueError
        threads = int(text)
    except (TypeError, ValueError):
        raise ValueError("每 Key 并发必须是整数") from None
    if not 1 <= threads <= MAX_THREADS_PER_KEY:
        raise ValueError(f"每 Key 并发必须在 1 到 {MAX_THREADS_PER_KEY} 之间")
    return threads


def _capacity_error(api_keys, threads_per_key):
    """返回配置容量错误；空字符串表示配置可由固定 Web worker 池承载。"""
    total_slots = len(api_keys) * threads_per_key
    if total_slots > MAX_TOTAL_UPSTREAM_SLOTS:
        return (
            f"总并发槽位不能超过 {MAX_TOTAL_UPSTREAM_SLOTS} "
            f"（当前为 {len(api_keys)} × {threads_per_key} = {total_slots}）"
        )
    return ""


def _ensure_api_keys():
    """确保 API Key 已设置：系统环境变量 > .nvidia_env > 交互输入。"""
    keys = _configured_api_keys()
    if keys:
        src = os.environ.get("_NVIDIA_KEY_SOURCE", "")
        env_file = os.path.join(BASE_DIR, ".nvidia_env")
        if src == "sys" or (not src and not os.path.exists(env_file)):
            return keys, "系统环境变量"
        return keys, ".nvidia_env"

    print("=" * 60)
    print("  未检测到 NVIDIA_API_KEY")
    print("=" * 60)
    print()
    print("  从 https://build.nvidia.com/ 获取 API Key")
    print("  登录后点击任一模型 → Get API Key")
    print()
    print("  你也可以设置 NVIDIA_API_KEY 或 NVIDIA_API_KEYS 后重启")
    print()

    try:
        key = getpass.getpass("  请输入你的 NVIDIA API Key: ").strip()
    except (EOFError, KeyboardInterrupt):
        key = ""

    if not key:
        print()
        print("  ERROR: 未输入 API Key，程序退出。")
        print()
        input("  按 Enter 退出...")
        sys.exit(1)

    env_file = os.path.join(BASE_DIR, ".nvidia_env")
    existing = {}
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                existing[k.strip()] = f"{k.strip()}={v.strip()}"
    existing["NVIDIA_API_KEY"] = f"NVIDIA_API_KEY={key}"

    with open(env_file, "w", encoding="utf-8") as f:
        for line in existing.values():
            f.write(line + "\n")
        if "NVIDIA_MODEL" not in existing:
            f.write("NVIDIA_MODEL=nvidia/llama-3.1-nemotron-70b-instruct\n")

    os.environ["NVIDIA_API_KEY"] = key
    print()
    print(f"  API Key 已保存到: {env_file}")
    print()
    return [key], ".nvidia_env (已保存)"


def _ensure_api_key():
    """兼容旧调用方：返回第一个 API Key。"""
    keys, source = _ensure_api_keys()
    return (keys[0] if keys else ""), source


_load_dotenv()

DEBUG_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvidia_proxy_debug.log")

app = Flask(__name__)

# ===================== 配置 =====================
NVIDIA_API_KEYS = _configured_api_keys()
NVIDIA_API_KEY = NVIDIA_API_KEYS[0] if NVIDIA_API_KEYS else ""
NVIDIA_MODEL = os.environ.get(
    "NVIDIA_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct"
).strip()
NVIDIA_BASE_URL = os.environ.get(
    "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
).strip()
NVIDIA_DEBUG = os.environ.get("NVIDIA_DEBUG", "0").strip() in ("1", "true", "True", "yes")
NVIDIA_THREADS_PER_KEY = _parse_threads_per_key(os.environ.get("NVIDIA_THREADS_PER_KEY", "4"))
# =================================================


# 每个 Key 只创建一个 OpenAI 客户端，并用独立并发槽限制该 Key 的并发数。
NVIDIA_CLIENT_POOL = []
_POOL_CONDITION = Condition(RLock())
_POOL_ROUND_ROBIN_INDEX = 0
# 槽位状态按 Key 共享，客户端池重建时也不会重置在途请求的占用计数。
_KEY_SLOT_STATES = {}
_SERVICE_STATE_LOCK = RLock()
_SERVICE_RUNNING = bool(NVIDIA_API_KEYS) and not _capacity_error(
    NVIDIA_API_KEYS, NVIDIA_THREADS_PER_KEY
)
_RUNTIME_LOGS = deque(maxlen=2000)
_RUNTIME_LOG_LOCK = RLock()
_RUNTIME_LOG_SEQUENCE = 0
_CSRF_TOKEN = secrets.token_urlsafe(32)
_RUNTIME_LOG_ENABLED = os.environ.get("NVIDIA_RUNTIME_LOG", "0").strip() in (
    "1", "true", "True", "yes"
)


def _runtime_log(level, event, message, request_id=""):
    """写入仅存放于内存的运行日志；不记录完整 API Key。"""
    global _RUNTIME_LOG_SEQUENCE
    # 关闭时尽早返回，尤其避免为每个上游 delta 加锁或格式化大段 JSON。
    if not _RUNTIME_LOG_ENABLED:
        return None
    text = str(message() if callable(message) else message)
    if len(text) > 12000:
        text = text[:12000] + "\n... [truncated]"
    with _RUNTIME_LOG_LOCK:
        _RUNTIME_LOG_SEQUENCE += 1
        entry = {
            "id": _RUNTIME_LOG_SEQUENCE,
            "time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "level": str(level).upper(),
            "event": event,
            "request_id": request_id,
            "message": text,
        }
        _RUNTIME_LOGS.append(entry)
        return entry


def _init_client_pool(api_keys=None):
    """为每个 API Key 创建客户端，并复用跨池共享的并发占用状态。"""
    global NVIDIA_CLIENT_POOL, NVIDIA_API_KEYS, NVIDIA_API_KEY, _POOL_ROUND_ROBIN_INDEX
    keys = list(dict.fromkeys(api_keys or _configured_api_keys()))
    capacity_error = _capacity_error(keys, NVIDIA_THREADS_PER_KEY)
    if capacity_error:
        raise ValueError(capacity_error)

    with _POOL_CONDITION:
        clients = []
        for index, key in enumerate(keys, start=1):
            slot_state = _KEY_SLOT_STATES.setdefault(key, {"active": 0})
            clients.append({
                "index": index,
                "key": key,
                "client": OpenAI(base_url=NVIDIA_BASE_URL, api_key=key),
                "slot_state": slot_state,
            })
        NVIDIA_API_KEYS = keys
        NVIDIA_API_KEY = keys[0] if keys else ""
        NVIDIA_CLIENT_POOL = clients
        _POOL_ROUND_ROBIN_INDEX %= max(1, len(clients))
        _POOL_CONDITION.notify_all()
    return clients


def _get_client_pool():
    """取得客户端池；直接导入 app 使用时也支持惰性初始化。"""
    if not NVIDIA_CLIENT_POOL:
        _init_client_pool()
    return NVIDIA_CLIENT_POOL


def _active_upstream_slots():
    """统计当前及已退出客户端池的全部在途请求；调用方须持有池锁。"""
    return sum(state["active"] for state in _KEY_SLOT_STATES.values())


def _acquire_client_slot(preferred_entry=None, blocking=True, deadline=None):
    """从客户端池中选择空闲 Key，同时执行跨热切换池的全局并发限制。"""
    global _POOL_ROUND_ROBIN_INDEX
    with _POOL_CONDITION:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                return None
            pool = _get_client_pool()
            if not pool:
                raise RuntimeError("未配置 NVIDIA API Key")

            # 被热切换移出池的 Key 仍可能有流式请求在途。它们的状态会保留在
            # _KEY_SLOT_STATES 中，并继续占用全局槽位，避免实际并发超过 Web
            # worker 为代理请求预留的上限。
            has_global_slot = _active_upstream_slots() < MAX_TOTAL_UPSTREAM_SLOTS
            if preferred_entry is not None:
                state = preferred_entry["slot_state"]
                if has_global_slot and state["active"] < NVIDIA_THREADS_PER_KEY:
                    state["active"] += 1
                    return preferred_entry
                if not blocking:
                    return None
                wait_timeout = None if deadline is None else max(0, deadline - time.monotonic())
                _POOL_CONDITION.wait(wait_timeout)
                continue

            if has_global_slot:
                count = len(pool)
                for offset in range(count):
                    idx = (_POOL_ROUND_ROBIN_INDEX + offset) % count
                    entry = pool[idx]
                    state = entry["slot_state"]
                    if state["active"] < NVIDIA_THREADS_PER_KEY:
                        state["active"] += 1
                        _POOL_ROUND_ROBIN_INDEX = (idx + 1) % count
                        return entry
            if not blocking:
                return None
            wait_timeout = None if deadline is None else max(0, deadline - time.monotonic())
            _POOL_CONDITION.wait(wait_timeout)


def _release_client_slot(entry):
    with _POOL_CONDITION:
        state = entry["slot_state"]
        if state["active"] <= 0:
            raise RuntimeError("客户端并发槽位被重复释放")
        state["active"] -= 1
        _POOL_CONDITION.notify_all()


def _mask_key(key):
    """只用于 UI 状态展示，避免把完整 API Key 返回给浏览器。"""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def _save_runtime_config(keys, model, threads_per_key):
    """保存运行配置；仅当 keys 非 None 时才持久化或改写凭据。"""
    env_file = os.path.join(BASE_DIR, ".nvidia_env")
    replacements = {
        "NVIDIA_MODEL": model,
        "NVIDIA_THREADS_PER_KEY": str(threads_per_key),
    }
    persist_keys = keys is not None
    if persist_keys:
        replacements["NVIDIA_API_KEYS"] = ",".join(keys)
    lines = []
    seen = set()
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if "=" in stripped and not stripped.startswith("#"):
                    name = stripped.split("=", 1)[0].strip()
                    if persist_keys and name == "NVIDIA_API_KEY":
                        # 多 Key 配置优先，删除旧的单 Key 行，避免产生歧义。
                        continue
                    if name in replacements:
                        if name not in seen:
                            lines.append(f"{name}={replacements[name]}\n")
                            seen.add(name)
                        continue
                lines.append(line if line.endswith("\n") else line + "\n")

    for name, value in replacements.items():
        if name not in seen:
            lines.append(f"{name}={value}\n")

    with open(env_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    if persist_keys:
        os.environ["NVIDIA_API_KEYS"] = ",".join(keys)
        os.environ.pop("NVIDIA_API_KEY", None)
        os.environ["_NVIDIA_KEY_SOURCE"] = "dotenv"
    os.environ["NVIDIA_MODEL"] = model
    os.environ["NVIDIA_THREADS_PER_KEY"] = str(threads_per_key)


def _service_status():
    with _SERVICE_STATE_LOCK:
        running = _SERVICE_RUNNING
        keys = list(NVIDIA_API_KEYS)
        threads_per_key = NVIDIA_THREADS_PER_KEY
        model = NVIDIA_MODEL
        base_url = NVIDIA_BASE_URL
    capacity_error = _capacity_error(keys, threads_per_key)
    return {
        "running": running,
        "configured": bool(keys),
        "key_count": len(keys),
        "keys": [_mask_key(key) for key in keys],
        "threads_per_key": threads_per_key,
        "total_threads": len(keys) * threads_per_key if not capacity_error else 0,
        "max_total_threads": MAX_TOTAL_UPSTREAM_SLOTS,
        "configuration_error": capacity_error,
        "model": model,
        "base_url": base_url,
        "runtime_log_enabled": _RUNTIME_LOG_ENABLED,
    }


def _set_service_running(value):
    global _SERVICE_RUNNING
    with _SERVICE_STATE_LOCK:
        _SERVICE_RUNNING = bool(value)


def _clean_schema(obj):
    if not isinstance(obj, dict):
        return obj
    cleaned = {}
    for k, v in obj.items():
        if k in ("additionalProperties", "strict"):
            continue
        if isinstance(v, dict):
            cleaned[k] = _clean_schema(v)
        elif isinstance(v, list):
            cleaned[k] = [_clean_schema(i) if isinstance(i, dict) else i for i in v]
        else:
            cleaned[k] = v
    return cleaned


def _convert_tools(tools: list) -> list:
    result = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        func = {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
        }
        if "parameters" in tool:
            func["parameters"] = _clean_schema(tool["parameters"])
        result.append({"type": "function", "function": func})
    return result


def _convert_tool_choice(tc):
    if tc is None:
        return "auto"
    if isinstance(tc, str):
        return tc
    if isinstance(tc, dict) and tc.get("type") == "function":
        return {"type": "function", "function": {"name": tc.get("name", "")}}
    return "auto"


def _estimate_tokens(text):
    return max(1, len(text) // 4)


def extract_messages(data: dict):
    """
    从 Responses API 请求中提取 messages 列表、tools 列表和 tool_choice。
    """
    ROLE_MAP = {"developer": "system"}
    raw_tools = data.get("tools", [])
    tools = _convert_tools(raw_tools)
    tool_choice = _convert_tool_choice(data.get("tool_choice"))

    if "input" not in data:
        if "messages" in data:
            return data["messages"], tools, tool_choice
        return [], tools, tool_choice

    inp = data["input"]
    if isinstance(inp, str):
        messages = []
        if "instructions" in data and data["instructions"]:
            messages.append({"role": "system", "content": data["instructions"]})
        messages.append({"role": "user", "content": inp})
        return messages, tools, tool_choice

    if not isinstance(inp, list):
        return [], tools, tool_choice

    messages = []
    if "instructions" in data and data["instructions"]:
        messages.append({"role": "system", "content": data["instructions"]})

    pending_tool_calls = []
    pending_reasoning = ""

    def _flush_tool_calls():
        nonlocal pending_tool_calls, pending_reasoning
        if pending_tool_calls:
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": pending_tool_calls,
            }
            if pending_reasoning:
                msg["reasoning_content"] = pending_reasoning
            messages.append(msg)
            pending_tool_calls = []
            pending_reasoning = ""

    for item in inp:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")

        if item_type == "message":
            _flush_tool_calls()
            role = item.get("role", "user")
            role = ROLE_MAP.get(role, role)
            content = item.get("content", "")
            if isinstance(content, list):
                texts = []
                tool_calls = []
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    c_type = c.get("type")
                    if c_type in ("text", "input_text", "output_text"):
                        t = c.get("text", "")
                        if t.strip():
                            texts.append(t)
                    elif c_type == "tool_call":
                        tool_calls.append({
                            "id": c.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": c.get("name", ""),
                                "arguments": c.get("arguments", ""),
                            }
                        })
                text_content = "\n".join(texts)
                if tool_calls:
                    msg = {"role": role, "content": text_content or ""}
                    msg["tool_calls"] = tool_calls
                    messages.append(msg)
                elif text_content:
                    msg = {"role": role, "content": text_content}
                    messages.append(msg)
            elif isinstance(content, str) and content.strip():
                msg = {"role": role, "content": content.strip()}
                messages.append(msg)

        elif item_type == "function_call":
            pending_tool_calls.append({
                "id": item.get("call_id", ""),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", ""),
                }
            })

        elif item_type == "function_call_output":
            _flush_tool_calls()
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": item.get("output", ""),
            })

    _flush_tool_calls()

    # 重排消息
    reordered = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            expected_ids = {tc["id"] for tc in msg["tool_calls"]}
            tool_msgs = []
            non_tool_msgs = []
            j = i + 1
            while j < len(messages) and expected_ids:
                nxt = messages[j]
                if nxt.get("role") == "tool" and nxt.get("tool_call_id") in expected_ids:
                    expected_ids.remove(nxt["tool_call_id"])
                    tool_msgs.append(nxt)
                elif nxt.get("role") in ("system", "developer"):
                    non_tool_msgs.append(nxt)
                else:
                    break
                j += 1
            reordered.extend(non_tool_msgs)
            reordered.append(msg)
            reordered.extend(tool_msgs)
            i = j
        else:
            reordered.append(msg)
            i += 1
    messages = reordered

    return messages, tools, tool_choice


# ---- Anthropic Messages API（Claude Code）适配 ----
def _anthropic_text(content):
    """将 Anthropic content block 转成可放入 OpenAI content 的文本。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    texts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") in ("text", "input_text", "output_text"):
            text = block.get("text", "")
            if text:
                texts.append(str(text))
    return "\n".join(texts)


def _anthropic_image_block(block):
    """尽量保留 Anthropic 图片输入；文本模型会由上游自行返回不支持错误。"""
    source = block.get("source") if isinstance(block, dict) else None
    if not isinstance(source, dict):
        return None
    if source.get("type") == "base64" and source.get("media_type") and source.get("data"):
        url = f"data:{source['media_type']};base64,{source['data']}"
    elif source.get("type") == "url" and source.get("url"):
        url = source["url"]
    else:
        return None
    return {"type": "image_url", "image_url": {"url": url}}


def _anthropic_messages(data: dict):
    """把 Anthropic Messages API 请求转换成 NVIDIA 所用的 OpenAI messages。"""
    messages = []
    system = data.get("system")
    if system:
        system_text = _anthropic_text(system)
        if system_text:
            messages.append({"role": "system", "content": system_text})

    raw_messages = data.get("messages", [])
    if not isinstance(raw_messages, list):
        return messages

    for raw in raw_messages:
        if not isinstance(raw, dict):
            continue
        role = raw.get("role", "user")
        content = raw.get("content", "")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        if not isinstance(blocks, list):
            blocks = []

        text_parts = []
        openai_parts = []
        tool_results = []
        tool_calls = []
        for block in blocks:
            if not isinstance(block, dict):
                if isinstance(block, str):
                    text_parts.append(block)
                continue
            block_type = block.get("type")
            if block_type in ("text", "input_text", "output_text"):
                text = block.get("text", "")
                if text:
                    text_parts.append(str(text))
                    openai_parts.append({"type": "text", "text": str(text)})
            elif block_type == "image":
                image = _anthropic_image_block(block)
                if image:
                    openai_parts.append(image)
            elif block_type == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                    },
                })
            elif block_type == "tool_result":
                result = block.get("content", "")
                result_text = _anthropic_text(result) if isinstance(result, list) else str(result or "")
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": result_text,
                })

        if role == "assistant":
            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": "\n".join(text_parts) or "",
                    "tool_calls": tool_calls,
                })
            elif text_parts:
                messages.append({"role": "assistant", "content": "\n".join(text_parts)})
        elif role == "user":
            # Anthropic 将 tool_result 放在 user 消息中；OpenAI 要求它们紧跟
            # assistant.tool_calls。混合块时先放 tool 消息，避免中间插入普通
            # user 消息导致 NVIDIA/OpenAI 校验失败。
            messages.extend(tool_results)
            if openai_parts:
                messages.append({
                    "role": "user",
                    "content": "\n".join(text_parts) if not any(p.get("type") == "image_url" for p in openai_parts) else openai_parts,
                })
            elif not tool_results:
                messages.append({"role": "user", "content": ""})
        else:
            # 兼容少数代理发送的 developer/system 消息。
            text = "\n".join(text_parts) or _anthropic_text(content)
            if text:
                messages.append({"role": "system" if role == "developer" else role, "content": text})

    # 工具结果必须跟在对应 assistant.tool_calls 后面。Claude Code 通常已经
    # 按此顺序发送，但这里再做一次稳定重排以兼容同一 user 块中的混合内容。
    reordered = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            ids = {call.get("id") for call in msg["tool_calls"]}
            pending = []
            j = i + 1
            while j < len(messages):
                nxt = messages[j]
                if nxt.get("role") == "tool" and nxt.get("tool_call_id") in ids:
                    pending.append(nxt)
                    ids.discard(nxt.get("tool_call_id"))
                    j += 1
                elif nxt.get("role") in ("system", "developer"):
                    reordered.append(nxt)
                    j += 1
                else:
                    break
            reordered.append(msg)
            reordered.extend(pending)
            i = j
        else:
            reordered.append(msg)
            i += 1
    return reordered


def _anthropic_tools(raw_tools):
    """Anthropic tool 定义 -> OpenAI function tool。"""
    result = []
    if not isinstance(raw_tools, list):
        return result
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        # Anthropic 使用 name/description/input_schema，没有 type=function 包装。
        name = tool.get("name")
        if not name:
            continue
        result.append({
            "type": "function",
            "function": {
                "name": name,
                "description": tool.get("description", ""),
                "parameters": _clean_schema(tool.get("input_schema", tool.get("parameters", {"type": "object"}))),
            },
        })
    return result


def _anthropic_tool_choice(choice):
    if not isinstance(choice, dict):
        return choice if choice in ("auto", "none", "required") else "auto"
    kind = choice.get("type", "auto")
    if kind == "auto":
        return "auto"
    if kind in ("any", "required"):
        return "required"
    if kind in ("tool", "function") and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    return "auto"


def _anthropic_parallel_tool_calls(choice):
    """保留 Anthropic tool_choice.disable_parallel_tool_use 约束。"""
    if isinstance(choice, dict) and choice.get("disable_parallel_tool_use") is True:
        return False
    return None


def _anthropic_stop_sequences(value):
    """规范化 Anthropic stop_sequences，便于响应中回填 stop_sequence。"""
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if item is not None and str(item)]


def _anthropic_stop_result(finish_reason, stop_sequences=None, text="", has_tools=False):
    """映射 OpenAI finish_reason 到 Anthropic stop_reason/stop_sequence。"""
    if finish_reason == "length":
        # 工具参数可能在 max_tokens 限制下只生成了一部分；不能把它
        # 报告为可执行的 tool_use，否则调用方可能用不完整参数执行工具。
        return "max_tokens", None
    if finish_reason in ("tool_calls", "function_call") or has_tools:
        return "tool_use", None
    stop_sequences = _anthropic_stop_sequences(stop_sequences)
    if finish_reason == "stop" and stop_sequences:
        # 部分 OpenAI 兼容端点会把停止串从输出中移除，因此无法总是从
        # 文本末尾判断命中的具体项；单项或无法辨别时回填第一个请求项。
        matched = next((item for item in stop_sequences if text.endswith(item)), None)
        return "stop_sequence", matched or stop_sequences[0]
    return "end_turn", None


def _anthropic_effective_model(requested):
    """Claude Code 发送的是 Claude 模型名，不能原样传给 NVIDIA NIM。"""
    if not requested:
        return NVIDIA_MODEL
    requested = str(requested).strip()
    if not requested:
        return NVIDIA_MODEL
    # Claude Code 默认使用 claude-*（或 anthropic/*）名称；这类名称统一
    # 映射到控制台/环境变量中的 NVIDIA_MODEL。手工传入 nvidia/* 等名称时
    # 仍允许按请求选择模型，便于测试多个 NIM 模型。
    if requested.lower().startswith(("claude", "anthropic/")):
        return NVIDIA_MODEL
    return requested


def _obj_get(obj, name, default=None):
    """同时支持 OpenAI SDK 对象和测试中使用的 dict。"""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# ---- Web UI / 控制 API ----
@app.get("/")
def web_ui():
    response = Response(render_template("index.html", csrf_token=_CSRF_TOKEN))
    response.headers["Cache-Control"] = "no-store"
    return response


@app.before_request
def protect_management_writes():
    """管理写接口必须携带控制台页面中的 CSRF token。"""
    if not request.path.startswith("/api/"):
        return None
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return None
    supplied_token = request.headers.get("X-CSRF-Token", "")
    if not supplied_token or not hmac.compare_digest(supplied_token, _CSRF_TOKEN):
        return jsonify({"error": "CSRF 校验失败，请从本机控制台执行此操作"}), 403
    return None


@app.get("/api/status")
def api_status():
    return jsonify(_service_status())


@app.get("/api/logs")
def api_logs():
    try:
        after_id = max(0, int(request.args.get("after", "0")))
    except ValueError:
        after_id = 0
    try:
        limit = min(1000, max(1, int(request.args.get("limit", "500"))))
    except ValueError:
        limit = 500
    with _RUNTIME_LOG_LOCK:
        entries = [entry.copy() for entry in _RUNTIME_LOGS if entry["id"] > after_id][:limit]
        latest_id = _RUNTIME_LOG_SEQUENCE
        enabled = _RUNTIME_LOG_ENABLED
    return jsonify({"logs": entries, "latest_id": latest_id, "enabled": enabled})


@app.post("/api/logs/toggle")
def api_logs_toggle():
    global _RUNTIME_LOG_ENABLED
    payload = request.get_json(silent=True) or {}
    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        return jsonify({"error": "enabled 必须是布尔值"}), 400
    with _RUNTIME_LOG_LOCK:
        _RUNTIME_LOG_ENABLED = enabled
    if enabled:
        _runtime_log("INFO", "logging.enabled", "详细运行日志记录已开启")
    return jsonify({"ok": True, "enabled": enabled})


@app.post("/api/logs/clear")
def api_logs_clear():
    with _RUNTIME_LOG_LOCK:
        _RUNTIME_LOGS.clear()
        latest_id = _RUNTIME_LOG_SEQUENCE
    return jsonify({"ok": True, "latest_id": latest_id, "enabled": _RUNTIME_LOG_ENABLED})


@app.post("/api/config")
def api_config():
    global NVIDIA_MODEL, NVIDIA_THREADS_PER_KEY

    payload = request.get_json(silent=True) or {}
    raw_keys = payload.get("api_keys", payload.get("keys", ""))
    if isinstance(raw_keys, list):
        raw_keys = "\n".join(str(item) for item in raw_keys)
    elif raw_keys is None:
        raw_keys = ""
    submitted_keys = list(dict.fromkeys(_parse_api_keys(str(raw_keys))))
    keys = submitted_keys
    if not submitted_keys:
        # UI 留空时保留已配置的 Key，避免状态接口只返回掩码后无法修改模型。
        keys = _configured_api_keys()
    if not keys:
        return jsonify({"error": "请至少填写一个 NVIDIA API Key"}), 400

    model = str(payload.get("model") or NVIDIA_MODEL).strip()
    if not model:
        return jsonify({"error": "模型名称不能为空"}), 400
    if "\n" in model or "\r" in model:
        return jsonify({"error": "模型名称不能包含换行符"}), 400
    try:
        threads = _validated_threads_per_key(
            payload.get("threads_per_key", NVIDIA_THREADS_PER_KEY)
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    capacity_error = _capacity_error(keys, threads)
    if capacity_error:
        return jsonify({"error": capacity_error}), 400

    # 配置、客户端池和运行状态必须在同一把锁下完成切换，避免与并发的
    # /api/service/start 交错，导致保存后代理被意外重新启动。
    with _SERVICE_STATE_LOCK:
        # 配置变更时先停止新请求，再替换客户端池；已有请求仍可安全释放旧槽位。
        _set_service_running(False)
        NVIDIA_THREADS_PER_KEY = threads
        # 输入框留空时只沿用内存中的 Key，不将可能来自系统环境变量的凭据写入磁盘。
        _save_runtime_config(submitted_keys or None, model, threads)
        _init_client_pool(keys)
        NVIDIA_MODEL = model
        _runtime_log(
            "INFO", "config.saved",
            f"配置已保存：keys={len(keys)}, threads_per_key={threads}, model={model}；代理保持停止",
        )
        return jsonify({"ok": True, **_service_status()})


@app.post("/api/service/start")
def api_service_start():
    # 与配置保存共用状态锁，保证检查配置、初始化客户端池、启动服务不可交错。
    with _SERVICE_STATE_LOCK:
        keys = _configured_api_keys()
        if not keys:
            return jsonify({"error": "未配置 NVIDIA API Key"}), 400
        capacity_error = _capacity_error(keys, NVIDIA_THREADS_PER_KEY)
        if capacity_error:
            return jsonify({"error": capacity_error}), 400
        _init_client_pool(keys)
        _set_service_running(True)
        _runtime_log("INFO", "service.started", f"代理已启动：keys={len(keys)}, model={NVIDIA_MODEL}")
        return jsonify({"ok": True, **_service_status()})


@app.post("/api/service/stop")
def api_service_stop():
    with _SERVICE_STATE_LOCK:
        _set_service_running(False)
        _runtime_log("INFO", "service.stopped", "代理已停止；管理控制台继续运行")
        return jsonify({"ok": True, **_service_status()})


@app.get("/api/models")
def api_models():
    """从全部 Key 聚合上游 NVIDIA 模型列表，不对返回模型做白名单过滤。"""
    log_id = f"models_{uuid.uuid4().hex[:8]}"
    keys = _configured_api_keys()
    if not keys:
        return jsonify({"error": "请先配置 NVIDIA API Key"}), 400
    _runtime_log("REQUEST", "models.list", f"开始从 {len(keys)} 个 Key 聚合模型列表", log_id)
    aggregation_deadline = time.monotonic() + MODEL_LIST_TOTAL_TIMEOUT

    pool = _get_client_pool()
    pool_keys = [entry["key"] for entry in pool]
    if pool_keys != keys:
        pool = _init_client_pool(keys)

    model_ids = set()
    successful_keys = []
    errors = []
    for index, entry in enumerate(pool, start=1):
        if time.monotonic() >= aggregation_deadline:
            errors.append(f"模型列表聚合超过 {MODEL_LIST_TOTAL_TIMEOUT:g} 秒总超时")
            break
        slot = None
        try:
            # 模型查询也占用对应 Key 的并发槽位，确保每个 Key 不超过配置上限。
            # 控制台查询不等待繁忙 Key，避免一个满载 Key 阻塞其他 Key；总截止时间
            # 仍传入槽位获取逻辑，防止未来改为阻塞模式时失去截止约束。
            slot = _acquire_client_slot(
                preferred_entry=entry, blocking=False, deadline=aggregation_deadline,
            )
            if slot is None:
                errors.append(f"Key {index}: 并发槽位已满或聚合已超时")
                continue
            remaining = aggregation_deadline - time.monotonic()
            if remaining <= 0:
                errors.append(f"Key {index}: 模型列表聚合已超时")
                continue
            request_timeout = min(MODEL_LIST_REQUEST_TIMEOUT, remaining)
            client = slot["client"]
            if hasattr(client, "with_options"):
                # 禁止 SDK 为不可达上游自动重试，否则单次 timeout 可能被重复叠加，
                # 使整个聚合请求超过截止时间。
                page = client.with_options(
                    timeout=request_timeout, max_retries=0,
                ).models.list()
            else:
                # 兼容较旧的 OpenAI SDK 或测试替身。
                page = client.models.list(timeout=request_timeout)
            items = page.auto_paging_iter() if hasattr(page, "auto_paging_iter") else getattr(page, "data", page)
            key_model_ids = set()
            for item in items:
                if time.monotonic() >= aggregation_deadline:
                    raise TimeoutError(
                        f"模型列表聚合超过 {MODEL_LIST_TOTAL_TIMEOUT:g} 秒总超时"
                    )
                model_id = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
                if model_id:
                    key_model_ids.add(str(model_id))
            if key_model_ids:
                model_ids.update(key_model_ids)
                successful_keys.append(index)
                _runtime_log("INFO", "models.key.done", f"Key {index} 返回 {len(key_model_ids)} 个模型", log_id)
            else:
                errors.append(f"Key {index} 返回空模型列表")
        except Exception as exc:
            errors.append(f"Key {index}: {type(exc).__name__}: {exc}")
            _runtime_log("ERROR", "models.key.failed", errors[-1], log_id)
        finally:
            if slot is not None:
                _release_client_slot(slot)

    if model_ids:
        _runtime_log("DONE", "models.completed", f"聚合完成，共 {len(model_ids)} 个模型", log_id)
        return jsonify({
            "models": sorted(model_ids, key=str.lower),
            "successful_keys": successful_keys,
            "count": len(model_ids),
            "errors": errors,
        })
    _runtime_log("ERROR", "models.failed", "所有 Key 均无法获取模型列表", log_id)
    return jsonify({"error": "无法获取上游模型列表", "details": errors}), 502


@app.post("/api/models/test")
def api_model_test():
    """使用用户输入的文字测试模型，并将上游流式返回过程写入运行日志。"""
    payload = request.get_json(silent=True) or {}
    model = str(payload.get("model") or "").strip()
    prompt = str(payload.get("prompt") or "你好，请简短介绍一下你自己。").strip()
    test_id = f"test_{uuid.uuid4().hex[:10]}"
    if not model:
        return jsonify({"error": "模型名称不能为空"}), 400
    if len(model) > 300:
        return jsonify({"error": "模型名称过长"}), 400
    if not prompt:
        return jsonify({"error": "测试文字不能为空"}), 400
    if len(prompt) > 50000:
        return jsonify({"error": "测试文字过长，最多 50000 个字符"}), 400

    keys = _configured_api_keys()
    if not keys:
        return jsonify({"error": "请先配置 NVIDIA API Key"}), 400
    pool = _get_client_pool()
    if [entry["key"] for entry in pool] != keys:
        pool = _init_client_pool(keys)

    _runtime_log(
        "REQUEST", "model.test.request",
        lambda: f"model={model}\nprompt={prompt}", test_id,
    )
    errors = []
    for index, entry in enumerate(pool, start=1):
        slot = None
        try:
            # 每个 Key 都只做非阻塞尝试，避免首个 Key 满载时阻塞后续空闲 Key。
            slot = _acquire_client_slot(preferred_entry=entry, blocking=False)
            if slot is None:
                errors.append(f"Key {index}: 并发槽位已满")
                _runtime_log("WARN", "model.test.key.busy", errors[-1], test_id)
                continue
            _runtime_log("INFO", "model.test.upstream", f"尝试 Key {index}", test_id)
            stream = slot["client"].chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                timeout=90,
            )
            response_parts = []
            for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                reasoning = getattr(delta, "reasoning_content", None) if delta is not None else None
                if reasoning:
                    _runtime_log("REASONING", "model.test.reasoning.delta", reasoning, test_id)
                content = getattr(delta, "content", None) if delta is not None else None
                if content:
                    response_parts.append(content)
                    _runtime_log("DELTA", "model.test.delta", content, test_id)
            response_text = "".join(response_parts).strip()
            _runtime_log(
                "DONE", "model.test.completed",
                lambda: f"Key {index} 请求成功\nresponse={response_text}", test_id,
            )
            return jsonify({
                "ok": True,
                "model": model,
                "key_index": index,
                "response": response_text,
                "request_id": test_id,
            })
        except Exception as exc:
            errors.append(f"Key {index}: {type(exc).__name__}: {exc}")
            _runtime_log("ERROR", "model.test.key.failed", errors[-1], test_id)
        finally:
            if slot is not None:
                _release_client_slot(slot)

    _runtime_log("ERROR", "model.test.failed", "所有 Key 尝试失败", test_id)
    return jsonify({
        "error": f"模型 {model} 尝试失败",
        "details": errors,
        "request_id": test_id,
    }), 502


# ---- CORS ----
@app.after_request
def add_cors(resp):
    # 只有代理协议端点允许跨域；控制台页面和管理 API 均不开放 CORS。
    if request.path in (
        "/responses", "/v1/responses", "/v1/chat/completions",
        "/messages", "/v1/messages",
    ):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    # 控制台包含启动、停止和日志清理等高影响操作，禁止被第三方页面嵌入后
    # 通过点击劫持诱导本机用户操作。两种响应头同时设置以兼容旧浏览器。
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Content-Security-Policy"] = "frame-ancestors 'none'"
    return resp


# ---- 路由处理 ----
def _anthropic_error(message, status=400, error_type="invalid_request_error"):
    return jsonify({
        "type": "error",
        "error": {"type": error_type, "message": str(message)},
    }), status


def _anthropic_upstream_kwargs(
    req_data, messages, tools, tool_choice, effective_model, stream,
    parallel_tool_calls=None,
):
    """构建 Anthropic -> NVIDIA OpenAI Chat Completions 的参数。"""
    kwargs = {"model": effective_model, "messages": messages, "stream": stream}
    # Claude Code 会发送 max_tokens；NVIDIA OpenAI 兼容端点支持这些常用字段。
    for source, target in (
        ("max_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("stop_sequences", "stop"),
    ):
        value = req_data.get(source)
        if source == "stop_sequences":
            value = _anthropic_stop_sequences(value)
            if not value:
                continue
        if value is not None:
            kwargs[target] = value
    if tools:
        kwargs["tools"] = tools
        if tool_choice != "auto":
            kwargs["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
    return kwargs


def _anthropic_message_from_completion(
    completion, effective_model, messages, stop_sequences=None,
):
    """将 OpenAI 非流式 completion 转为 Anthropic Message JSON。"""
    choices = _obj_get(completion, "choices", []) or []
    choice = choices[0] if choices else {}
    message = _obj_get(choice, "message", {}) or {}
    finish_reason = _obj_get(choice, "finish_reason", "stop")
    raw_tool_calls = _obj_get(message, "tool_calls", []) or []
    text = _obj_get(message, "content", "") or ""
    blocks = []
    if text:
        blocks.append({"type": "text", "text": text})
    # finish_reason=length 时工具参数可能是不完整 JSON。省略该 tool_use，
    # 并通过 stop_reason=max_tokens 明确告知调用方不得执行它。
    tool_calls = [] if finish_reason == "length" else raw_tool_calls
    for tool_call in tool_calls:
        function = _obj_get(tool_call, "function", {}) or {}
        arguments = _obj_get(function, "arguments", "") or "{}"
        try:
            parsed_input = json.loads(arguments)
        except (TypeError, ValueError):
            parsed_input = {}
        blocks.append({
            "type": "tool_use",
            "id": _obj_get(tool_call, "id", "") or f"toolu_{uuid.uuid4().hex[:12]}",
            "name": _obj_get(function, "name", "") or "",
            "input": parsed_input,
        })
    stop_reason, stop_sequence = _anthropic_stop_result(
        finish_reason, stop_sequences, text, bool(tool_calls),
    )
    usage = _obj_get(completion, "usage", {}) or {}
    input_tokens = _obj_get(usage, "prompt_tokens", 0) or _estimate_tokens(json.dumps(messages, ensure_ascii=False))
    output_basis = text + "".join(
        _obj_get(_obj_get(tc, "function", {}) or {}, "arguments", "") or ""
        for tc in tool_calls
    )
    output_tokens = _obj_get(usage, "completion_tokens", 0) or _estimate_tokens(output_basis)
    return {
        "id": f"msg_{uuid.uuid4().hex[:20]}",
        "type": "message",
        "role": "assistant",
        "content": blocks,
        "model": effective_model,
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _make_anthropic_response():
    """Anthropic Messages API 兼容入口，供 Claude Code/Claude CLI 使用。"""
    if request.method == "OPTIONS":
        return Response()
    with _SERVICE_STATE_LOCK:
        service_running = _SERVICE_RUNNING
    if not service_running:
        return _anthropic_error("代理服务当前已停止，请先通过网页控制台启动服务", 503, "api_error")

    req_data = request.get_json(silent=True)
    if not isinstance(req_data, dict):
        return _anthropic_error("请求体必须是 JSON 对象")
    raw_messages = req_data.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        return _anthropic_error("messages 必须是非空数组")

    messages = _anthropic_messages(req_data)
    tools = _anthropic_tools(req_data.get("tools", []))
    tool_choice = _anthropic_tool_choice(req_data.get("tool_choice"))
    parallel_tool_calls = _anthropic_parallel_tool_calls(req_data.get("tool_choice"))
    stop_sequences = _anthropic_stop_sequences(req_data.get("stop_sequences"))
    effective_model = _anthropic_effective_model(req_data.get("model"))
    response_id = f"msg_{uuid.uuid4().hex[:20]}"
    stream_requested = req_data.get("stream") is True
    _runtime_log(
        "REQUEST", "anthropic.request",
        lambda: json.dumps({
            "path": request.path,
            "model": effective_model,
            "messages": messages,
            "tools_count": len(tools),
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
            "stop_sequences": stop_sequences,
            "stream": stream_requested,
        }, ensure_ascii=False, default=str),
        response_id,
    )

    kwargs = _anthropic_upstream_kwargs(
        req_data, messages, tools, tool_choice, effective_model, stream_requested,
        parallel_tool_calls,
    )

    if not stream_requested:
        slot = None
        try:
            slot = _acquire_client_slot(blocking=False)
            if slot is None:
                return _anthropic_error("所有 NVIDIA API Key 的并发槽位均已占满，请稍后重试", 429, "rate_limit_error")
            _runtime_log("INFO", "anthropic.upstream.start", f"Key {slot['index']} -> model={effective_model}", response_id)
            completion = slot["client"].chat.completions.create(**kwargs)
            result = _anthropic_message_from_completion(
                completion, effective_model, messages, stop_sequences,
            )
            _runtime_log("DONE", "anthropic.completed", lambda: json.dumps(result, ensure_ascii=False), response_id)
            return jsonify(result)
        except Exception as exc:
            err_msg = f"NVIDIA API error: {type(exc).__name__}: {exc}"
            _runtime_log("ERROR", "anthropic.failed", err_msg, response_id)
            return _anthropic_error(err_msg, 502, "api_error")
        finally:
            if slot is not None:
                _release_client_slot(slot)

    def generate_anthropic():
        # Claude Code 依赖标准 Anthropic SSE 事件顺序，而不是 Responses API 事件。
        def sse(event, payload):
            return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        input_tokens = _estimate_tokens(json.dumps(messages, ensure_ascii=False))
        output_tokens = 0
        full_text = ""
        tool_acc = {}
        block_indices = {}
        block_order = []
        finish_reason = "stop"
        slot = None

        yield sse("message_start", {
            "type": "message_start",
            "message": {
                "id": response_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": effective_model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        })
        try:
            slot = _acquire_client_slot(blocking=False)
            if slot is None:
                raise RuntimeError("所有 NVIDIA API Key 的并发槽位均已占满，请稍后重试")
            _runtime_log("INFO", "anthropic.upstream.start", f"Key {slot['index']} -> model={effective_model}", response_id)
            stream = slot["client"].chat.completions.create(**kwargs)
            for chunk in stream:
                usage = _obj_get(chunk, "usage")
                if usage:
                    input_tokens = _obj_get(usage, "prompt_tokens", input_tokens) or input_tokens
                    output_tokens = _obj_get(usage, "completion_tokens", output_tokens) or output_tokens
                choices = _obj_get(chunk, "choices", []) or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = _obj_get(choice, "finish_reason", finish_reason) or finish_reason
                delta = _obj_get(choice, "delta", {}) or {}
                content = _obj_get(delta, "content", "") or ""
                if not isinstance(content, str):
                    content = _anthropic_text(content) if isinstance(content, list) else str(content)
                if content:
                    key = ("text", 0)
                    if key not in block_indices:
                        block_indices[key] = len(block_order)
                        block_order.append(key)
                        yield sse("content_block_start", {
                            "type": "content_block_start", "index": block_indices[key],
                            "content_block": {"type": "text", "text": ""},
                        })
                    full_text += content
                    _runtime_log("DELTA", "anthropic.response.delta", content, response_id)
                    yield sse("content_block_delta", {
                        "type": "content_block_delta", "index": block_indices[key],
                        "delta": {"type": "text_delta", "text": content},
                    })

                for tool_call in _obj_get(delta, "tool_calls", []) or []:
                    index = _obj_get(tool_call, "index", 0)
                    key = ("tool", index)
                    function = _obj_get(tool_call, "function", {}) or {}
                    acc = tool_acc.setdefault(index, {
                        "id": "", "name": "", "arguments": "",
                    })
                    call_id = _obj_get(tool_call, "id")
                    name = _obj_get(function, "name")
                    arguments = _obj_get(function, "arguments", "") or ""
                    if call_id:
                        acc["id"] = call_id
                    if name:
                        acc["name"] = name
                    if key not in block_indices:
                        if not acc["id"]:
                            acc["id"] = f"toolu_{uuid.uuid4().hex[:12]}"
                        block_indices[key] = len(block_order)
                        block_order.append(key)
                        yield sse("content_block_start", {
                            "type": "content_block_start", "index": block_indices[key],
                            "content_block": {
                                "type": "tool_use", "id": acc["id"],
                                "name": acc["name"], "input": {},
                            },
                        })
                    if arguments:
                        acc["arguments"] += arguments
                        _runtime_log("TOOL", "anthropic.tool.arguments.delta", arguments, response_id)
                        yield sse("content_block_delta", {
                            "type": "content_block_delta", "index": block_indices[key],
                            "delta": {"type": "input_json_delta", "partial_json": arguments},
                        })

            for key in block_order:
                yield sse("content_block_stop", {
                    "type": "content_block_stop", "index": block_indices[key],
                })
            stop_reason, stop_sequence = _anthropic_stop_result(
                finish_reason, stop_sequences, full_text, bool(tool_acc),
            )
            if not output_tokens:
                output_tokens = _estimate_tokens(full_text + "".join(v["arguments"] for v in tool_acc.values()))
            yield sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
                "usage": {"output_tokens": output_tokens},
            })
            _runtime_log("DONE", "anthropic.completed", lambda: json.dumps({
                "model": effective_model, "text": full_text, "tool_calls": tool_acc,
            }, ensure_ascii=False, default=str), response_id)
            yield sse("message_stop", {"type": "message_stop"})
        except Exception as exc:
            err_msg = f"NVIDIA API error: {type(exc).__name__}: {exc}"
            _runtime_log("ERROR", "anthropic.failed", err_msg, response_id)
            yield sse("error", {
                "type": "error",
                "error": {"type": "api_error", "message": err_msg},
            })
        finally:
            if slot is not None:
                _release_client_slot(slot)

    return Response(
        generate_anthropic(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _make_response():
    if request.method == "OPTIONS":
        return Response()
    with _SERVICE_STATE_LOCK:
        service_running = _SERVICE_RUNNING
    if not service_running:
        _runtime_log("WARN", "proxy.rejected", f"服务已停止，拒绝请求：{request.path}")
        return jsonify({
            "error": "代理服务当前已停止",
            "message": "请先通过网页控制台启动服务",
        }), 503

    req_data = request.get_json(silent=True) or {}
    messages, tools, tool_choice = extract_messages(req_data)
    effective_model = req_data.get("model") or NVIDIA_MODEL
    response_id = f"resp_{uuid.uuid4().hex[:12]}"
    _runtime_log(
        "REQUEST", "proxy.request",
        lambda: json.dumps({
            "path": request.path,
            "model": effective_model,
            "messages": messages,
            "tools_count": len(tools),
            "tool_choice": tool_choice,
        }, ensure_ascii=False, default=str),
        response_id,
    )

    if NVIDIA_DEBUG:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n--- [{__import__('datetime').datetime.now()}] ---\n")
            f.write(f"Messages:\n{json.dumps(messages, indent=2, ensure_ascii=False)}\n")
            if tools:
                f.write(f"Tools count: {len(tools)}\n")

    def generate():
        if not messages:
            _runtime_log("DONE", "proxy.completed", "空消息请求已直接完成", response_id)
            yield "event: response.completed\n"
            yield "data: " + json.dumps({
                "type": "response.completed",
                "response": {
                    "id": response_id, "object": "response",
                    "status": "completed", "model": effective_model,
                    "output": [], "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                },
            }, ensure_ascii=False) + "\n\n"
            return

        # response.created
        yield "event: response.created\n"
        yield "data: " + json.dumps({
            "type": "response.created",
            "response": {
                "id": response_id, "object": "response",
                "status": "in_progress", "model": effective_model,
                "output": [], "usage": None,
            },
        }, ensure_ascii=False) + "\n\n"

        # response.in_progress
        yield "event: response.in_progress\n"
        yield "data: " + json.dumps({
            "type": "response.in_progress",
            "response": {
                "id": response_id, "object": "response",
                "status": "in_progress", "model": effective_model,
                "output": [], "usage": None,
            },
        }, ensure_ascii=False) + "\n\n"

        # 构建请求参数
        kwargs = {
            "model": effective_model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            if tool_choice != "auto":
                kwargs["tool_choice"] = tool_choice

        # 状态跟踪
        text_item_id = f"item_{uuid.uuid4().hex[:12]}"
        full_text = ""
        has_text = False
        text_started = False
        tool_calls_acc = {}
        input_tokens = 0
        output_tokens = 0
        seq = 0
        client_entry = None

        try:
            # 不让额外代理请求阻塞占满预留给管理 API 的 Web worker。
            client_entry = _acquire_client_slot(blocking=False)
            if client_entry is None:
                raise RuntimeError("所有 NVIDIA API Key 的并发槽位均已占满，请稍后重试")
            client = client_entry["client"]
            _runtime_log(
                "INFO", "proxy.upstream.start",
                f"Key {client_entry['index']} -> model={effective_model}", response_id,
            )
            stream = client.chat.completions.create(**kwargs)

            for chunk in stream:
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens or 0
                    output_tokens = chunk.usage.completion_tokens or 0

                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue

                delta = choice.delta

                reasoning_content = getattr(delta, "reasoning_content", None)
                if reasoning_content:
                    _runtime_log(
                        "REASONING", "proxy.reasoning.delta",
                        reasoning_content, response_id,
                    )

                # 文本内容
                content = delta.content
                if content:
                    _runtime_log("DELTA", "proxy.response.delta", content, response_id)
                    if not text_started:
                        text_started = True
                        has_text = True
                        yield "event: response.output_item.added\n"
                        yield "data: " + json.dumps({
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": {
                                "id": text_item_id, "type": "message",
                                "status": "in_progress", "role": "assistant",
                                "content": [],
                            },
                        }, ensure_ascii=False) + "\n\n"
                        yield "event: response.content_part.added\n"
                        yield "data: " + json.dumps({
                            "type": "response.content_part.added",
                            "item_id": text_item_id,
                            "output_index": 0, "content_index": 0,
                            "part": {"type": "text", "text": ""},
                        }, ensure_ascii=False) + "\n\n"

                    full_text += content
                    seq += 1
                    yield "event: response.output_text.delta\n"
                    yield "data: " + json.dumps({
                        "type": "response.output_text.delta",
                        "delta": content,
                        "item_id": text_item_id,
                        "output_index": 0, "content_index": 0,
                        "sequence_number": seq,
                    }, ensure_ascii=False) + "\n\n"

                # 工具调用
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_acc:
                            item_id = f"item_{uuid.uuid4().hex[:12]}"
                            tool_calls_acc[idx] = {
                                "id": tc.id or "",
                                "name": "",
                                "arguments": "",
                                "item_id": item_id,
                                "started": False,
                            }

                        acc = tool_calls_acc[idx]
                        if tc.function and tc.function.name:
                            acc["name"] = tc.function.name
                            _runtime_log(
                                "TOOL", "proxy.tool.name",
                                f"tool={tc.function.name}", response_id,
                            )
                        if tc.id:
                            acc["id"] = tc.id
                        if tc.function and tc.function.arguments:
                            acc["arguments"] += tc.function.arguments
                            _runtime_log(
                                "TOOL", "proxy.tool.arguments.delta",
                                tc.function.arguments, response_id,
                            )
                            out_idx = (1 if has_text else 0) + sorted(tool_calls_acc.keys()).index(idx)

                            if not acc["started"]:
                                acc["started"] = True
                                yield "event: response.output_item.added\n"
                                yield "data: " + json.dumps({
                                    "type": "response.output_item.added",
                                    "output_index": out_idx,
                                    "item": {
                                        "id": acc["item_id"],
                                        "type": "function_call",
                                        "status": "in_progress",
                                        "call_id": acc["id"],
                                        "name": acc["name"],
                                        "arguments": "",
                                    },
                                }, ensure_ascii=False) + "\n\n"

                            yield "event: response.function_call_arguments.delta\n"
                            yield "data: " + json.dumps({
                                "type": "response.function_call_arguments.delta",
                                "item_id": acc["item_id"],
                                "output_index": out_idx,
                                "delta": tc.function.arguments,
                            }, ensure_ascii=False) + "\n\n"

            # 文本完成
            if has_text:
                yield "event: response.output_text.done\n"
                yield "data: " + json.dumps({
                    "type": "response.output_text.done",
                    "text": full_text, "item_id": text_item_id,
                    "output_index": 0, "content_index": 0,
                }, ensure_ascii=False) + "\n\n"
                yield "event: response.content_part.done\n"
                yield "data: " + json.dumps({
                    "type": "response.content_part.done",
                    "item_id": text_item_id,
                    "output_index": 0, "content_index": 0,
                    "part": {"type": "text", "text": full_text},
                }, ensure_ascii=False) + "\n\n"
                output_item_text = {
                    "id": text_item_id, "type": "message",
                    "status": "completed", "role": "assistant",
                    "content": [{"type": "text", "text": full_text}],
                }
                yield "event: response.output_item.done\n"
                yield "data: " + json.dumps({
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": output_item_text,
                }, ensure_ascii=False) + "\n\n"

            # 工具调用完成
            output_items = []
            if has_text:
                output_items.append(output_item_text)

            for idx in sorted(tool_calls_acc.keys()):
                acc = tool_calls_acc[idx]
                out_idx = (1 if has_text else 0) + sorted(tool_calls_acc.keys()).index(idx)

                yield "event: response.function_call_arguments.done\n"
                yield "data: " + json.dumps({
                    "type": "response.function_call_arguments.done",
                    "item_id": acc["item_id"],
                    "output_index": out_idx,
                    "arguments": acc["arguments"],
                }, ensure_ascii=False) + "\n\n"

                func_item = {
                    "id": acc["item_id"],
                    "type": "function_call",
                    "status": "completed",
                    "call_id": acc["id"],
                    "name": acc["name"],
                    "arguments": acc["arguments"],
                }
                yield "event: response.output_item.done\n"
                yield "data: " + json.dumps({
                    "type": "response.output_item.done",
                    "output_index": out_idx,
                    "item": func_item,
                }, ensure_ascii=False) + "\n\n"

                output_items.append(func_item)

            # response.completed
            _runtime_log(
                "DONE", "proxy.completed",
                lambda: json.dumps({
                    "model": effective_model,
                    "key_index": client_entry["index"],
                    "input_tokens": input_tokens or _estimate_tokens(json.dumps(messages)),
                    "output_tokens": output_tokens or _estimate_tokens(full_text),
                    "text": full_text,
                    "tool_calls": [tool_calls_acc[idx] for idx in sorted(tool_calls_acc)],
                }, ensure_ascii=False, default=str),
                response_id,
            )
            yield "event: response.completed\n"
            yield "data: " + json.dumps({
                "type": "response.completed",
                "response": {
                    "id": response_id, "object": "response",
                    "status": "completed", "model": effective_model,
                    "output": output_items,
                    "usage": {
                        "input_tokens": input_tokens or _estimate_tokens(json.dumps(messages)),
                        "output_tokens": output_tokens or _estimate_tokens(full_text),
                        "total_tokens": (input_tokens or _estimate_tokens(json.dumps(messages)))
                                        + (output_tokens or _estimate_tokens(full_text)),
                    },
                },
            }, ensure_ascii=False) + "\n\n"

        except Exception as e:
            err_msg = f"NVIDIA API error: {type(e).__name__}: {e}"
            _runtime_log("ERROR", "proxy.failed", err_msg, response_id)
            if NVIDIA_DEBUG:
                with open(DEBUG_LOG, "a", encoding="utf-8") as f:
                    f.write(f"ERROR: {err_msg}\n")
            yield "event: response.failed\n"
            yield "data: " + json.dumps({
                "type": "response.failed",
                "response": {
                    "id": response_id, "object": "response",
                    "status": "failed", "model": effective_model,
                    "error": {"message": err_msg, "type": "upstream_error"},
                    "output": [], "usage": None,
                },
            }, ensure_ascii=False) + "\n\n"
        finally:
            if client_entry is not None:
                _release_client_slot(client_entry)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---- 注册路由 ----
app.add_url_rule("/responses", "responses", _make_response, methods=["POST", "OPTIONS"])
app.add_url_rule("/v1/responses", "v1_responses", _make_response, methods=["POST", "OPTIONS"])
app.add_url_rule("/v1/chat/completions", "v1_chat", _make_response, methods=["POST", "OPTIONS"])
app.add_url_rule("/messages", "messages", _make_anthropic_response, methods=["POST", "OPTIONS"])
app.add_url_rule("/v1/messages", "v1_messages", _make_anthropic_response, methods=["POST", "OPTIONS"])


if __name__ == "__main__":
    # 启动 Web 控制台本身不再强制要求 Key；可直接在网页中配置并启动代理。
    keys = _configured_api_keys()
    source = "环境变量/.nvidia_env" if keys else "未配置"
    capacity_error = _capacity_error(keys, NVIDIA_THREADS_PER_KEY)
    if keys and not capacity_error:
        _init_client_pool(keys)
        _set_service_running(True)
    else:
        _set_service_running(False)
    total_threads = len(keys) * NVIDIA_THREADS_PER_KEY if not capacity_error else 0
    # 固定按后端允许的最大代理并发预留 worker，并额外保留管理线程。
    server_threads = WEB_SERVER_THREADS
    _runtime_log(
        "INFO", "process.started",
        f"Web 控制台启动；keys={len(keys)}, "
        f"proxy_running={bool(keys) and not capacity_error}, model={NVIDIA_MODEL}",
    )

    from waitress import serve
    print("codex_nvidia_proxy starting ...")
    print(f"   Web UI:   http://127.0.0.1:5000/")
    print(f"   Endpoint: http://127.0.0.1:5000")
    print(f"   Model:    {NVIDIA_MODEL}")
    print(f"   Keys:     {len(keys)} ({source})")
    print(f"   Threads:  {NVIDIA_THREADS_PER_KEY} per key, {total_threads} upstream slots")
    print(f"   Workers:  {server_threads} web threads")
    if capacity_error:
        print(f"   Config:   ERROR: {capacity_error}")
    print(f"   Debug:    {'ON' if NVIDIA_DEBUG else 'OFF'}")
    print(
        "   Routes:   / (UI), /api/*, /responses, /v1/responses, "
        "/v1/chat/completions, /v1/messages (Anthropic)"
    )
    serve(app, host="127.0.0.1", port=5000, threads=server_threads)
