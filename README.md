# Codex NVIDIA Proxy

一个轻量级的 NVIDIA API 代理服务，将 NVIDIA NIM API 转换为兼容 OpenAI Chat Completions、Responses API 以及 Anthropic Messages API 的格式，方便接入 Claude Code、Cursor 等开发工具。

## 功能特性

- 支持 `/responses` 接口（OpenAI Responses API）
- 支持 `/v1/responses` 和 `/v1/chat/completions` 接口
- 支持 `/v1/messages`（Anthropic Messages API，Claude Code 原生协议）
- 自动处理函数调用（Function Calling）
- 支持工具调用（Tools）
- CORS 跨域支持
- SSE 流式响应
- 灵活的 API Key 配置方式
- 支持多 API Key 轮询及按 Key 限制并发
- 调试日志功能

## 环境要求

- Python 3.8+
- NVIDIA API Key（从 [build.nvidia.com](https://build.nvidia.com/) 获取）

## 安装

1. 克隆项目
```bash
git clone https://github.com/yourusername/codex-nvidia-proxy.git
cd codex-nvidia-proxy
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置 API Key

方式一：创建 `.nvidia_env` 文件
```bash
echo NVIDIA_API_KEY=your_api_key_here > .nvidia_env
echo NVIDIA_MODEL=nvidia/llama-3.1-nemotron-70b-instruct >> .nvidia_env
```

如需使用多个 NVIDIA API Key，可改为配置 `NVIDIA_API_KEYS`（用逗号、分号或换行分隔）：

```bash
NVIDIA_API_KEYS=key_1,key_2,key_3
NVIDIA_THREADS_PER_KEY=4
```

Windows PowerShell：

```powershell
$env:NVIDIA_API_KEYS="key_1,key_2,key_3"
$env:NVIDIA_THREADS_PER_KEY="4"
```

`NVIDIA_API_KEYS` 优先于单 Key 配置 `NVIDIA_API_KEY`。服务会为每个 Key 创建一个独立的 OpenAI 客户端，并在 Key 之间轮询分配请求；每个 Key 默认最多同时处理 4 个请求。每 Key 并发范围为 1～64，所有 Key 的总并发槽位最多为 64；Web 服务固定使用 68 个工作线程，并为控制台管理请求预留容量。

方式二：设置系统环境变量
```bash
# Linux/macOS
export NVIDIA_API_KEY=your_api_key_here
export NVIDIA_MODEL=nvidia/llama-3.1-nemotron-70b-instruct

# Windows PowerShell
$env:NVIDIA_API_KEY="your_api_key_here"
$env:NVIDIA_MODEL="nvidia/llama-3.1-nemotron-70b-instruct"
```

## 使用方法

### 启动服务

```bash
python codex_nvidia_proxy.py
```

服务启动后显示（即使尚未配置 Key，也会先启动网页控制台）：
```
codex_nvidia_proxy starting ...
   Web UI:   http://127.0.0.1:5000/
   Endpoint: http://127.0.0.1:5000
   Model:    nvidia/llama-3.1-nemotron-70b-instruct
   Keys:     1 (.nvidia_env (已保存))
   Threads:  4 per key, 4 upstream slots
   Workers:  68 web threads
   Debug:    OFF
   Routes:   / (UI), /api/*, /responses, /v1/responses, /v1/chat/completions, /v1/messages
```

### 网页控制台

浏览器打开 `http://127.0.0.1:5000/`，可以：

- 填写一个或多个 NVIDIA API Key，并保存到 `.nvidia_env`
- 设置每个 Key 的并发数（默认 4）
- 启动/停止代理服务（停止时网页控制台仍可访问）
- 调用全部 Key 的上游 `/models` 接口，聚合并按厂商分类展示模型
- 在独立的“模型库”Tab 中按厂商分类、搜索模型，并选择默认模型
- 在独立的“聊天测试”Tab 中对列表中的任意模型或手工输入的模型发送测试消息并查看完整回复
- 在“运行日志”Tab 实时查看请求内容、Key 分配、上游增量返回、工具调用及错误

网页控制台仅绑定 `127.0.0.1`，不要将该端口暴露到公网；Key 会以明文保存到本地 `.nvidia_env`。
Key 输入框留空时仅沿用当前配置，来自系统环境变量的 Key 不会因此写入 `.nvidia_env`。
管理写接口使用控制台页面生成的 `X-CSRF-Token` 请求头，跨站表单无法直接启动、停止代理或改写配置。
模型名称没有本地白名单限制：即使上游 `/models` 没有返回，也可以手工填写并尝试请求。
详细运行日志默认关闭，可在“运行日志”Tab 点击按钮随时开启或关闭。关闭后不再记录上游 delta，也停止定时拉取日志，从而避免高并发时产生额外开销。日志只保存在当前进程内存中，最多保留最近 2000 条；重启进程或在网页中清空后即消失。日志包含请求和模型回复正文，但不会记录完整 NVIDIA API Key。

### 配置开发工具

#### Claude Code / Claude CLI
```bash
# Claude Code 使用 Anthropic Messages API（不是 OpenAI /chat/completions）。
# Windows PowerShell：
$env:ANTHROPIC_BASE_URL="http://127.0.0.1:5000"
$env:ANTHROPIC_API_KEY="any_key"  # 本地代理仅用于兼容，任意非空值即可
$env:ANTHROPIC_MODEL="nvidia/llama-3.1-nemotron-70b-instruct"  # 可选：让 Claude Code 在请求中指定 NIM 模型
# Linux/macOS：export ANTHROPIC_BASE_URL=...; export ANTHROPIC_API_KEY=...
claude
```

也可以直接使用 `claude --model nvidia/llama-3.1-nemotron-70b-instruct`。代理会优先使用请求中的模型名；但 Claude Code 默认发送的 `claude-*` 模型名会自动映射到 `NVIDIA_MODEL`，因为 NVIDIA 上游不认识 Claude 的模型 ID。

#### Cursor
在 Cursor 设置中配置：
- API Base URL: `http://127.0.0.1:5000`
- API Key: `any_key`

### API 端点

| 端点 | 说明 |
|------|------|
| `GET /` | 网页控制台 |
| `GET /api/status` | 查询服务状态 |
| `GET /api/logs` | 增量获取内存运行日志 |
| `POST /api/logs/clear` | 清空内存运行日志 |
| `POST /api/logs/toggle` | 开启或关闭详细运行日志 |
| `POST /api/config` | 保存 Key、模型和并发配置 |
| `POST /api/service/start` | 启动代理 |
| `POST /api/service/stop` | 停止代理（控制台继续运行） |
| `GET /api/models` | 获取上游支持模型 |
| `POST /api/models/test` | 使用自定义文字和任一可用 Key 尝试请求指定模型 |
| `POST /responses` | OpenAI Responses API |
| `POST /v1/responses` | OpenAI Responses API (v1) |
| `POST /v1/chat/completions` | OpenAI Chat Completions API |
| `POST /v1/messages` | Anthropic Messages API（Claude Code） |

## 配置选项

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `NVIDIA_API_KEY` | 单个 NVIDIA API Key（未配置多 Key 时必填） | 可选 |
| `NVIDIA_API_KEYS` | 多个 NVIDIA API Key，逗号/分号/换行分隔；优先于单 Key | 可选 |
| `NVIDIA_THREADS_PER_KEY` | 每个 Key 的最大并发请求数（1～64，所有 Key 合计不超过 64） | `4` |
| `NVIDIA_MODEL` | 使用的模型 | `nvidia/llama-3.1-nemotron-70b-instruct` |
| `NVIDIA_BASE_URL` | API Base URL | `https://integrate.api.nvidia.com/v1` |
| `NVIDIA_DEBUG` | 开启调试日志 | `0` |
| `NVIDIA_RUNTIME_LOG` | 进程启动时是否默认开启详细内存日志 | `0` |

### 调试模式

开启调试模式后会生成 `nvidia_proxy_debug.log` 日志文件：

```bash
# Linux/macOS
export NVIDIA_DEBUG=1

# Windows PowerShell
$env:NVIDIA_DEBUG="1"
```

## 快速测试

使用 curl 测试服务：

```bash
curl -X POST http://127.0.0.1:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer any_key" \
  -d '{
    "model": "nvidia/llama-3.1-nemotron-70b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## 项目结构

```
codex-nvidia-proxy/
├── codex_nvidia_proxy.py   # 主程序
├── requirements.txt        # 依赖列表
├── .nvidia_env.example     # 配置文件示例
├── README.md               # 本文件
└── LICENSE                # MIT 许可证
```

## 可用模型

推荐在 [build.nvidia.com](https://build.nvidia.com/) 查看可用模型，热门模型包括：

- `nvidia/llama-3.1-nemotron-70b-instruct`
- `nvidia/llama-3.3-nemotron-70b-instruct`
- `nvidia/nemotron-4-340b-instruct`
- `mistralai/mixtral-8x7b-instruct-v0.1`
- `google/gemma-2-27b-it`

## 常见问题

### Q: 获取 NVIDIA API Key？
访问 [build.nvidia.com](https://build.nvidia.com/)，登录后点击任意模型，选择 "Get API Key"。

### Q: 支持流式响应吗？
支持，所有接口都默认使用 SSE 流式响应。

### Q: 如何处理函数调用？
代理会自动转换工具调用格式，支持 OpenAI Tools 协议。

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交 Issue 和 Pull Request！
