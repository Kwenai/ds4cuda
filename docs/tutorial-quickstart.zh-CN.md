# Tutorial: 第一次跑通

[English](tutorial-quickstart.md) | **中文**

从空仓库到一个 HTTP 响应。4 步。

## 准备

- NVIDIA DGX Spark，或 VRAM 大于 128 GB 的 GPU
- 空闲磁盘 ≥ 150 GB
- CUDA 12+（`/usr/local/cuda/bin/nvcc`）
- `cc`、`make`、Python 3.10+、`huggingface-cli` 或 `curl`

独立 GPU、多卡切分、纯 CPU 都不支持。

## 1. 拉取仓库

```bash
git clone https://github.com/Kwenai/ds4cuda.git
cd ds4cuda
```

## 2. 拉取模型

```bash
mkdir -p $HOME/models/deepseek-v4-flash
huggingface-cli download antirez/deepseek-v4-gguf \
  DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --local-dir $HOME/models/deepseek-v4-flash
```

或直链：

```bash
mkdir -p $HOME/models/deepseek-v4-flash
curl -L -C - \
  -o $HOME/models/deepseek-v4-flash/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf
```

文件 80.76 GiB。

## 3. 编译

```bash
make server-main chat-cli repack-tool
```

产物：

- `build/ds4cuda_server`：驻留 HTTP 服务
- `build/chat_cli`：一次性 prompt CLI
- `build/repack_gguf_soa`：GGUF 重排工具

接下来把原始 GGUF repack 成 SoA v2 布局。生产路径默认走这一步（+13% token/s）：

```bash
./build/repack_gguf_soa --replace \
  $HOME/models/deepseek-v4-flash/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  $HOME/models/deepseek-v4-flash/DeepSeek-V4-Flash-soa-v2.gguf
```

约 6 分钟。产物约 81 GB。`--replace` 丢掉原始 AoS，只保留 SoA v2。Repack 不改模型能力，只改字节布局。

## 4. 运行

起 server：

```bash
./build/ds4cuda_server \
  --gguf $HOME/models/deepseek-v4-flash/DeepSeek-V4-Flash-soa-v2.gguf \
  --port 8080 \
  --max-context 262144 \
  --max-tokens 262144
```

`--max-context 262144` 是 server 端 KV cache 最大容量（256K tokens），约 10 GB 会话内存。`--max-tokens 262144` 是每请求默认生成上限；客户端给更小的 `max_tokens` 优先生效。endpoint 不再硬封顶；生成长度由客户端决定。

权重加载约 60 秒。等到 `listening on http://127.0.0.1:8080` 再发请求。

发请求：

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {"role": "user", "content": "Write a short essay on spaceborne intelligence"}
    ],
    "max_tokens": 256
  }'
```

Anthropic 路径 `/v1/messages` 同样可用。

Ctrl-C 退出 server。权重在进程退出后才释放。

## 常见问题

- `nvcc not found`：装 CUDA 12+，或把 `/usr/local/cuda/bin` 加进 `PATH`。
- 加载时 `out of memory`：确认 128+ GB 统一内存，无其他进程占着 VRAM。
- Server 启动卡住：首次权重加载约 60 s，等监听行出现。
- `repack_gguf_soa` 中途失败：检查空闲磁盘 ≥ 81 GB，去掉 `--replace` 重跑可保留原文件。
- HTTP 请求返回 400：检查 JSON payload，确认 `model` 和 `messages` 字段都在。
