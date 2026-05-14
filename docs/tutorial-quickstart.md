# Tutorial: First Run

**English** | [中文](tutorial-quickstart.zh-CN.md)

From an empty repo to an HTTP response. 4 steps.

## Prereqs

- NVIDIA DGX Spark, or a GPU with more than 128 GB VRAM
- ≥ 150 GB free disk
- CUDA 12+ (`/usr/local/cuda/bin/nvcc`)
- `cc`, `make`, Python 3.10+, `huggingface-cli` or `curl`

Discrete GPUs, multi-GPU sharding, and CPU-only are not supported.

## 1. Clone the repo

```bash
git clone https://github.com/Kwenai/ds4cuda.git
cd ds4cuda
```

## 2. Download the model

```bash
mkdir -p $HOME/models/deepseek-v4-flash
huggingface-cli download antirez/deepseek-v4-gguf \
  DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --local-dir $HOME/models/deepseek-v4-flash
```

Or direct link:

```bash
mkdir -p $HOME/models/deepseek-v4-flash
curl -L -C - \
  -o $HOME/models/deepseek-v4-flash/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf
```

File is 80.76 GiB.

## 3. Build

```bash
make server-main chat-cli repack-tool
```

Outputs:

- `build/ds4cuda_server`: resident HTTP server
- `build/chat_cli`: one-shot prompt CLI
- `build/repack_gguf_soa`: GGUF repack tool

Next, repack the original GGUF into the SoA v2 layout. The production path defaults to this step (+13% token/s):

```bash
./build/repack_gguf_soa --replace \
  $HOME/models/deepseek-v4-flash/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  $HOME/models/deepseek-v4-flash/DeepSeek-V4-Flash-soa-v2.gguf
```

About 6 minutes. Output is ~81 GB. `--replace` drops the original AoS and keeps only SoA v2. Repack does not change model capability, only byte layout.

## 4. Run

Start the server:

```bash
./build/ds4cuda_server \
  --gguf $HOME/models/deepseek-v4-flash/DeepSeek-V4-Flash-soa-v2.gguf \
  --port 8080 \
  --max-context 262144 \
  --max-tokens 262144
```

`--max-context 262144` is the server's max KV cache capacity (256K tokens), about 10 GB of session memory. `--max-tokens 262144` is the default per-request generation ceiling; a smaller client-side `max_tokens` takes priority. The endpoint no longer hard-caps; client decides generation length.

Weight load takes about 60 seconds. Wait for `listening on http://127.0.0.1:8080` before sending requests.

Send a request:

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

The Anthropic path `/v1/messages` works too.

Ctrl-C exits the server. Weights are released only after the process exits.

## Troubleshooting

- `nvcc not found`: install CUDA 12+, or add `/usr/local/cuda/bin` to `PATH`.
- `out of memory` on load: confirm 128+ GB unified memory, no other processes holding VRAM.
- Server hangs at startup: weight load takes ~60 s on first run, wait for the listen line.
- `repack_gguf_soa` fails mid-way: check free disk ≥ 81 GB, rerun without `--replace` to keep the original.
- HTTP request returns 400: check the JSON payload, confirm `model` and `messages` fields are present.
