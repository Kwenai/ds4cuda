# Changelog

[English](CHANGELOG.md) | **中文**

## [0.5.0] — 2026-05-13

ds4cuda 首次开源发布。

只为 DeepSeek-V4-Flash 而写的 CUDA 推理引擎。目标硬件：
NVIDIA DGX Spark。从权重加载到 HTTP API，整条本地推理链路打通。

### 工作能力

- GGUF v3 parser + 81 GB managed-memory 加载（分块，4 GiB / 块）
- 43 层 streaming forward，单 token / 单序列
- 手写 CUDA kernel：
  - Q8_0 x Q8_0 matvec（按需走 dp4a 路径）
  - IQ2_XXS pair-SwiGLU（gate + up，routed MoE）
  - Q2_K sum-6 down（routed MoE）
  - flash-attention（decode_raw / decode_mixed）
  - RMSNorm、tail-RoPE YaRN、FP8 KV cache（E4M3FN）
  - HC sinkhorn 3-kernel multi-CTA split
- 256 专家 / 选 6 的 routed MoE。前 3 层 hash router，其余 40 层 biased top-k
- ratio-4 / ratio-128 compressor + indexer，含短路逻辑与长 prompt top-K 打分
- 离线 GGUF repack 工具（`tools/repack_gguf_soa.c`）。产出 SoA v2 专家张量。managed 后端直接读取。
- 驻留 HTTP 服务：OpenAI `/v1/chat/completions` + Anthropic `/v1/messages`，支持 SSE 流式
- 单会话 FIFO 推理引擎，含 prompt-prefix 同步 + KV 磁盘 save/load
- GPT-2 字节级 BPE tokenizer（识别 chat-template）
- chat_cli 交互工具

### 性能

| 阶段 | 速度 | 说明 |
|---|---:|---|
| First runnable path | 1.7 token/s | 跑通，欠打磨 |
| Stable baseline | 9.16 token/s | 端到端链路稳定 |
| Current path | 11.39 token/s | 布局调整 + 边际收尾 |

### 不做

- 其他模型
- 通用 GGUF 推理
- 多模型 serving
- 多用户并发调度
- 小显存 GPU 支持
- speculative decoding
- paged KV cache

### Blocked / Deferred

- CUDA Graphs 未落地。当前链路有结构性成本挡在前面。能做。这次不值得。

### License

GPLv2（与 Linux kernel 同）。
