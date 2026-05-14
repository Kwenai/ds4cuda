# Changelog

**English** | [中文](CHANGELOG.zh-CN.md)

## [0.5.0] — 2026-05-13

First open-source release of ds4cuda.

A CUDA inference engine built only for DeepSeek-V4-Flash. Target hardware:
NVIDIA DGX Spark. Complete local inference path from weight load to HTTP API.

### Capabilities

- GGUF v3 parser + 81 GB managed-memory load (chunked, 4 GiB / chunk)
- 43-layer streaming forward, single-token / single-sequence
- Hand-tuned CUDA kernels:
  - Q8_0 x Q8_0 matvec (with selective dp4a routing)
  - IQ2_XXS pair-SwiGLU (gate + up, routed MoE)
  - Q2_K sum-6 down (routed MoE)
  - flash-attention (decode_raw / decode_mixed)
  - RMSNorm, tail-RoPE YaRN, FP8 KV cache (E4M3FN)
  - HC sinkhorn 3-kernel multi-CTA split
- 256-expert / 6-used routed MoE. First 3 layers hash router, remaining 40 layers biased top-k
- ratio-4 / ratio-128 compressor + indexer, with short-circuit and long-prompt top-K scoring
- Offline GGUF repack tool (`tools/repack_gguf_soa.c`). Produces SoA v2 expert tensors. Managed backend reads directly.
- Resident HTTP server: OpenAI `/v1/chat/completions` + Anthropic `/v1/messages`, with SSE streaming
- Single-session FIFO inference engine, with prompt-prefix sync + disk KV save/load
- GPT-2 byte-level BPE tokenizer (chat-template aware)
- chat_cli interactive tool

### Performance

| Stage | Speed | Note |
|---|---:|---|
| First runnable path | 1.7 token/s | Ran, but underbuilt |
| Current path | 11.39 token/s | Layout + marginal follow-ups |

### Not in scope

- Other models
- General GGUF inference
- Multi-model serving
- Multi-user concurrent scheduling
- Small-VRAM GPU support
- Speculative decoding
- Paged KV cache

### Blocked / Deferred

- CUDA Graphs did not land. Current path has structural cost in the way. Can be done. Not worth it this time.

### License

GPLv2 (same as the Linux kernel).
