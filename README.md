# ds4cuda

**English** | [中文](README.zh-CN.md)

A CUDA inference engine built only for DeepSeek-V4-Flash. Target hardware: NVIDIA DGX Spark.

81 GB weights.
43 CUDA forward layers.
11.39 token/s.

## Tribute

To the **DeepSeek team**.

People thought a moat protected frontier AI.
DeepSeek walked through it, and proved it was paper.

V3. R1. V4-Flash.
Each release, the industry had to either re-price itself or stay silent.
Each release, a closed lab's "differentiated capability" became a
weekend hobbyist's local run.

Local frontier AI exists because you decided it would.

V4-Flash is the model ds4cuda runs on.
The deeper debt is older —
without an open-weights frontier, "what does the local inference path
look like" is a question with no subject.

Not standing on your shoulders.
You opened the territory. That is why we have a path.

## Acknowledgments

Engineering shoulders this work stood on:

- **Salvatore Sanfilippo (antirez)** and the **ds4 project** — single-file CPU implementation that proved this path was worth taking, and a per-stage reference to align against.
- **GGUF / llama.cpp community** — quantization formats and tooling.
- **NVIDIA** — DGX Spark unified-memory architecture and the CUDA toolchain.

One model.
One machine.
One path from weights to tokens.

Not a general GGUF runner.
Not a vLLM replacement.
Not a multi-model serving system.

It asks one narrow question:

If you drop generality.
Only serve DeepSeek-V4-Flash.
Only target DGX Spark-class unified-memory CUDA machines.
Can the inference path be written, run, and measured end-to-end.

It runs now.

## Result

| Item | Value |
|---|---|
| Model | DeepSeek-V4-Flash |
| Hardware | NVIDIA DGX Spark / GB10 |
| Weights | ~81 GB |
| Path | GGUF -> CUDA forward -> HTTP API |
| Speed | 11.39 token/s |
| Starting point | 1.7 token/s |
| Current form | Single-model CUDA runtime |

From 1.7 token/s to 11.39 token/s, the gain did not come from a more complex serving stack.

It came from breaking the model path apart and rewriting the data layout into a form the GPU prefers to read.

The most critical step was MoE expert layout.

The original layout stores fine.
Parses fine.
But does not match CUDA warp access.

Lanes cannot read adjacent data.
Transactions get split.
The kernel looks busy. It is actually waiting on memory.

SoA v2 does one direct thing:

Reorder expert blocks the way the GPU wants to read them.

This matters more than many "advanced" optimizations.

## Why

My day job is commercial aerospace operating systems, and the harness for satellite-borne embodied intelligence.

In that system, software does not stop at "it runs."

It has to know why it runs.
Know where the limits are.
Know when it degrades.
Know whether one memory growth, one precision loss, one scheduling jitter turns into a system failure.

When I moved into AI, I was not interested in writing another chat app.

The question that mattered was lower:

When intelligence leaves the cloud and enters a machine with limited power, memory, and thermal budget, what should the local inference path look like.

Cloud inference has its preconditions.

Power.
Cooling.
Clusters.
Schedulers.
Replicas.
A software stack thick enough to absorb the complexity.

Not all intelligence runs under those preconditions.

If a machine cannot assume the cloud is always there, inference is not just an API.

It is a system path.

Where the weights live.
How the bytes move.
What stays in cache.
Where the kernel waits for memory.
When the system degrades.
How the boundary becomes visible.

That is why ds4cuda exists.

## What DGX Spark is here

DGX Spark is not satellite hardware.

Here it stands in as an engineering proxy.

It has unified memory.
It can hold 81 GB of weights.
Its memory and bandwidth are limited.
It is small enough to lay out the full single-machine local inference path.

It cannot answer every question that satellite hardware raises.

It can answer an earlier one:

When a model cannot assume the cloud is always there, where should the local runtime start.

## Why not a general framework

General frameworks solve a breadth problem.

Multi-model.
Multi-hardware.
Multi-user.
Multi-batch.
Multi-abstraction.

That is the right engineering direction.

ds4cuda picks another:

One model.
One weight format.
One class of machine.
One path.

Generality brings abstraction.
Abstraction brings branches.
Branches bring uncontrolled paths.

This project does not aim to cover more scenarios.

It pushes one scenario to the end.

## What you can see here

This repository can be read as a reference for inference systems.

It shows large-model inference taken apart from an engineering view:

- How GGUF weights enter the CUDA runtime
- How 81 GB of weights stay resident
- How 43 forward layers unfold
- Why MoE experts stall on memory access
- Why data layout decides kernel cost
- How KV cache and prefix cache attach to the serving path
- How an HTTP server wraps a resident model
- When to keep optimizing, and when to stop

You can read it without a DGX Spark.

Read the architecture first.
Then the performance trajectory.
Then the kernels.
Then the serving path.

Running it end-to-end needs hardware.
Reading the path does not.

## Performance trajectory

It started at 1.7 token/s.

The path ran, but was far from usable.
Weights loaded.
Forward went through.
Output came out.
But most of the time was spent paying for the wrong data layout and an immature path.

Then the path settled at 9.16 token/s.

That was the first usable baseline.

After that, the main gain came from MoE expert layout.

Later gains are much smaller.

That is also a result.

After the first real bottleneck is solved, the system enters a different regime.

Past that, not every idea is worth doing.
Not every projection cashes out.
Not every "looks advanced" plan should land.

Engineering is not doing everything you could.

Engineering is knowing when to stop.

## Current capabilities

The current code path covers:

- DeepSeek-V4-Flash GGUF loading
- CUDA forward
- HTTP server with resident model
- Local chat-style requests
- Cache reuse
- On-disk KV save and load
- Long-form output

These serve one goal:

Bring one large model to a complete local inference path on one unified-memory CUDA machine.

## Model download

The repository does not include weights.

Running ds4cuda needs the DeepSeek-V4-Flash GGUF quantized weights, about 81 GB (80.76 GiB).

The public weights currently used:

```text
https://huggingface.co/antirez/deepseek-v4-gguf/blob/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf
```

Download with the Hugging Face CLI:

```bash
mkdir -p models/deepseek-v4-flash

huggingface-cli download antirez/deepseek-v4-gguf \
  DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --local-dir models/deepseek-v4-flash
```

Or with a direct link:

```bash
mkdir -p models/deepseek-v4-flash

curl -L -C - \
  -o models/deepseek-v4-flash/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf
```

For the current fastest path, repack the GGUF offline into SoA v2 layout. This is the default fastest path.

Repack does not change the model.

It only changes how weights are arranged in the file, so CUDA kernels read them better.

## Architecture

The full path:

```text
GGUF weights
    |
    v
managed weight base
    |
    v
HTTP request
    |
    v
tokenizer
    |
    v
session state + cache
    |
    v
43 layers x forward
    |
    v
logits
    |
    v
token
    |
    v
HTTP response
```

Main modules:

| Path | Content |
|---|---|
| `cuda/` | CUDA kernels and forward path |
| `src/` | runtime, server glue, tokenizer integration |
| `src/server/` | HTTP server |
| `src/tokenizer/` | tokenizer |
| `tools/` | tools and layout/repack flow |
| `docs/` | tutorial |

## Run

Build:

```bash
make server-main
```

Start:

```bash
./build/ds4cuda_server \
  --gguf models/deepseek-v4-flash/DeepSeek-V4-Flash-soa-v2.gguf \
  --port 8080 \
  --max-context 262144 \
  --max-tokens 262144
```

Request:

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {"role": "user", "content": "Write a short piece on satellite-borne intelligence"}
    ],
    "max_tokens": 256
  }'
```

Full steps:

- `docs/tutorial-quickstart.md`

## What we do not do

Limits stated directly.

ds4cuda does not do:

- Other models
- General GGUF inference
- Multi-model serving
- Multi-user concurrent scheduling
- Small-VRAM GPU support
- Speculative decoding
- Paged KV cache

These are not omissions.

They are choices.

## Failure and stop

CUDA Graphs did not land.

Not because they do not matter.

Because the current path has structural cost in the way.

It can be done.
Not worth it this time.

There is a temptation in engineering:

You already spent a lot of time, so spend more.

Not this time.

If it is blocked, write blocked.
If the gain is not enough, stop.
If risk exceeds gain, defer.

That is also part of the engineering result.

## Documentation

- `docs/tutorial-quickstart.md`: first run
- `CONTRIBUTING.md`: contribution rules
- `CHANGELOG.md`: version log

## Whom we hope to meet

If you need production-grade serving, look at mature frameworks first.

If you care about a different class of problem, we may be worth talking to:

- Local inference on constrained hardware
- Satellite-borne embodied intelligence
- CUDA kernels and quantization layout
- Large-model loading on unified memory
- LLM inference seen from an OS / runtime view
- Model-specific inference engines

These questions will matter more over time.

Not because every model should leave the cloud.
Because a portion of intelligence must.

## Repository

```text
https://github.com/Kwenai/ds4cuda.git
```

## License

GPLv2 (same as the Linux kernel).
