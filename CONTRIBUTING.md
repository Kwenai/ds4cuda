# Contributing to ds4cuda

**English** | [中文](CONTRIBUTING.zh-CN.md)

ds4cuda is a single-model CUDA inference engine for DeepSeek-V4-Flash on DGX
Spark. It is not a general GGUF runner. Layouts and kernels are hand-tuned
for one model. Patches that broaden scope (other models, other GPUs, FP4
without a hard reason) will be rejected.

## Environment Setup

The resident server needs the production GGUF. It is not shipped in the
repo; you point at it via an env var:

| env var | what | example |
|---|---|---|
| `DS4CUDA_GGUF` | full path to the production GGUF (SoA-repacked V4-Flash) | `~/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf` |

`patches/ds4_cpu_stage_dump.patch` is kept for advanced users who want to
do their own alignment work against `antirez/ds4`. It tags every CPU-side
stage tensor with a DSST header so a patched `ds4_native` run can dump fp32
buffers per stage. Not needed for normal builds.

## Before you submit

1. Read `README.md`.
2. Build clean on Spark (sm_120). `make` should exit 0 with no warnings.
3. Build the server and chat CLI:

   ```
   make server-main
   make chat-cli
   ```

   Both should build clean.
4. Start the server, send a request, observe coherent output. If your
   change touches the forward path or any kernel, do this against the
   81 GB GGUF; output must be coherent and reproducible across two
   identical requests.

## What's accepted

- Bug fixes
- Performance work on the top kernels: `mul_mv_q8_0_q8_0`, `moe_iq2_xxs_pair_swiglu_resident_soa_v2`, `hc_pre_*`, `attn_out`, `flash_attn`
- Documentation fixes
- Build hygiene (Makefile clarity, dependency declarations)

## What's not accepted

- Other model support (T4 / 3090 / Llama / Mistral). Fork instead.
- Frameworks (PyTorch / JAX / ONNX bindings). Out of scope.
- Rewrites of core kernels for "readability" without measured perf gain
- Code style PRs that don't change behavior

## Coding rules

C / CUDA:

- `gnu11` for host C, `c++17` for `.cu` files
- No new dependencies. The repo links only against CUDA toolkit + libc.
- One commit per logical change. Squash if you have to rewrite history.
- Commit messages: first line ≤ 70 chars, describes the change. Body
  explains why.

Documentation:

- Short sentences.
- Strip decorations: "elegant", "robust", "powerful", "significant
  improvements" — gone.
- State facts. State numbers.
- If something failed, say it failed. Don't paint it as a "challenge".


## Reporting issues

State:
- Hardware (Spark? Other GB10? Discrete GPU?)
- CUDA version, driver version
- Steady-state RSS (`free -g`)
- Which `make` target fails
- First N lines of failure output

Don't report:
- "Performance is slow" without an `nsys` profile or `perf_timeline`
  capture showing the bottleneck stage
- "Doesn't work on T4" — see "What's not accepted"

## License

By contributing you agree to license your changes under GPL-2.0
(`LICENSE` in repo root).
