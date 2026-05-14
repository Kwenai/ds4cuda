# Contributing to ds4cuda

[English](CONTRIBUTING.md) | **中文**

ds4cuda 是 DeepSeek-V4-Flash 在 DGX Spark 上的单模型 CUDA 推理引擎。不是
通用 GGUF 运行器。布局与 kernel 为这一个模型手调。扩展范围的补丁（其他
模型、其他 GPU、无硬性理由的 FP4）一律不收。

## 环境设置

驻留服务需要生产 GGUF。不随仓库分发，通过环境变量指向：

| env var | what | example |
|---|---|---|
| `DS4CUDA_GGUF` | 生产 GGUF 全路径（SoA 重排后的 V4-Flash） | `~/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf` |

`patches/ds4_cpu_stage_dump.patch` 给需要对照 `antirez/ds4` 做对齐的高级
用户。它给每个 CPU 侧 stage 张量打 DSST 头，打过补丁的 `ds4_native` 跑
起来可以按 stage dump fp32 缓冲。普通构建不需要。

## 提交前

1. 读 `README.md`。
2. 在 Spark（sm_120）上干净编译。`make` 应当退出 0，无 warning。
3. 编译 server 和 chat CLI：

   ```
   make server-main
   make chat-cli
   ```

   都要干净编过。
4. 起 server，发请求，看输出连贯。改动若涉及 forward 路径或任何 kernel，
   必须用 81 GB GGUF 验证；连续两次相同请求，输出连贯且可复现。

## 接受的改动

- Bug 修复
- 顶层 kernel 的性能优化：`mul_mv_q8_0_q8_0`、`moe_iq2_xxs_pair_swiglu_resident_soa_v2`、`hc_pre_*`、`attn_out`、`flash_attn`
- 文档修订
- 构建卫生（Makefile 清晰度、依赖声明）

## 不接受的改动

- 其他模型支持（T4 / 3090 / Llama / Mistral）。请 fork。
- 框架绑定（PyTorch / JAX / ONNX）。不在范围。
- 没有实测性能收益、仅为“可读性”重写核心 kernel
- 不改行为的代码风格 PR

## 编码规则

C / CUDA：

- host C 用 `gnu11`，`.cu` 用 `c++17`
- 不引新依赖。仓库只链 CUDA toolkit + libc。
- 一个逻辑改动一个 commit。要改写历史就 squash。
- Commit 信息：首行 ≤ 70 字符，描述改动。正文写为什么。

文档：

- 短句。
- 去修饰：“优雅”、“健壮”、“强大”、“显著提升”——删掉。
- 陈述事实。陈述数字。
- 失败就说失败。别包装成“挑战”。


## 问题反馈

写清：
- 硬件（Spark？其他 GB10？独立 GPU？）
- CUDA 版本，驱动版本
- 稳态 RSS（`free -g`）
- 哪个 `make` 目标失败
- 失败输出前 N 行

不要报：
- “性能慢”但没有 `nsys` profile 或 `perf_timeline` 抓到瓶颈 stage
- “T4 上跑不起来”——见“不接受的改动”

## License

贡献即同意按 GPL-2.0 授权（仓库根目录的 `LICENSE`）。
