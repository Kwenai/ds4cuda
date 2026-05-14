# ds4cuda

[English](README.md) | **中文**

DeepSeek-V4-Flash 专用 CUDA 推理引擎。目标硬件：NVIDIA DGX Spark。

81 GB 权重。
43 层 CUDA forward。
11.39 token/s。

## 致敬

致 **DeepSeek 团队**。

人们以为护城河保护着 frontier AI。
DeepSeek 走过去，证明它一直只是纸。

V3。R1。V4-Flash。
每一次 release，整个行业要不重新定价自己，要不沉默。
每一次 release，闭源实验室的"独家能力"变成周末爱好者的本地路径。

本地 frontier AI 能存在，因为你们决定它能存在。

V4-Flash 是 ds4cuda 跑的那个模型。
更深的债更老 ——
没有开源 frontier，"本地推理路径长什么样"这个问题，没有主语。

不是站在你们的肩膀上。
是因为你们打开了这片领土，我们才有路可走。

## 致谢

工程肩膀：

- **Salvatore Sanfilippo (antirez)** 与 **ds4 工程** —— 单文件 CPU 实现证明这条路值得走，每个 stage 都有 CPU 参考可对。
- **GGUF / llama.cpp 社区** —— 量化格式与工具链。
- **NVIDIA** —— DGX Spark 统一内存架构 + CUDA 工具链。

一个模型。
一台机器。
一条从权重到 token 的路径。

不是通用 GGUF runner。
不是 vLLM 替代品。
不是多模型服务系统。

它只问一个窄问题：

放弃通用性。
只服务 DeepSeek-V4-Flash。
只面向 DGX Spark 这一类统一内存 CUDA 机器。
推理路径能不能端到端写出来、跑起来、测出来。

它现在跑得起来。

## 结果

| 项目 | 值 |
|---|---|
| 模型 | DeepSeek-V4-Flash |
| 硬件 | NVIDIA DGX Spark / GB10 |
| 权重 | ~81 GB |
| 路径 | GGUF -> CUDA forward -> HTTP API |
| 速度 | 11.39 token/s |
| 起点 | 1.7 token/s |
| 当前形态 | 单模型 CUDA 运行时 |

从 1.7 token/s 到 11.39 token/s，提升不是来自更复杂的服务栈。

来自把模型路径拆开，把数据布局改写成 GPU 愿意读的样子。

最关键的一步是 MoE 专家布局。

原始布局存得下。
解析得了。
但不匹配 CUDA warp 访问。

Lane 读不到相邻数据。
事务被拆开。
kernel 看起来在忙。其实在等内存。

SoA v2 做了一件直接的事：

按 GPU 想读的方式重排专家块。

这件事比很多“高级”优化都重要。

## 为什么做

我本职做商业航天操作系统，做星载具身智能的承载平台。

那个体系里，软件不止步于“能跑”。

要知道为什么能跑。
要知道边界在哪。
要知道什么时候开始劣化。
要知道一次内存增长、一次精度损失、一次调度抖动会不会变成系统故障。

转到 AI 这一侧，我没兴趣再写一个聊天 app。

我关心的问题更低一层：

智能离开云、进入一台功耗、内存、散热都受限的机器，本地推理路径该长什么样。

云端推理有它的前提。

电力。
冷却。
集群。
调度器。
副本。
一套厚到足以吸收复杂度的软件栈。

不是所有智能都跑在这些前提下。

如果一台机器不能默认云一直在，推理就不只是一个 API。

它是一条系统路径。

权重住在哪。
字节怎么走。
什么留在 cache。
kernel 在哪里等内存。
系统什么时候劣化。
边界怎么显形。

这就是 ds4cuda 存在的理由。

## DGX Spark 在这里是什么

DGX Spark 不是卫星硬件。

在这里它是一个工程代理物。

它有统一内存。
它放得下 81 GB 权重。
它的内存和带宽都有限。
它小到可以把一台机器上的本地推理路径完整摊开。

它回答不了卫星硬件提出的全部问题。

它能回答更早的一个：

当一个模型不能默认云一直在，本地运行时该从哪里起步。

## 为什么不做通用框架

通用框架解决的是广度问题。

多模型。
多硬件。
多用户。
多 batch。
多抽象。

那是正确的工程方向。

ds4cuda 选另一条：

一个模型。
一种权重格式。
一类机器。
一条路径。

通用带抽象。
抽象带分支。
分支带不受控的路径。

这个项目不覆盖更多场景。

它把一个场景推到底。

## 在这里能看到什么

这个仓库可以当推理系统的参考来读。

它把大模型推理从工程视角拆开：

- GGUF 权重怎么进入 CUDA 运行时
- 81 GB 权重怎么常驻
- 43 层 forward 怎么展开
- MoE 专家为什么卡在内存访问上
- 数据布局为什么决定 kernel 成本
- KV cache 和 prefix cache 怎么挂到服务路径上
- HTTP 服务怎么包住一个常驻模型
- 什么时候继续优化，什么时候停

没有 DGX Spark 也能读。

先读架构。
再读性能轨迹。
再读 kernel。
再读服务路径。

端到端跑起来需要硬件。
读路径不需要。

## 性能轨迹

起点是 1.7 token/s。

路径跑通，但远没到可用。
权重加载了。
forward 走通了。
输出出来了。
但大部分时间花在为错误的数据布局和不成熟的路径买单。

之后路径稳在 9.16 token/s。

那是第一个可用基线。

再往后，主要收益来自 MoE 专家布局。

后续收益要小得多。

这本身也是一种结果。

第一个真正的瓶颈解掉以后，系统进入另一个体制。

过了那条线，不是每个想法都值得做。
不是每个估算都能兑现。
不是每个“看着先进”的方案都该落地。

工程不是把能做的都做了。

工程是知道什么时候停。

## 当前能力

当前代码路径覆盖：

- DeepSeek-V4-Flash GGUF 加载
- CUDA forward
- 模型常驻的 HTTP 服务
- 本地 chat 风格请求
- 缓存复用
- KV 落盘保存与加载
- 长输出

这些服务一个目标：

把一个大模型在一台统一内存 CUDA 机器上跑出一条完整的本地推理路径。

## 模型下载

仓库不附带权重。

跑 ds4cuda 需要 DeepSeek-V4-Flash GGUF 量化权重，约 81 GB（80.76 GiB）。

当前使用的公开权重：

```text
https://huggingface.co/antirez/deepseek-v4-gguf/blob/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf
```

用 Hugging Face CLI 下载：

```bash
mkdir -p models/deepseek-v4-flash

huggingface-cli download antirez/deepseek-v4-gguf \
  DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --local-dir models/deepseek-v4-flash
```

或者直接链接：

```bash
mkdir -p models/deepseek-v4-flash

curl -L -C - \
  -o models/deepseek-v4-flash/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf
```

跑当前最快路径，需要把 GGUF 离线 repack 成 SoA v2 布局。这是默认的最快路径。

Repack 不改模型。

只改权重在文件里的排列方式，让 CUDA kernel 读得更顺。

## 架构

完整路径：

```text
GGUF 权重
    |
    v
托管权重基址
    |
    v
HTTP 请求
    |
    v
tokenizer
    |
    v
会话状态 + 缓存
    |
    v
43 层 x forward
    |
    v
logits
    |
    v
token
    |
    v
HTTP 响应
```

主要模块：

| 路径 | 内容 |
|---|---|
| `cuda/` | CUDA kernel 和 forward 路径 |
| `src/` | 运行时、服务粘合层、tokenizer 接入 |
| `src/server/` | HTTP 服务 |
| `src/tokenizer/` | tokenizer |
| `tools/` | 工具与布局/repack 流程 |
| `docs/` | 教程 |

## 运行

编译：

```bash
make server-main
```

启动：

```bash
./build/ds4cuda_server \
  --gguf models/deepseek-v4-flash/DeepSeek-V4-Flash-soa-v2.gguf \
  --port 8080 \
  --max-context 262144 \
  --max-tokens 262144
```

请求：

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

完整步骤：

- `docs/tutorial-quickstart.md`

## 不做什么

边界直接写出来。

ds4cuda 不做：

- 其他模型
- 通用 GGUF 推理
- 多模型服务
- 多用户并发调度
- 小显存 GPU 支持
- 投机解码
- 分页 KV cache

这些不是遗漏。

是选择。

## 失败和停止

CUDA Graphs 没落地。

不是因为它不重要。

是因为当前路径上有结构性成本挡着。

可以做。
这次不值得。

工程里有一种诱惑：

已经花了很多时间，所以再花一点。

这次不。

如果是被堵住，写被堵住。
如果收益不够，停。
如果风险超过收益，推后。

那也是工程结果的一部分。

## 文档

- `docs/tutorial-quickstart.md`：第一次运行
- `CONTRIBUTING.md`：贡献规则
- `CHANGELOG.md`：版本日志

## 希望遇到的人

需要生产级服务，先看成熟框架。

关心另一类问题的，可以聊：

- 受限硬件上的本地推理
- 星载具身智能
- CUDA kernel 与量化布局
- 统一内存上的大模型加载
- 从 OS / 运行时视角看 LLM 推理
- 模型专用推理引擎

这些问题会越来越重要。

不是因为每个模型都该离开云。
因为有一部分智能必须离开。

## 仓库

```text
https://github.com/Kwenai/ds4cuda.git
```

## License

GPLv2（同 Linux kernel）。
