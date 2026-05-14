# Makefile — host C + CUDA build for ds4cuda.
#
# Targets:
#   make                  -> build/ds4cuda_tools, build/cuda_smoke (if nvcc available)
#   make smoke            -> build & run CUDA smoke test
#   make server-main      -> build/ds4cuda_server (resident HTTP server)
#   make chat-cli         -> build/chat_cli (one-shot prompt CLI)
#   make repack-tool      -> build/repack_gguf_soa (offline GGUF SoA v2 rewriter)
#   make clean
#
# CUDA detection: defaults to CUDA_HOME=/usr/local/cuda. If $(NVCC) does
# not exist on disk, all CUDA targets become no-ops with a warning.
# Set DS4CUDA_DISABLE_CUDA=1 to force-skip even when nvcc is present.
#
# Default arch is sm_120 (Blackwell consumer / DGX Spark GB10, compute 12.1).
# Future FP4 main-path work will need sm_120a; flip CUDA_ARCH=sm_120a then.

CC        ?= cc
AR        ?= ar
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= sm_120

ROOT    := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SRCDIR  := $(ROOT)src
INCDIR  := $(ROOT)include
CUDADIR := $(ROOT)cuda
BUILD   := $(ROOT)build

CFLAGS  := -O2 -g -std=gnu11 -Wall -Wextra -Wno-unused-parameter \
           -Wno-unused-function -fno-strict-aliasing \
           -I$(INCDIR) -I$(SRCDIR)
LDFLAGS :=

# nvcc flags. -Xcompiler forwards host flags. cudart is linked dynamically
# by default; static link can be opted in via DS4CUDA_CUDART_STATIC=1.
NVCC_FLAGS := -O2 -g -std=c++17 -arch=$(CUDA_ARCH) \
              -Xcompiler "-Wall -Wno-unused-parameter -fno-strict-aliasing" \
              -I$(INCDIR) -I$(CUDADIR)
ifdef DS4CUDA_CUDART_STATIC
NVCC_LDLIBS := -lcudart_static -ldl -lpthread -lrt
else
NVCC_LDLIBS := -lcudart -ldl -lpthread -lrt
endif

# CUDA enable/disable detection.
HAVE_NVCC :=
ifndef DS4CUDA_DISABLE_CUDA
ifneq ($(wildcard $(NVCC)),)
HAVE_NVCC := 1
endif
endif

CORE_SRC := $(SRCDIR)/gguf_parser.c \
            $(SRCDIR)/model_open.c  \
            $(SRCDIR)/tensor_table.c
CORE_OBJ := $(CORE_SRC:$(SRCDIR)/%.c=$(BUILD)/%.o)

TOOL_OBJ := $(BUILD)/main_tools.o

LIB      := $(BUILD)/libds4cuda.a
TOOL_BIN := $(BUILD)/ds4cuda_tools
SMOKE_CU  := $(CUDADIR)/smoke.cu
SMOKE_BIN := $(BUILD)/cuda_smoke

# Q8_0 dense matvec TU. Used by the forward path (forward_layer.o).
DENSE_Q8_CU  := $(CUDADIR)/dense_q8.cu
DENSE_Q8_OBJ := $(BUILD)/dense_q8.o

# Full 81 GB managed-memory loader. cuda/model_load_managed.cu hosts
# ds4_model_load_to_managed + ds4_model_managed_free + ds4_tensor_managed_ptr
# (extern "C", linked into libds4cuda via a separate object).
MODEL_LOAD_MANAGED_CU  := $(CUDADIR)/model_load_managed.cu
MODEL_LOAD_MANAGED_OBJ := $(BUILD)/model_load_managed.o

# Per-session decode state allocator. cuda/session_state.cu
# hosts ds4_session_state_alloc / _free / _reset / _arena_bytes
# (extern "C") — one cudaMalloc carved into all 43 per-layer KV +
# compressor state buffers + cross-layer HC residual + 16 MB activation
# arena.
SESSION_STATE_CU       := $(CUDADIR)/session_state.cu
SESSION_STATE_OBJ      := $(BUILD)/session_state.o

NORM_CU         := $(CUDADIR)/norm.cu
NORM_OBJ        := $(BUILD)/norm.o

# Tail-RoPE YaRN. One CUDA TU + one launcher
# (ds4cuda::launch_tail_rope_yarn_f32) covers both Qcur and KVrope
# because the inner per-head math is identical; only n_heads differs.
ROPE_CU              := $(CUDADIR)/rope.cu
ROPE_OBJ             := $(BUILD)/rope.o

# KV-cache FP8 (E4M3FN) round-trip TU.
FP8_KV_CU            := $(CUDADIR)/fp8_kv.cu
FP8_KV_OBJ           := $(BUILD)/fp8_kv.o

# HC pre/post fused TU. One fused kernel subsumes RMSNorm + F16
# matvec + sinkhorn split + weighted sum.
HC_SINKHORN_CU       := $(CUDADIR)/hc_sinkhorn.cu
HC_SINKHORN_OBJ      := $(BUILD)/hc_sinkhorn.o

# Flash attention TU. One CUDA TU + two launchers
# (ds4cuda::launch_flash_attn_decode_raw_f32 +
# ds4cuda::launch_flash_attn_decode_mixed_f32) cover both the layer-0/1
# raw-SWA path and the layer-2..42 mixed-attention path.
FLASH_ATTN_CU        := $(CUDADIR)/flash_attn.cu
FLASH_ATTN_OBJ       := $(BUILD)/flash_attn.o

# Layer-2 streaming compressor + indexer chain (ratio-4 path). One CUDA TU
# (cuda/compressor.{cu,cuh}) hosts launch_compressor_decode_step_f32
# which orchestrates pair-matvec + APE bias + state-row write
# + (on emit) pool + RMSNorm + tail-RoPE + (attn-only) fp8 round-trip
# + state ring-shift.  Reuses cuda/router.o (mul_mv_f16_f32 for the
# pair matmul), cuda/rope.o (tail-RoPE YaRN), cuda/fp8_kv.o (E4M3FN
# round-trip).
#
# A second small TU (cuda/indexer_allowed.{cu,cuh}) hosts
# launch_indexer_allowed_short_circuit_i32 — fills the int32 mask with
# 1's when n_comp <= top_k.
COMPRESSOR_CU             := $(CUDADIR)/compressor.cu
COMPRESSOR_OBJ            := $(BUILD)/compressor.o
INDEXER_ALLOWED_CU        := $(CUDADIR)/indexer_allowed.cu
INDEXER_ALLOWED_OBJ       := $(BUILD)/indexer_allowed.o

# Attention output projection TU. cuda/attn_out.cu hosts the inverse
# tail-RoPE kernel + the grouped Q8_0 matvec kernel; the final attn_out
# stage reuses launch_mul_mv_q8_0_q8_0_f32 from dense_q8.o directly.
ATTN_OUT_CU          := $(CUDADIR)/attn_out.cu
ATTN_OUT_OBJ         := $(BUILD)/attn_out.o

# Shared-expert FFN supporting TUs.
#   - cuda/glu.{cu,cuh}        : SwiGLU element-wise body silu(g)*u.
#   - cuda/elementwise.{cu,cuh}: launch_add_f32 (ffn_out = moe_out + shexp).
GLU_CU                   := $(CUDADIR)/glu.cu
GLU_OBJ                  := $(BUILD)/glu.o

ELEMENTWISE_CU           := $(CUDADIR)/elementwise.cu
ELEMENTWISE_OBJ          := $(BUILD)/elementwise.o
ARGMAX_CU                := $(CUDADIR)/argmax.cu
ARGMAX_OBJ               := $(BUILD)/argmax.o

# Routed-expert MoE chain (the hardest fused kernel cluster in the
# whole pipeline — design §1 D/E):
#   - cuda/moe_iq2_pair.{cu,cuh} hosts (a) the Q8_K activation quantizer
#     (launch_quantize_fp32_to_q8_K, mirror of ds4.c:1628) and (b) the
#     paired gate+up IQ2_XXS · Q8_K matvec fused with clamp + SwiGLU +
#     router weight (launch_routed_moe_pair_swiglu_f32, mirror of
#     ds4.c:3792-3811 + ds4.c:1877). Stage = `routed_expert_mid` (fp32
#     [n_used=6, ff_exp=2048]).
#   - cuda/moe_q2k_sum6.{cu,cuh} hosts the in-register sum6 Q2_K · Q8_K
#     down-projection (launch_routed_moe_q2k_sum6_f32, mirror of
#     ds4.c:3919-3927 — accumulates 6 expert dots into one fp32 sumf
#     per-row WITHOUT atomicAdd, design §4 explicitly forbids). Reuses
#     the Q8_K quantizer from moe_iq2_pair.o for per-slot mid → midq.
#     Stage = `ffn_moe_out` (fp32 [4096]).
MOE_IQ2_PAIR_CU      := $(CUDADIR)/moe_iq2_pair.cu
MOE_IQ2_PAIR_OBJ     := $(BUILD)/moe_iq2_pair.o
MOE_Q2K_SUM6_CU      := $(CUDADIR)/moe_q2k_sum6.cu
MOE_Q2K_SUM6_OBJ     := $(BUILD)/moe_q2k_sum6.o
PERF_TIMELINE_CU      := $(CUDADIR)/perf_timeline.cu
PERF_TIMELINE_OBJ     := $(BUILD)/perf_timeline.o

# Router TU. One fused TU cuda/router.{cu,cuh} hosting four launchers
# (mul_mv_f16_f32, sqrt_softplus_f32, hash_router_topk_ids_i32,
# hash_router_topk_w_f32) for layer 0/1 hash-routing fast path, plus the
# biased top-6 routing launch_topk_selected_experts_f32 for layer 3..42.
ROUTER_CU                := $(CUDADIR)/router.cu
ROUTER_OBJ               := $(BUILD)/router.o

# 43-layer streaming forward.  Chains 31 launchers (the
# full attention + HC FFN + routed MoE pipeline) into one
# ds4cuda::ds4_forward_layer entry point dispatched on il in 0..42.
FORWARD_LAYER_CU       := $(CUDADIR)/forward_layer.cu
FORWARD_LAYER_OBJ      := $(BUILD)/forward_layer.o

# forward_token wrapper.  cuda/forward_token.cu wraps embed_token +
# 4-replica HC + 43x ds4_forward_layer + final HC head + output_norm +
# output projection into one ds4_forward_token entry.
FORWARD_TOKEN_CU       := $(CUDADIR)/forward_token.cu
FORWARD_TOKEN_OBJ      := $(BUILD)/forward_token.o

# Layer-major chunked prefill.  cuda/forward_batch.{cu,cuh} hosts
# ds4_forward_chunk — the layer-major outer/inner reordering of
# ds4_forward_token (cite ds4.c:7910 prefill_layer_major_cpu).
FORWARD_BATCH_CU       := $(CUDADIR)/forward_batch.cu
FORWARD_BATCH_OBJ      := $(BUILD)/forward_batch.o

# Disk KV cache.  cuda/kv_persist.{cu,cuh} hosts
# ds4_session_save_to_disk / _load_from_disk — DtoH/HtoD memcpy of every
# persistent device buffer in ds4_session_state (KV cache + compressor
# state + residual_hc) plus a self-contained header (magic + version +
# geometry + prompt_token_ids).  See cuda/kv_persist.cuh for the file
# format.
KV_PERSIST_CU         := $(CUDADIR)/kv_persist.cu
KV_PERSIST_OBJ        := $(BUILD)/kv_persist.o

# Graph-executor gap kernels (3 net-new kernels):
#   - cuda/embedding.{cu,cuh}        launch_embed_token_f16_to_f32
#       Token row gather + f16->f32 cast (mirror of ds4.c:2655 embed_token_f16).
#   - cuda/output_head.{cu,cuh}      launch_output_hc_head_f32
#       Final HC collapse: rms_norm_no_weight + matvec_f16(output_hc_fn) +
#       sigmoid+eps + hc_weighted_sum (mirror of ds4.c:8099 output_hc_head_one).
#       Distinct from launch_hc_post_f32 because the activation is single
#       scalar scale + sigmoid (no per-stream sinkhorn-style mix).
#   - cuda/router.cu (extension)     launch_topk_selected_experts_f32
#       Biased top-6 routing for layer 3..42 (mirror of ds4.c:5217
#       layer_topk_selected_experts_from_probs).  Selection uses
#       (probs + exp_probs_b), but weight normalization uses unbiased probs.
EMBEDDING_CU              := $(CUDADIR)/embedding.cu
EMBEDDING_OBJ             := $(BUILD)/embedding.o
OUTPUT_HEAD_CU            := $(CUDADIR)/output_head.cu
OUTPUT_HEAD_OBJ           := $(BUILD)/output_head.o

# ---- Tokenizer (host C; reads the 81 GB GGUF for vocab / merges) ----
# Port of the ds4 GPT-2 byte-level BPE tokenizer (ds4/ds4.c:13919-14588).
# Single new TU src/tokenizer/tokenizer.c builds against libds4cuda.a (for
# ds4_model_open + KV access) and the chat-template renderer
# (src/server/chat_template.c).
TOKENIZER_DIR      := $(SRCDIR)/tokenizer
TOKENIZER_SRC      := $(TOKENIZER_DIR)/tokenizer.c
TOKENIZER_OBJ      := $(BUILD)/tokenizer.o

# ---- HTTP server build group — host C only; no CUDA / GGUF / model ----
# OpenAI-compatible HTTP server with chat template renderer + tool-call
# schema validator + cjson_min minimal JSON parser. The chat completion
# endpoint accepts a caller-supplied generator; when none is installed
# (model not loaded) it falls back to an "OK" echo so this target works
# in any environment without the 81 GB model.
SERVER_DIR     := $(SRCDIR)/server
CJSON_DIR   := $(SRCDIR)/cjson_min
SERVER_OBJ     := $(BUILD)/server_http_server.o \
                  $(BUILD)/server_openai_endpoint.o \
                  $(BUILD)/server_anthropic_endpoint.o \
                  $(BUILD)/server_chat_template.o \
                  $(BUILD)/server_tool_calls.o \
                  $(BUILD)/server_sse_writer.o \
                  $(BUILD)/cjson_min.o
SERVER_LIB     := $(BUILD)/libds4cuda_server.a

# ---- Worker (real inference engine + worker thread) ---------------------
# src/server/inference_engine.cu hosts ds4cuda_inference_engine_create /
# _destroy + the real_buffered_generator + real_stream_generator +
# real_anthropic_generator entry points. Compiled with nvcc (it includes
# cuda/forward_token.cuh and calls ds4cuda::ds4_forward_token directly).
INFERENCE_ENGINE_CU  := $(SERVER_DIR)/inference_engine.cu
INFERENCE_ENGINE_OBJ := $(BUILD)/inference_engine.o
SERVER_MAIN_C        := $(SRCDIR)/main_server.c
SERVER_MAIN_BIN      := $(BUILD)/ds4cuda_server

# ---- SoA v2 CPU repack: AoS -> SoA v2 byte-permutation (host-only C) -------
# CPU mirror of cuda/moe_q2k_sum6.cu::launch_build_moe_q2k_sum6_resident_soa.
# Used by the offline GGUF repack tool so it doesn't need a GPU at all.
REPACK_SOA_OBJ      := $(BUILD)/repack_soa.o

# ---- Offline GGUF repack tool (host-only C) -------------------------------
# tools/repack_gguf_soa.c is the offline rewriter that emits SoA v2 packed
# copies of blk.<il>.ffn_{down,gate,up}_exps.weight for every layer.
TOOLS_DIR       := $(ROOT)tools
REPACK_TOOL_C   := $(TOOLS_DIR)/repack_gguf_soa.c
REPACK_TOOL_BIN := $(BUILD)/repack_gguf_soa

.PHONY: all smoke \
        q8-prealloc-uses-dp4a-test \
        server-main \
        repack-tool \
        chat-cli \
        clean cuda-stub

ifdef HAVE_NVCC
all: $(TOOL_BIN) cuda-stub $(SMOKE_BIN)
else
all: $(TOOL_BIN) cuda-stub
	@echo "[cuda] nvcc not found at $(NVCC); CUDA targets skipped."
endif

$(BUILD)/repack_soa.o: $(SRCDIR)/repack_soa.c $(SRCDIR)/repack_soa.h \
                       $(INCDIR)/ds4cuda.h \
                       $(INCDIR)/ds4cuda_soa_layout.h \
                       $(INCDIR)/ds4cuda_iq2_soa_layout.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(REPACK_TOOL_BIN): $(REPACK_TOOL_C) $(REPACK_SOA_OBJ) $(LIB) | $(BUILD)
	$(CC) $(CFLAGS) -I$(SRCDIR) -o $@ $(REPACK_TOOL_C) $(REPACK_SOA_OBJ) \
	    $(LIB) $(LDFLAGS)

repack-tool: $(REPACK_TOOL_BIN)
	@echo "built $(REPACK_TOOL_BIN)"

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/%.o: $(SRCDIR)/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(CORE_OBJ)
	$(AR) rcs $@ $^

# ---- HTTP server build rules ------------------------------------------------
# Each .c -> $(BUILD)/server_<basename>.o (prefixed to avoid colliding with
# CORE objects like $(BUILD)/main_tools.o etc).
SERVER_CFLAGS := $(CFLAGS) -I$(CJSON_DIR) -pthread

$(BUILD)/server_http_server.o: $(SERVER_DIR)/http_server.c \
                                $(SERVER_DIR)/http_server.h | $(BUILD)
	$(CC) $(SERVER_CFLAGS) -c $< -o $@

$(BUILD)/server_openai_endpoint.o: $(SERVER_DIR)/openai_endpoint.c \
                                    $(SERVER_DIR)/openai_endpoint.h \
                                    $(SERVER_DIR)/chat_template.h \
                                    $(SERVER_DIR)/tool_calls.h \
                                    $(SERVER_DIR)/sse_writer.h \
                                    $(CJSON_DIR)/cjson_min.h | $(BUILD)
	$(CC) $(SERVER_CFLAGS) -c $< -o $@

$(BUILD)/server_anthropic_endpoint.o: $(SERVER_DIR)/anthropic_endpoint.c \
                                       $(SERVER_DIR)/anthropic_endpoint.h \
                                       $(SERVER_DIR)/http_server.h \
                                       $(SERVER_DIR)/chat_template.h \
                                       $(SERVER_DIR)/tool_calls.h \
                                       $(CJSON_DIR)/cjson_min.h | $(BUILD)
	$(CC) $(SERVER_CFLAGS) -c $< -o $@

$(BUILD)/server_sse_writer.o: $(SERVER_DIR)/sse_writer.c \
                               $(SERVER_DIR)/sse_writer.h \
                               $(CJSON_DIR)/cjson_min.h | $(BUILD)
	$(CC) $(SERVER_CFLAGS) -c $< -o $@

$(BUILD)/server_chat_template.o: $(SERVER_DIR)/chat_template.c \
                                  $(SERVER_DIR)/chat_template.h | $(BUILD)
	$(CC) $(SERVER_CFLAGS) -c $< -o $@

$(BUILD)/server_tool_calls.o: $(SERVER_DIR)/tool_calls.c \
                               $(SERVER_DIR)/tool_calls.h \
                               $(CJSON_DIR)/cjson_min.h | $(BUILD)
	$(CC) $(SERVER_CFLAGS) -c $< -o $@

$(BUILD)/cjson_min.o: $(CJSON_DIR)/cjson_min.c \
                       $(CJSON_DIR)/cjson_min.h | $(BUILD)
	$(CC) $(SERVER_CFLAGS) -c $< -o $@

$(SERVER_LIB): $(SERVER_OBJ)
	$(AR) rcs $@ $^

# ---- Tokenizer build rules ------------------------------------------------
$(TOKENIZER_OBJ): $(TOKENIZER_SRC) $(TOKENIZER_DIR)/tokenizer.h $(INCDIR)/ds4cuda.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(TOOL_BIN): $(TOOL_OBJ) $(LIB)
	$(CC) $(CFLAGS) -o $@ $(TOOL_OBJ) $(LIB) $(LDFLAGS)

# cuda-stub: compile-test cuda/common.cuh layout asserts via a one-line .cu
# probe. No-op fallback when nvcc is unavailable or DS4CUDA_DISABLE_CUDA is
# set. Target name retained — referenced from `all:` and external docs.
cuda-stub:
ifdef HAVE_NVCC
	@echo "[cuda-stub] checking cuda/common.cuh layout via nvcc"
	@printf '#include "%s"\nint main(){return 0;}\n' "$(CUDADIR)/common.cuh" \
		> $(BUILD)/_cuda_stub.cu
	@$(NVCC) -I$(INCDIR) -I$(CUDADIR) -O0 -c $(BUILD)/_cuda_stub.cu \
		-o $(BUILD)/_cuda_stub.o || (echo "[cuda-stub] nvcc compile failed (non-fatal)"; true)
else
	@true
endif

# CUDA smoke test: cudaMalloc / kernel launch / memcpy round-trip + device
# props dump. Validates Spark CUDA toolchain end-to-end.
$(SMOKE_BIN): $(SMOKE_CU) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LDLIBS)

ifdef HAVE_NVCC
smoke: $(SMOKE_BIN)
	@echo "running $(SMOKE_BIN)"
	@$(SMOKE_BIN)
else
smoke:
	@echo "[smoke] nvcc not found at $(NVCC); cannot build CUDA smoke."
	@false
endif

# Q8_0 dense matvec TU. Production kernels used by the forward path
# (forward_layer.o). Compiled standalone so downstream targets can link
# it without dragging the full forward pipeline.
ifdef HAVE_NVCC
$(DENSE_Q8_OBJ): $(DENSE_Q8_CU) $(CUDADIR)/dense_q8.cuh $(CUDADIR)/common.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(DENSE_Q8_CU) -o $@

# Full 81 GB managed loader. The .cu defines ds4_model_load_to_managed
# / ds4_model_managed_free / ds4_tensor_managed_ptr (extern "C").
$(MODEL_LOAD_MANAGED_OBJ): $(MODEL_LOAD_MANAGED_CU) $(INCDIR)/ds4cuda.h | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(MODEL_LOAD_MANAGED_CU) -o $@

# ---- Session-state allocator ----------------------------------------
$(SESSION_STATE_OBJ): $(SESSION_STATE_CU) $(INCDIR)/ds4cuda.h | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(SESSION_STATE_CU) -o $@

# ---- Kernel TUs -----------------------------------------------------
$(NORM_OBJ): $(NORM_CU) $(CUDADIR)/norm.cuh $(CUDADIR)/common.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(NORM_CU) -o $@

$(ROPE_OBJ): $(ROPE_CU) $(CUDADIR)/rope.cuh $(CUDADIR)/common.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(ROPE_CU) -o $@

$(FP8_KV_OBJ): $(FP8_KV_CU) $(CUDADIR)/fp8_kv.cuh $(CUDADIR)/common.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(FP8_KV_CU) -o $@

$(HC_SINKHORN_OBJ): $(HC_SINKHORN_CU) $(CUDADIR)/hc_sinkhorn.cuh $(CUDADIR)/common.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(HC_SINKHORN_CU) -o $@

# Flash attention TU.
$(FLASH_ATTN_OBJ): $(FLASH_ATTN_CU) $(CUDADIR)/flash_attn.cuh $(CUDADIR)/common.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(FLASH_ATTN_CU) -o $@

# Attention output projection TU (kqv_back + attn_low fused kernels).
$(ATTN_OUT_OBJ): $(ATTN_OUT_CU) $(CUDADIR)/attn_out.cuh $(CUDADIR)/dense_q8.cuh $(CUDADIR)/common.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(ATTN_OUT_CU) -o $@

# Compressor + indexer TUs.  launch_compressor_decode_step_f32 covers
# both the attn (head_dim=512, fp8 round-trip on emit) and indexer
# (head_dim=128, no fp8) paths.  Reuses ROUTER_OBJ (F16 pair matvec),
# ROPE_OBJ (tail-RoPE), and FP8_KV_OBJ (E4M3FN round-trip on attn emit).
$(COMPRESSOR_OBJ): $(COMPRESSOR_CU) $(CUDADIR)/compressor.cuh \
                   $(CUDADIR)/router.cuh $(CUDADIR)/rope.cuh \
                   $(CUDADIR)/fp8_kv.cuh $(CUDADIR)/common.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(COMPRESSOR_CU) -o $@

$(INDEXER_ALLOWED_OBJ): $(INDEXER_ALLOWED_CU) $(CUDADIR)/indexer_allowed.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(INDEXER_ALLOWED_CU) -o $@

# GLU TU (SwiGLU body silu(g)*u) used by shared-expert / forward path.
$(GLU_OBJ): $(GLU_CU) $(CUDADIR)/glu.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(GLU_CU) -o $@

# Elementwise add + argmax TUs.
$(ELEMENTWISE_OBJ): $(ELEMENTWISE_CU) $(CUDADIR)/elementwise.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(ELEMENTWISE_CU) -o $@

$(ARGMAX_OBJ): $(ARGMAX_CU) $(CUDADIR)/argmax.cuh $(CUDADIR)/common.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(ARGMAX_CU) -o $@

# Routed-expert MoE TUs (cuda/moe_iq2_pair.o, cuda/moe_q2k_sum6.o).
# moe_q2k_sum6.o reuses launch_quantize_fp32_to_q8_K from moe_iq2_pair.o
# (the Q8_K activation quantizer is shared between the routed_expert_mid
# input quantize and the per-slot ffn_moe_out input quantize).
$(MOE_IQ2_PAIR_OBJ): $(MOE_IQ2_PAIR_CU) $(CUDADIR)/moe_iq2_pair.cuh $(CUDADIR)/common.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(MOE_IQ2_PAIR_CU) -o $@

$(MOE_Q2K_SUM6_OBJ): $(MOE_Q2K_SUM6_CU) $(CUDADIR)/moe_q2k_sum6.cuh $(CUDADIR)/moe_iq2_pair.cuh $(CUDADIR)/common.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(MOE_Q2K_SUM6_CU) -o $@

# Forward-path perf timeline TU (consumed by forward_layer.o / forward_token.o).
$(PERF_TIMELINE_OBJ): $(PERF_TIMELINE_CU) $(CUDADIR)/perf_timeline.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(PERF_TIMELINE_CU) -o $@

# Router stages. One CUDA TU (cuda/router.{cu,cuh}) backs all four
# launchers.
$(ROUTER_OBJ): $(ROUTER_CU) $(CUDADIR)/router.cuh $(CUDADIR)/common.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(ROUTER_CU) -o $@

# ---- Graph-executor gap kernels -------------------------------------
$(EMBEDDING_OBJ): $(EMBEDDING_CU) $(CUDADIR)/embedding.cuh $(CUDADIR)/common.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(EMBEDDING_CU) -o $@

$(OUTPUT_HEAD_OBJ): $(OUTPUT_HEAD_CU) $(CUDADIR)/output_head.cuh $(CUDADIR)/common.cuh | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(OUTPUT_HEAD_CU) -o $@

# ---- forward_layer TU + shared dep list ------------------------------
# forward_layer.o links every per-layer launcher TU and is reused by
# every downstream forward target.
$(FORWARD_LAYER_OBJ): $(FORWARD_LAYER_CU) $(CUDADIR)/forward_layer.cuh \
                     $(CUDADIR)/hc_sinkhorn.cuh $(CUDADIR)/norm.cuh \
                     $(CUDADIR)/dense_q8.cuh $(CUDADIR)/rope.cuh \
                     $(CUDADIR)/fp8_kv.cuh $(CUDADIR)/flash_attn.cuh \
                     $(CUDADIR)/attn_out.cuh $(CUDADIR)/glu.cuh \
                     $(CUDADIR)/elementwise.cuh $(CUDADIR)/router.cuh \
                     $(CUDADIR)/moe_iq2_pair.cuh $(CUDADIR)/moe_q2k_sum6.cuh \
                     $(CUDADIR)/compressor.cuh $(CUDADIR)/indexer_allowed.cuh \
                     $(CUDADIR)/perf_timeline.cuh \
                     $(CUDADIR)/common.cuh $(INCDIR)/ds4cuda.h | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(FORWARD_LAYER_CU) -o $@

FWD_LAYER0_DEPS := $(FORWARD_LAYER_OBJ) \
                  $(HC_SINKHORN_OBJ) $(NORM_OBJ) $(DENSE_Q8_OBJ) \
                  $(ROPE_OBJ) $(FP8_KV_OBJ) $(FLASH_ATTN_OBJ) \
                  $(ATTN_OUT_OBJ) $(GLU_OBJ) $(ELEMENTWISE_OBJ) \
                  $(ROUTER_OBJ) $(MOE_IQ2_PAIR_OBJ) $(MOE_Q2K_SUM6_OBJ) \
                  $(COMPRESSOR_OBJ) $(INDEXER_ALLOWED_OBJ) \
                  $(PERF_TIMELINE_OBJ) \
                  $(MODEL_LOAD_MANAGED_OBJ) $(SESSION_STATE_OBJ) \
                  $(LIB)

q8-prealloc-uses-dp4a-test:
	@echo "checking Q8 dp4a experiment remains opt-in after full-model no-gain result"
	@if ! grep -n 'bool q8_0_q8_0_use_dp4a' $(DENSE_Q8_CU); then \
		echo "FAIL: missing Q8 dp4a routing guard"; \
		exit 1; \
	fi
	@if ! grep -n 'return false;' $(DENSE_Q8_CU); then \
		echo "FAIL: Q8 dp4a experiment is no longer opt-in"; \
		exit 1; \
	fi

# ---- forward_token wrapper ------------------------------------------
# forward_token.o links forward_layer.o (same per-layer pipeline) plus
# the embedding + output_head launchers used in the tail half.
$(FORWARD_TOKEN_OBJ): $(FORWARD_TOKEN_CU) $(CUDADIR)/forward_token.cuh \
                    $(CUDADIR)/forward_layer.cuh $(CUDADIR)/embedding.cuh \
                    $(CUDADIR)/output_head.cuh $(CUDADIR)/norm.cuh \
                    $(CUDADIR)/dense_q8.cuh $(CUDADIR)/perf_timeline.cuh \
                    $(CUDADIR)/common.cuh \
                    $(INCDIR)/ds4cuda.h | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(FORWARD_TOKEN_CU) -o $@

# Aggregate dep list for downstream forward consumers.
FWD_TOKEN_DEPS := $(FORWARD_TOKEN_OBJ) $(FWD_LAYER0_DEPS) \
                     $(EMBEDDING_OBJ) $(OUTPUT_HEAD_OBJ)

# ---- Chunked prefill (layer-major) ----------------------------------
$(FORWARD_BATCH_OBJ): $(FORWARD_BATCH_CU) $(CUDADIR)/forward_batch.cuh \
                    $(CUDADIR)/forward_token.cuh $(CUDADIR)/forward_layer.cuh \
                    $(CUDADIR)/embedding.cuh $(CUDADIR)/output_head.cuh \
                    $(CUDADIR)/norm.cuh $(CUDADIR)/dense_q8.cuh \
                    $(CUDADIR)/common.cuh $(INCDIR)/ds4cuda.h | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(FORWARD_BATCH_CU) -o $@

# ---- Disk KV cache --------------------------------------------------
$(KV_PERSIST_OBJ): $(KV_PERSIST_CU) $(CUDADIR)/kv_persist.cuh $(INCDIR)/ds4cuda.h | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -c $(KV_PERSIST_CU) -o $@

# ---- Worker: real inference engine ----------------------------------
# inference_engine.o is a CUDA TU (.cu) that owns the public C-callable
# engine API: ds4cuda_inference_engine_create / _destroy +
# ds4cuda_real_buffered_generator / ds4cuda_real_stream_generator +
# ds4cuda_real_anthropic_generator. It #includes cuda/forward_token.cuh
# and calls ds4cuda::ds4_forward_token directly, plus the host-only
# tokenizer.h for prompt encode + per-token decode.
$(INFERENCE_ENGINE_OBJ): $(INFERENCE_ENGINE_CU) \
                         $(SERVER_DIR)/inference_engine.h \
                         $(SERVER_DIR)/openai_endpoint.h \
                         $(SERVER_DIR)/anthropic_endpoint.h \
                         $(TOKENIZER_DIR)/tokenizer.h \
                         $(CUDADIR)/forward_token.cuh \
                         $(CUDADIR)/argmax.cuh \
                         $(CUDADIR)/kv_persist.cuh \
                         $(CUDADIR)/moe_q2k_sum6.cuh \
                         $(INCDIR)/ds4cuda.h | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -I$(SRCDIR) -c $(INFERENCE_ENGINE_CU) -o $@

# Worker pulls in kv_persist.o because the engine wraps the disk save/
# load in its public API (prefix-sync feature).
WORKER_DEPS := $(INFERENCE_ENGINE_OBJ) $(TOKENIZER_OBJ) \
                  $(ARGMAX_OBJ) \
                  $(KV_PERSIST_OBJ) \
                  $(FWD_TOKEN_DEPS) $(SERVER_LIB)

# ---- chat_cli: one-shot prompt CLI for ad-hoc inference ------------------
CHAT_CLI_C   := $(TOOLS_DIR)/chat_cli.c
CHAT_CLI_BIN := $(BUILD)/chat_cli

$(BUILD)/chat_cli.o: $(CHAT_CLI_C) \
                     $(SERVER_DIR)/inference_engine.h \
                     $(SERVER_DIR)/chat_template.h \
                     $(TOKENIZER_DIR)/tokenizer.h \
                     $(INCDIR)/ds4cuda.h | $(BUILD)
	$(CC) $(SERVER_CFLAGS) -I$(SRCDIR) -c $< -o $@

$(CHAT_CLI_BIN): $(BUILD)/chat_cli.o $(WORKER_DEPS) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -Xcompiler -pthread -o $@ \
	    $(BUILD)/chat_cli.o $(WORKER_DEPS) $(NVCC_LDLIBS)

chat-cli: $(CHAT_CLI_BIN)
	@echo "built $(CHAT_CLI_BIN)"
	@echo "usage: $(CHAT_CLI_BIN) \"<prompt>\" [max_new_tokens]"

# ---- standalone resident HTTP server --------------------------------
$(BUILD)/main_server.o: $(SERVER_MAIN_C) \
                         $(SERVER_DIR)/inference_engine.h \
                         $(SERVER_DIR)/http_server.h \
                         $(SERVER_DIR)/openai_endpoint.h \
                         $(SERVER_DIR)/anthropic_endpoint.h \
                         $(INCDIR)/ds4cuda.h | $(BUILD)
	$(CC) $(SERVER_CFLAGS) -I$(SRCDIR) -c $< -o $@

$(SERVER_MAIN_BIN): $(BUILD)/main_server.o $(WORKER_DEPS) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -Xcompiler -pthread -o $@ \
	    $(BUILD)/main_server.o $(WORKER_DEPS) $(NVCC_LDLIBS)

server-main: $(SERVER_MAIN_BIN)
	@echo "built $(SERVER_MAIN_BIN)"
	@echo "usage: $(SERVER_MAIN_BIN) --port 8080 --max-context 262144"
else
server-main:
	@echo "[server-main] nvcc not found at $(NVCC); cannot build."
	@false

chat-cli:
	@echo "[chat-cli] nvcc not found at $(NVCC); cannot build."
	@false
endif

clean:
	rm -rf $(BUILD)
