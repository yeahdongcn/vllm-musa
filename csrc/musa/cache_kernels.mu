#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <torch/all.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

#include <cstdlib>

#include "vec_utils.muh"

// =============================================================================
// PERF NOTE  (MUSA-0151, characterized 2026-05-24)
// =============================================================================
//
// Kernel: reshape_and_cache_flash_nhd_kernel<T, BLOCK_X=512, TOKENS_PER_BLOCK>
// Workload: pure bandwidth-bound gather (key, value) -> scatter (kv-cache),
// 4 × num_tokens × num_heads × head_size × sizeof(T) bytes per call,
// no compute beyond index arithmetic.
//
// Current design layers:
//   1. LSU cache-bypass loads (`vload16_byp_slc`) — the V3 floor from the
//      sphere-kb arc (without LSU bypass plain loads are 2-5× slower).
//   2. Vec16<T> (128-bit) vectorized loads + stores.
//   3. Shape-adaptive dispatcher: TOKENS_PER_BLOCK ∈ {1,2,4,8} chosen by
//      vecs_per_token band; env override
//      VLLM_MUSA_RESHAPE_CACHE_TOKENS_PER_BLOCK.
//
// sphere-kb status: no probe exists yet for this kernel (closest is
// `probes/sgl_kernel_ports_20260428/03-apply_rope_pos_ids/` which fuses
// RoPE + KV-write but is a separate op). Verify-and-document baseline
// recorded by the regression bench
// `benchmarks/op_perf/bench_reshape_and_cache_flash.py`.
//
// Deferred optimization candidates (V2+):
//   - Adaptive BLOCK_X: with current BLOCK_X=512 fixed, vecs_per_token=128
//     (typical M2.5: num_kv_heads=8, head_size=128) under TOKENS_PER_BLOCK=1
//     leaves 75 % of threads idle (128/512). Switching to BLOCK_X = next
//     power-of-2 above vecs_per_token would zero idle and quadruple
//     concurrent CTAs/SM, potentially +5..15 %. Mirrors the V13 'Layer 4
//     adaptive BLOCK_X' rule that transferred from RoPE to RMSNorm in the
//     sphere-kb arc. Easy to land; needs a bench cycle to verify.
//   - Non-temporal store hint for kv-cache writes (kv-cache is rarely
//     re-read until attention; NT store may bypass an L2 fill that's
//     immediately discarded). Needs a probe-style mu-asm 'slc=nt' load
//     equivalent for store; speculative until measured.
//   - TME tile-stride load to overlap key + value streams. Gated behind
//     MUTE_ARCH_TME_MP31_ACTIVATED. Speculative.
//
// =============================================================================

namespace vllm_musa {

template <typename T, int BLOCK_X, int TOKENS_PER_BLOCK>
__global__ void __launch_bounds__(BLOCK_X, 1)
    reshape_and_cache_flash_nhd_kernel(
        const T* __restrict__ key, const T* __restrict__ value,
        T* __restrict__ key_cache, T* __restrict__ value_cache,
        const int64_t* __restrict__ slot_mapping, int num_tokens,
        int vecs_per_token, int block_size, int64_t key_stride,
        int64_t value_stride, int64_t block_stride, int64_t page_stride) {
  constexpr int VEC_SIZE = 8;
  const int base_token = blockIdx.x * TOKENS_PER_BLOCK;
  const int total_vecs = TOKENS_PER_BLOCK * vecs_per_token;

  for (int linear = threadIdx.x; linear < total_vecs; linear += BLOCK_X) {
    const int local_token = linear / vecs_per_token;
    const int token_idx = base_token + local_token;
    if (token_idx >= num_tokens) {
      continue;
    }

    const int64_t slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) {
      continue;
    }

    const int vec_idx = linear - local_token * vecs_per_token;
    const int elem_offset = vec_idx * VEC_SIZE;
    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx - block_idx * block_size;

    const int64_t src_k = static_cast<int64_t>(token_idx) * key_stride +
                          elem_offset;
    const int64_t src_v = static_cast<int64_t>(token_idx) * value_stride +
                          elem_offset;
    const int64_t dst = block_idx * block_stride + block_offset * page_stride +
                        elem_offset;

    Vec16<T> key_vec = vload16_byp_slc(key + src_k);
    Vec16<T> value_vec = vload16_byp_slc(value + src_v);
    vstore16(key_cache + dst, key_vec);
    vstore16(value_cache + dst, value_vec);
  }
}

template <typename T>
void dispatch_reshape_and_cache_flash_nhd(
    const T* key, const T* value, T* key_cache, T* value_cache,
    const int64_t* slot_mapping, int num_tokens, int num_heads, int head_size,
    int block_size, int64_t key_stride, int64_t value_stride,
    int64_t block_stride, int64_t page_stride, musaStream_t stream) {
  constexpr int BLOCK_X = 512;
  const int vecs_per_token = (num_heads * head_size) / 8;

  static const int forced_tokens_per_block = []() {
    const char* env = std::getenv("VLLM_MUSA_RESHAPE_CACHE_TOKENS_PER_BLOCK");
    return env == nullptr ? 0 : std::atoi(env);
  }();

#define LAUNCH(TPB)                                                        \
  reshape_and_cache_flash_nhd_kernel<T, BLOCK_X, TPB>                      \
      <<<(num_tokens + TPB - 1) / TPB, BLOCK_X, 0, stream>>>(              \
          key, value, key_cache, value_cache, slot_mapping, num_tokens,    \
          vecs_per_token, block_size, key_stride, value_stride,            \
          block_stride, page_stride)

  if (forced_tokens_per_block == 1) {
    LAUNCH(1);
  } else if (forced_tokens_per_block == 2) {
    LAUNCH(2);
  } else if (forced_tokens_per_block == 4) {
    LAUNCH(4);
  } else if (forced_tokens_per_block == 8) {
    LAUNCH(8);
  } else if (vecs_per_token <= 64) {
    LAUNCH(8);
  } else if (vecs_per_token <= 128) {
    LAUNCH(4);
  } else if (vecs_per_token <= 256) {
    LAUNCH(2);
  } else {
    LAUNCH(1);
  }

#undef LAUNCH
}

}  // namespace vllm_musa

void musa_reshape_and_cache_flash_nhd(torch::Tensor& key, torch::Tensor& value,
                                      torch::Tensor& key_cache,
                                      torch::Tensor& value_cache,
                                      torch::Tensor& slot_mapping) {
  TORCH_CHECK(key.device().is_privateuseone(), "key must be a MUSA tensor");
  TORCH_CHECK(value.device().is_privateuseone(), "value must be a MUSA tensor");
  TORCH_CHECK(key_cache.device().is_privateuseone(),
              "key_cache must be a MUSA tensor");
  TORCH_CHECK(value_cache.device().is_privateuseone(),
              "value_cache must be a MUSA tensor");
  TORCH_CHECK(slot_mapping.device().is_privateuseone(),
              "slot_mapping must be a MUSA tensor");
  TORCH_CHECK(key.dim() == 3, "key must be [tokens, heads, head_size]");
  TORCH_CHECK(value.dim() == 3, "value must be [tokens, heads, head_size]");
  TORCH_CHECK(key_cache.dim() == 4,
              "key_cache must be [blocks, block_size, heads, head_size]");
  TORCH_CHECK(value_cache.dim() == 4,
              "value_cache must be [blocks, block_size, heads, head_size]");
  TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be 1-D");
  TORCH_CHECK(key.scalar_type() == value.scalar_type(),
              "key/value dtype mismatch");
  TORCH_CHECK(key.scalar_type() == key_cache.scalar_type(),
              "key/key_cache dtype mismatch");
  TORCH_CHECK(value.scalar_type() == value_cache.scalar_type(),
              "value/value_cache dtype mismatch");
  TORCH_CHECK(slot_mapping.scalar_type() == at::ScalarType::Long,
              "slot_mapping must be int64");
  TORCH_CHECK(key.stride(2) == 1 && key.stride(1) == key.size(2),
              "key head/head_size dimensions must be contiguous");
  TORCH_CHECK(value.stride(2) == 1 && value.stride(1) == value.size(2),
              "value head/head_size dimensions must be contiguous");
  TORCH_CHECK(key_cache.size(0) == value_cache.size(0),
              "key/value cache block count mismatch");
  TORCH_CHECK(key_cache.size(1) == value_cache.size(1),
              "key/value cache block size mismatch");
  TORCH_CHECK(key_cache.size(2) == value_cache.size(2),
              "key/value cache head count mismatch");
  TORCH_CHECK(key_cache.size(3) == value_cache.size(3),
              "key/value cache head size mismatch");
  TORCH_CHECK(key.size(1) == key_cache.size(2), "head count mismatch");
  TORCH_CHECK(value.size(1) == value_cache.size(2), "value head count mismatch");
  TORCH_CHECK(key.size(2) == key_cache.size(3), "head size mismatch");
  TORCH_CHECK(value.size(2) == value_cache.size(3), "value head size mismatch");
  TORCH_CHECK(slot_mapping.size(0) <= key.size(0),
              "slot_mapping longer than key rows");
  TORCH_CHECK(slot_mapping.size(0) <= value.size(0),
              "slot_mapping longer than value rows");
  TORCH_CHECK(key.size(2) % 8 == 0, "head_size must be a multiple of 8");
  TORCH_CHECK(key_cache.stride(3) == 1 && value_cache.stride(3) == 1,
              "cache head dimension must be contiguous");
  TORCH_CHECK(key_cache.stride(2) == key_cache.size(3),
              "key_cache must use NHD layout");
  TORCH_CHECK(value_cache.stride(2) == value_cache.size(3),
              "value_cache must use NHD layout");
  TORCH_CHECK(key_cache.stride(1) == key_cache.size(2) * key_cache.size(3),
              "key_cache page stride must be contiguous NHD");
  TORCH_CHECK(value_cache.stride(1) ==
                  value_cache.size(2) * value_cache.size(3),
              "value_cache page stride must be contiguous NHD");

  const c10::musa::OptionalMUSAGuard guard(device_of(key));
  musaStream_t stream = c10::musa::getCurrentMUSAStream().stream();
  const int num_tokens = static_cast<int>(slot_mapping.size(0));
  const int num_heads = static_cast<int>(key.size(1));
  const int head_size = static_cast<int>(key.size(2));
  const int block_size = static_cast<int>(key_cache.size(1));
  const int64_t key_stride = key.stride(0);
  const int64_t value_stride = value.stride(0);
  const int64_t block_stride = key_cache.stride(0);
  const int64_t page_stride = key_cache.stride(1);

  if (num_tokens == 0) {
    return;
  }

  if (key.scalar_type() == at::ScalarType::Half) {
    vllm_musa::dispatch_reshape_and_cache_flash_nhd<__half>(
        static_cast<const __half*>(key.data_ptr()),
        static_cast<const __half*>(value.data_ptr()),
        static_cast<__half*>(key_cache.data_ptr()),
        static_cast<__half*>(value_cache.data_ptr()),
        slot_mapping.data_ptr<int64_t>(), num_tokens, num_heads, head_size,
        block_size, key_stride, value_stride, block_stride, page_stride,
        stream);
  } else if (key.scalar_type() == at::ScalarType::BFloat16) {
    vllm_musa::dispatch_reshape_and_cache_flash_nhd<__mt_bfloat16>(
        static_cast<const __mt_bfloat16*>(key.data_ptr()),
        static_cast<const __mt_bfloat16*>(value.data_ptr()),
        static_cast<__mt_bfloat16*>(key_cache.data_ptr()),
        static_cast<__mt_bfloat16*>(value_cache.data_ptr()),
        slot_mapping.data_ptr<int64_t>(), num_tokens, num_heads, head_size,
        block_size, key_stride, value_stride, block_stride, page_stride,
        stream);
  } else {
    TORCH_CHECK(false, "only fp16 and bf16 are supported");
  }
}
