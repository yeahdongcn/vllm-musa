// MUSA-0157: native MUSA port of fused_qk_norm_rope.
//
// The upstream csrc/fused_qknorm_rope_kernel.cu has an early-return guard
// `#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)`
// that fires on MUSA (where __CUDA_ARCH__ is undefined). The guard was needed
// because removing it causes mcc to fail compiling musa_bf16.hpp:1670 inline
// asm ("couldn't allocate output register for constraint 'h'") — the
// MUSA-0105 mcc bf16 blocker.
//
// Net effect: upstream silently no-ops for bf16 on MUSA. Verified empirically:
//   bf16 input scale=10 → kernel output=10 (RMSNorm never runs)
//   fp16 input scale=10 → kernel output=1.0 (RMSNorm runs correctly)
//
// This native port avoids musa_bf16.hpp's broken inline asm by doing manual
// `static_cast<float>(bf16)` conversions — same pattern as MUSA-0150
// csrc/musa/fused_add_rmsnorm.mu.
//
// Algorithm: per-warp processing of one (token, head). 32 threads × 4 elements
// per thread = head_dim=128. Warp-only RMSNorm reduction + per-warp RoPE
// rotation via __shfl_xor_sync (NEOX style: pair across half-warp).
//
// Compatibility note: this kernel reads/writes qkv in-place at offsets
// matching the upstream layout (Q heads [0, NHQ), K heads [NHQ, NHQ+NHK),
// V heads [NHQ+NHK, NHQ+NHK+NHV)). V is untouched.

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <torch/all.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace vllm_musa {

// Explicit dtype conversion helpers — static_cast doesn't reliably lower
// to the MUSA bf16↔fp32 hardware conversion path. These wrappers use the
// __bfloat162float / __float2bfloat16 intrinsics directly.
template <typename T>
__device__ __forceinline__ float qknr_to_fp32(T x) {
  return static_cast<float>(x);  // fp16 fallback
}
template <>
__device__ __forceinline__ float qknr_to_fp32<__mt_bfloat16>(__mt_bfloat16 x) {
  return __bfloat162float(x);
}
template <>
__device__ __forceinline__ float qknr_to_fp32<__half>(__half x) {
  return __half2float(x);
}

template <typename T>
__device__ __forceinline__ T qknr_from_fp32(float x) {
  return static_cast<T>(x);
}
template <>
__device__ __forceinline__ __mt_bfloat16 qknr_from_fp32<__mt_bfloat16>(float x) {
  return __float2bfloat16(x);
}
template <>
__device__ __forceinline__ __half qknr_from_fp32<__half>(float x) {
  return __float2half(x);
}

__device__ __forceinline__ float qknr_warp_reduce_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, o);
  }
  return v;
}

// One warp processes one (token, head) of head_dim=128.
// 32 threads × 4 elements/thread = 128 elements = head_dim.
template <typename T>
__global__ void fused_qk_norm_rope_kernel(
    T* __restrict__ qkv,
    const int num_heads_q,
    const int num_heads_k,
    const int num_heads_v,
    const float eps,
    const T* __restrict__ q_weight,
    const T* __restrict__ k_weight,
    const T* __restrict__ cos_sin_cache,
    const int is_neox,
    const int64_t* __restrict__ position_ids,
    const int num_tokens,
    const int head_dim,
    const int rotary_dim) {
  constexpr int NUM_ELEMS_PER_THREAD = 4;  // head_dim=128 / 32 lanes
  const int num_heads_total = num_heads_q + num_heads_k + num_heads_v;
  const int total_qk_heads = num_heads_q + num_heads_k;

  // Each warp handles one (token, head_qk).
  const int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane_id = threadIdx.x & 31;
  if (global_warp >= num_tokens * total_qk_heads) return;

  const int token_idx = global_warp / total_qk_heads;
  const int head_idx_qk = global_warp % total_qk_heads;
  const bool is_q = (head_idx_qk < num_heads_q);
  const int head_offset_in_token = head_idx_qk;  // Q at [0, NHQ), K at [NHQ, NHQ+NHK)
  const size_t base = (static_cast<size_t>(token_idx) * num_heads_total +
                       head_offset_in_token) * head_dim;

  // Pass 1: load + accumulate sum-of-squares
  float elements[NUM_ELEMS_PER_THREAD];
  float sum_of_squares = 0.0f;
#pragma unroll
  for (int i = 0; i < NUM_ELEMS_PER_THREAD; ++i) {
    const int dim = lane_id * NUM_ELEMS_PER_THREAD + i;
    const float val = qknr_to_fp32<T>(qkv[base + dim]);
    elements[i] = val;
    sum_of_squares += val * val;
  }

  sum_of_squares = qknr_warp_reduce_sum(sum_of_squares);
  const float rms_rcp =
      rsqrtf(sum_of_squares / static_cast<float>(head_dim) + eps);

  // Apply per-head RMSNorm * weight (weight is per-dim, same for all heads)
  const T* weight = is_q ? q_weight : k_weight;
#pragma unroll
  for (int i = 0; i < NUM_ELEMS_PER_THREAD; ++i) {
    const int dim = lane_id * NUM_ELEMS_PER_THREAD + i;
    const float w = qknr_to_fp32<T>(weight[dim]);
    elements[i] *= rms_rcp * w;
  }

  // Apply RoPE (only the first rotary_dim elements of head_dim).
  // is_neox=1: split rotary_dim into halves, pair (i, i+rotary_dim/2) across
  // the half-warp via __shfl_xor.
  // is_neox=0 (interleave): pairs (2i, 2i+1) — different shuffle pattern.
  if (lane_id * NUM_ELEMS_PER_THREAD < rotary_dim) {
    const int64_t pos_id = position_ids[token_idx];
    const T* cos_ptr = cos_sin_cache + pos_id * rotary_dim;
    const T* sin_ptr = cos_ptr + rotary_dim / 2;

    if (is_neox) {
      const int pair_offset = (rotary_dim / 2) / NUM_ELEMS_PER_THREAD;
      __syncwarp();
      float elements_partner[NUM_ELEMS_PER_THREAD];
#pragma unroll
      for (int i = 0; i < NUM_ELEMS_PER_THREAD; ++i) {
        elements_partner[i] =
            __shfl_xor_sync(0xffffffff, elements[i], pair_offset);
        if (lane_id < pair_offset) {
          elements_partner[i] = -elements_partner[i];
        }
      }
#pragma unroll
      for (int i = 0; i < NUM_ELEMS_PER_THREAD; ++i) {
        int dim_idx = lane_id * NUM_ELEMS_PER_THREAD + i;
        dim_idx = (dim_idx * 2) % rotary_dim;
        const int half_dim = dim_idx / 2;
        const float cos_val = qknr_to_fp32<T>(cos_ptr[half_dim]);
        const float sin_val = qknr_to_fp32<T>(sin_ptr[half_dim]);
        elements[i] = elements[i] * cos_val + elements_partner[i] * sin_val;
      }
      __syncwarp();
    } else {
      // GPT-J style: pairs (2i, 2i+1)
#pragma unroll
      for (int i = 0; i < NUM_ELEMS_PER_THREAD / 2; ++i) {
        const int idx0 = 2 * i;
        const int idx1 = 2 * i + 1;
        const int dim_idx = lane_id * NUM_ELEMS_PER_THREAD + idx0;
        const int half_dim = dim_idx / 2;
        const float cos_val = qknr_to_fp32<T>(cos_ptr[half_dim]);
        const float sin_val = qknr_to_fp32<T>(sin_ptr[half_dim]);
        const float v0 = elements[idx0];
        const float v1 = elements[idx1];
        elements[idx0] = v0 * cos_val - v1 * sin_val;
        elements[idx1] = v0 * sin_val + v1 * cos_val;
      }
    }
  }

  // Store back
#pragma unroll
  for (int i = 0; i < NUM_ELEMS_PER_THREAD; ++i) {
    const int dim = lane_id * NUM_ELEMS_PER_THREAD + i;
    qkv[base + dim] = qknr_from_fp32<T>(elements[i]);
  }
}

template <typename T>
void dispatch_fused_qk_norm_rope(
    T* qkv, int num_tokens, int num_heads_q, int num_heads_k, int num_heads_v,
    float eps, const T* q_weight, const T* k_weight, const T* cos_sin_cache,
    bool is_neox, const int64_t* position_ids, int head_dim, int rotary_dim,
    musaStream_t stream) {
  const int total_qk_heads = num_heads_q + num_heads_k;
  const int total_warps = num_tokens * total_qk_heads;
  constexpr int BLOCK_X = 256;       // 8 warps per CTA
  constexpr int WARPS_PER_BLOCK = BLOCK_X / 32;
  const int grid = (total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
  fused_qk_norm_rope_kernel<T><<<grid, BLOCK_X, 0, stream>>>(
      qkv, num_heads_q, num_heads_k, num_heads_v, eps, q_weight, k_weight,
      cos_sin_cache, static_cast<int>(is_neox), position_ids, num_tokens,
      head_dim, rotary_dim);
}

}  // namespace vllm_musa

void musa_fused_qk_norm_rope(torch::Tensor& qkv, int64_t num_heads_q,
                             int64_t num_heads_k, int64_t num_heads_v,
                             int64_t head_dim, double eps,
                             torch::Tensor& q_weight,
                             torch::Tensor& k_weight,
                             torch::Tensor& cos_sin_cache, bool is_neox,
                             torch::Tensor& position_ids,
                             int64_t forced_token_heads_per_warp) {
  TORCH_CHECK(qkv.device().is_privateuseone(), "qkv must be a MUSA tensor");
  TORCH_CHECK(qkv.is_contiguous(), "qkv must be contiguous");
  TORCH_CHECK(qkv.dim() == 2, "qkv must be 2-D (T, total_heads*head_dim)");
  TORCH_CHECK(q_weight.is_contiguous() && q_weight.dim() == 1,
              "q_weight must be 1-D contiguous");
  TORCH_CHECK(k_weight.is_contiguous() && k_weight.dim() == 1,
              "k_weight must be 1-D contiguous");
  TORCH_CHECK(q_weight.size(0) == head_dim, "q_weight size must = head_dim");
  TORCH_CHECK(k_weight.size(0) == head_dim, "k_weight size must = head_dim");
  TORCH_CHECK(cos_sin_cache.dim() == 2,
              "cos_sin_cache must be 2-D (max_pos, rotary_dim)");
  TORCH_CHECK(position_ids.dim() == 1,
              "position_ids must be 1-D (num_tokens,)");
  TORCH_CHECK(position_ids.scalar_type() == c10::ScalarType::Long,
              "position_ids must be int64");
  TORCH_CHECK(head_dim == 128,
              "MUSA fused_qk_norm_rope: head_dim=128 only in this V1 port");

  const int num_tokens = qkv.size(0);
  const int rotary_dim = cos_sin_cache.size(1);
  TORCH_CHECK(position_ids.size(0) == num_tokens,
              "position_ids must match num_tokens");

  const c10::musa::OptionalMUSAGuard guard(device_of(qkv));
  musaStream_t stream = c10::musa::getCurrentMUSAStream().stream();

  if (qkv.scalar_type() == c10::ScalarType::BFloat16) {
    vllm_musa::dispatch_fused_qk_norm_rope<__mt_bfloat16>(
        reinterpret_cast<__mt_bfloat16*>(qkv.data_ptr()), num_tokens,
        num_heads_q, num_heads_k, num_heads_v, static_cast<float>(eps),
        reinterpret_cast<const __mt_bfloat16*>(q_weight.data_ptr()),
        reinterpret_cast<const __mt_bfloat16*>(k_weight.data_ptr()),
        reinterpret_cast<const __mt_bfloat16*>(cos_sin_cache.data_ptr()),
        is_neox,
        reinterpret_cast<const int64_t*>(position_ids.data_ptr()),
        head_dim, rotary_dim, stream);
  } else if (qkv.scalar_type() == c10::ScalarType::Half) {
    vllm_musa::dispatch_fused_qk_norm_rope<__half>(
        reinterpret_cast<__half*>(qkv.data_ptr()), num_tokens, num_heads_q,
        num_heads_k, num_heads_v, static_cast<float>(eps),
        reinterpret_cast<const __half*>(q_weight.data_ptr()),
        reinterpret_cast<const __half*>(k_weight.data_ptr()),
        reinterpret_cast<const __half*>(cos_sin_cache.data_ptr()), is_neox,
        reinterpret_cast<const int64_t*>(position_ids.data_ptr()), head_dim,
        rotary_dim, stream);
  } else {
    TORCH_CHECK(false, "Unsupported dtype for musa_fused_qk_norm_rope");
  }
}
