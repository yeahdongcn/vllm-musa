// MUSA-0161 V2: native MUSA port of merge_attn_states (rewrite for large T).
//
// V1 (e50de0b63) used 1 CTA per (token, head) = 131072 CTAs at T=4096 H=32,
// which beat upstream at small T but lost ~9% at T=4096 (launch overhead).
//
// V2 strategy: match upstream's 1D grid layout — each thread handles
// head_size/K elements for one (token, head). With K=8 threads-per-head and
// BLOCK_X=128, each thread does 1 Vec16 load. At T=4096 H=32:
//   total_threads = 4096 * 32 * 8 = 1,048,576
//   grid = 8192 CTAs × 128 threads/CTA  (16× fewer CTAs than V1)

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <torch/all.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

#include "../vec_utils.muh"

namespace vllm_musa {

template <typename T, int HEAD_DIM, int THREADS_PER_HEAD, int BLOCK_X>
__global__ void __launch_bounds__(BLOCK_X, 1) merge_attn_states_kernel_v3(
    T* __restrict__ output, float* __restrict__ output_lse,
    const T* __restrict__ prefix_output, const float* __restrict__ prefix_lse,
    const T* __restrict__ suffix_output, const float* __restrict__ suffix_lse,
    int num_tokens, int num_heads) {
  constexpr int VEC_SIZE = 8;
  constexpr int ELEMS_PER_THREAD = HEAD_DIM / THREADS_PER_HEAD;
  constexpr int VECS_PER_THREAD = ELEMS_PER_THREAD / VEC_SIZE;

  const int gtid = blockIdx.x * BLOCK_X + threadIdx.x;
  const int total_thread_heads =
      num_tokens * num_heads * THREADS_PER_HEAD;
  if (gtid >= total_thread_heads) return;

  const int thread_in_head = gtid % THREADS_PER_HEAD;
  const int token_head = gtid / THREADS_PER_HEAD;
  const int head_idx = token_head % num_heads;
  const int token_idx = token_head / num_heads;

  const int lse_idx = token_idx * num_heads + head_idx;
  const float p_lse = prefix_lse[lse_idx];
  const float s_lse = suffix_lse[lse_idx];
  const float p_max = fmaxf(p_lse, s_lse);
  const float p_scale = expf(p_lse - p_max);
  const float s_scale = expf(s_lse - p_max);
  const float z = p_scale + s_scale;
  const float inv_z = 1.0f / z;
  const float p_norm = p_scale * inv_z;
  const float s_norm = s_scale * inv_z;

  if (output_lse != nullptr && thread_in_head == 0) {
    output_lse[lse_idx] = logf(z) + p_max;
  }

  const size_t base =
      (static_cast<size_t>(token_idx) * num_heads + head_idx) * HEAD_DIM;

#pragma unroll
  for (int v = 0; v < VECS_PER_THREAD; ++v) {
    const int vec_off = (thread_in_head * VECS_PER_THREAD + v) * VEC_SIZE;
    Vec16<T> p_vec = vload16_byp_slc(prefix_output + base + vec_off);
    Vec16<T> s_vec = vload16_byp_slc(suffix_output + base + vec_off);
    Vec16<T> out_vec;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      const float pv = static_cast<float>(p_vec.x[i]);
      const float sv = static_cast<float>(s_vec.x[i]);
      out_vec.x[i] = static_cast<T>(pv * p_norm + sv * s_norm);
    }
    vstore16(output + base + vec_off, out_vec);
  }
}

template <typename T>
void dispatch_merge_attn_states_v2(T* output, float* output_lse,
                                   const T* prefix_output,
                                   const float* prefix_lse,
                                   const T* suffix_output,
                                   const float* suffix_lse, int num_tokens,
                                   int num_heads, int head_dim,
                                   musaStream_t stream) {
  TORCH_CHECK(head_dim == 128, "head_dim must be 128 for MUSA port V2");
  constexpr int HEAD_DIM = 128;
  // Complete BLOCK_X sweep (T=4096 H=32):
  // V5 BLOCK_X=64:    73.84 %% (within noise of V2; smaller CTAs didn't help)
  // V2 BLOCK_X=128:   73.68 %% ← keeper (best by tiniest margin, established)
  // V4 BLOCK_X=256:   60.53 %% (regressed — too few CTAs in flight)
  // V3 TPH=8:         17.22 %% (regressed — register pressure)
  // Conclusion: this kernel is fundamentally launch-overhead+sync bound at
  // 73.68 %% with the current TPH=16/BLOCK_X=128 layout. BLOCK_X tuning
  // saturated. Remaining 6.32 pt gap to 80 %% bar would need async copy or
  // a different reduction strategy.
  constexpr int THREADS_PER_HEAD = 16;  // 128/16 = 8 elements/thread = 1 Vec16
  constexpr int BLOCK_X = 128;
  const int total_thread_heads = num_tokens * num_heads * THREADS_PER_HEAD;
  const int grid = (total_thread_heads + BLOCK_X - 1) / BLOCK_X;
  merge_attn_states_kernel_v3<T, HEAD_DIM, THREADS_PER_HEAD, BLOCK_X>
      <<<grid, BLOCK_X, 0, stream>>>(output, output_lse, prefix_output,
                                     prefix_lse, suffix_output, suffix_lse,
                                     num_tokens, num_heads);
}

}  // namespace vllm_musa

void musa_merge_attn_states(torch::Tensor& output,
                            std::optional<torch::Tensor> output_lse,
                            torch::Tensor const& prefix_output,
                            torch::Tensor const& prefix_lse,
                            torch::Tensor const& suffix_output,
                            torch::Tensor const& suffix_lse) {
  TORCH_CHECK(output.device().is_privateuseone(),
              "output must be a MUSA tensor");
  TORCH_CHECK(output.is_contiguous() && prefix_output.is_contiguous() &&
                  suffix_output.is_contiguous(),
              "outputs must be contiguous");
  TORCH_CHECK(output.dim() == 3 && prefix_output.dim() == 3 &&
                  suffix_output.dim() == 3,
              "outputs must be 3-D (T, H, D)");
  TORCH_CHECK(prefix_lse.dim() == 2 && suffix_lse.dim() == 2,
              "lse tensors must be 2-D (T, H)");
  TORCH_CHECK(prefix_lse.dtype() == c10::ScalarType::Float &&
                  suffix_lse.dtype() == c10::ScalarType::Float,
              "lse tensors must be FP32");

  const c10::musa::OptionalMUSAGuard guard(device_of(output));
  musaStream_t stream = c10::musa::getCurrentMUSAStream().stream();

  const int num_tokens = output.size(0);
  const int num_heads = output.size(1);
  const int head_dim = output.size(2);
  float* lse_ptr =
      output_lse.has_value() ? output_lse->data_ptr<float>() : nullptr;

  if (output.scalar_type() == c10::ScalarType::BFloat16) {
    vllm_musa::dispatch_merge_attn_states_v2<__mt_bfloat16>(
        reinterpret_cast<__mt_bfloat16*>(output.data_ptr()), lse_ptr,
        reinterpret_cast<const __mt_bfloat16*>(prefix_output.data_ptr()),
        prefix_lse.data_ptr<float>(),
        reinterpret_cast<const __mt_bfloat16*>(suffix_output.data_ptr()),
        suffix_lse.data_ptr<float>(), num_tokens, num_heads, head_dim, stream);
  } else if (output.scalar_type() == c10::ScalarType::Half) {
    vllm_musa::dispatch_merge_attn_states_v2<__half>(
        reinterpret_cast<__half*>(output.data_ptr()), lse_ptr,
        reinterpret_cast<const __half*>(prefix_output.data_ptr()),
        prefix_lse.data_ptr<float>(),
        reinterpret_cast<const __half*>(suffix_output.data_ptr()),
        suffix_lse.data_ptr<float>(), num_tokens, num_heads, head_dim, stream);
  } else {
    TORCH_CHECK(false, "Unsupported dtype for musa_merge_attn_states");
  }
}
