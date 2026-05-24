// MUSA-0162 V2: native MUSA port of topk_softmax (MoE routing).
//
// V1 (32f31a7b6) used 1 CTA per token = 4096 CTAs at T=4096; regressed 50%
// vs upstream at large T due to launch overhead (each CTA has small compute
// budget E=128, K=8 = ~5µs of work per CTA).
//
// V2 strategy: TOKENS_PER_BLOCK = 4 tokens per CTA → 1024 CTAs at T=4096,
// each CTA has 4 * E = 512 threads (one warp group per token). Per-token
// reductions stay within the token's 128-thread group via warp shuffles +
// per-token smem entries.

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <torch/all.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace vllm_musa {

template <int E, int K, int TOKENS_PER_BLOCK>
__global__ void __launch_bounds__(E* TOKENS_PER_BLOCK, 1)
    topk_softmax_kernel_v2(
        float* __restrict__ topk_weights,    // [T, K]
        int* __restrict__ topk_indices,      // [T, K]
        int* __restrict__ token_expert_idx,  // [T, K]
        const float* __restrict__ gating,    // [T, E]
        int num_tokens, bool renormalize) {
  // Each CTA handles TOKENS_PER_BLOCK tokens.
  // Threads in the CTA are split into TOKENS_PER_BLOCK groups of E threads each.
  // Group g handles token (blockIdx.x * TOKENS_PER_BLOCK + g).
  const int local_tok = threadIdx.x / E;
  const int tid_in_token = threadIdx.x % E;
  const int global_tok = blockIdx.x * TOKENS_PER_BLOCK + local_tok;
  if (global_tok >= num_tokens) return;

  // Per-token smem rows
  __shared__ float smem_softmax[TOKENS_PER_BLOCK][E];
  __shared__ float smem_chosen[TOKENS_PER_BLOCK][K];
  __shared__ int   smem_chosen_idx[TOKENS_PER_BLOCK][K];
  __shared__ float smem_max[TOKENS_PER_BLOCK];
  __shared__ float smem_sum[TOKENS_PER_BLOCK];

  const float my_val = gating[global_tok * E + tid_in_token];

  // Softmax max (within 128-thread per-token group)
  float m = my_val;
  for (int o = 16; o > 0; o >>= 1) {
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
  }
  // Cross-warp reduction within token group
  if constexpr (E > 32) {
    if ((tid_in_token & 31) == 0) {
      smem_softmax[local_tok][tid_in_token >> 5] = m;
    }
    __syncthreads();
    if (tid_in_token < (E / 32)) {
      m = smem_softmax[local_tok][tid_in_token];
      for (int o = (E / 32) / 2; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
      }
      if (tid_in_token == 0) smem_max[local_tok] = m;
    }
    __syncthreads();
    m = smem_max[local_tok];
  }

  // Softmax sum
  const float e_val = expf(my_val - m);
  float s = e_val;
  for (int o = 16; o > 0; o >>= 1) {
    s += __shfl_xor_sync(0xffffffff, s, o);
  }
  if constexpr (E > 32) {
    if ((tid_in_token & 31) == 0) {
      smem_softmax[local_tok][tid_in_token >> 5] = s;
    }
    __syncthreads();
    if (tid_in_token < (E / 32)) {
      s = smem_softmax[local_tok][tid_in_token];
      for (int o = (E / 32) / 2; o > 0; o >>= 1) {
        s += __shfl_xor_sync(0xffffffff, s, o);
      }
      if (tid_in_token == 0) smem_sum[local_tok] = s;
    }
    __syncthreads();
    s = smem_sum[local_tok];
  }

  const float my_softmax = e_val / s;
  smem_softmax[local_tok][tid_in_token] = my_softmax;
  __syncthreads();

  // Top-K via sequential max-and-mask using 1 warp per token (lower-id warp of the token group)
  if (tid_in_token < 32) {
    for (int k = 0; k < K; ++k) {
      float local_max = -1e30f;
      int local_idx = -1;
#pragma unroll
      for (int j = tid_in_token; j < E; j += 32) {
        const float v = smem_softmax[local_tok][j];
        if (v > local_max) { local_max = v; local_idx = j; }
      }
      float w_max = local_max;
      int w_idx = local_idx;
      for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffff, w_max, o);
        const int oi = __shfl_xor_sync(0xffffffff, w_idx, o);
        if (ov > w_max || (ov == w_max && oi < w_idx)) {
          w_max = ov; w_idx = oi;
        }
      }
      if (tid_in_token == 0) {
        smem_chosen[local_tok][k] = w_max;
        smem_chosen_idx[local_tok][k] = w_idx;
        smem_softmax[local_tok][w_idx] = -1e30f;
      }
    }
  }
  __syncthreads();

  // Optional renormalize
  float renorm = 1.0f;
  if (renormalize && tid_in_token == 0) {
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < K; ++k) ss += smem_chosen[local_tok][k];
    renorm = (ss > 0.0f) ? (1.0f / ss) : 1.0f;
  }
  if (renormalize) {
    renorm = __shfl_sync(0xffffffff, renorm, 0);
  }

  // Write outputs
  if (tid_in_token < K) {
    const int o_idx = global_tok * K + tid_in_token;
    topk_weights[o_idx] = smem_chosen[local_tok][tid_in_token] * renorm;
    topk_indices[o_idx] = smem_chosen_idx[local_tok][tid_in_token];
    token_expert_idx[o_idx] = global_tok * K + tid_in_token;
  }
}

}  // namespace vllm_musa

void musa_topk_softmax(torch::Tensor& topk_weights, torch::Tensor& topk_indices,
                       torch::Tensor& token_expert_indices,
                       torch::Tensor const& gating_output, bool renormalize) {
  TORCH_CHECK(gating_output.device().is_privateuseone(),
              "gating must be a MUSA tensor");
  TORCH_CHECK(gating_output.is_contiguous(), "gating must be contiguous");
  TORCH_CHECK(topk_weights.is_contiguous() && topk_indices.is_contiguous(),
              "outputs must be contiguous");
  TORCH_CHECK(gating_output.dim() == 2, "gating must be 2-D (T, E)");
  TORCH_CHECK(gating_output.dtype() == c10::ScalarType::Float,
              "gating must be FP32");
  TORCH_CHECK(topk_weights.dtype() == c10::ScalarType::Float,
              "topk_weights must be FP32");

  const int T = gating_output.size(0);
  const int E = gating_output.size(1);
  const int K = topk_weights.size(1);

  const c10::musa::OptionalMUSAGuard guard(device_of(gating_output));
  musaStream_t stream = c10::musa::getCurrentMUSAStream().stream();

  // V2: 4 tokens per CTA, threads_per_CTA = E * 4
  if (E == 128 && K == 8) {
    constexpr int TPB = 4;
    const int grid = (T + TPB - 1) / TPB;
    vllm_musa::topk_softmax_kernel_v2<128, 8, TPB>
        <<<grid, 128 * TPB, 0, stream>>>(
            topk_weights.data_ptr<float>(), topk_indices.data_ptr<int>(),
            token_expert_indices.data_ptr<int>(), gating_output.data_ptr<float>(),
            T, renormalize);
  } else if (E == 64 && K == 6) {
    constexpr int TPB = 8;
    const int grid = (T + TPB - 1) / TPB;
    vllm_musa::topk_softmax_kernel_v2<64, 6, TPB>
        <<<grid, 64 * TPB, 0, stream>>>(
            topk_weights.data_ptr<float>(), topk_indices.data_ptr<int>(),
            token_expert_indices.data_ptr<int>(), gating_output.data_ptr<float>(),
            T, renormalize);
  } else if (E == 256 && K == 8) {
    constexpr int TPB = 2;
    const int grid = (T + TPB - 1) / TPB;
    vllm_musa::topk_softmax_kernel_v2<256, 8, TPB>
        <<<grid, 256 * TPB, 0, stream>>>(
            topk_weights.data_ptr<float>(), topk_indices.data_ptr<int>(),
            token_expert_indices.data_ptr<int>(), gating_output.data_ptr<float>(),
            T, renormalize);
  } else {
    TORCH_CHECK(false, "musa_topk_softmax V2: unsupported (E=", E,
                ", K=", K, "); supported: (128,8) (64,6) (256,8)");
  }
}
