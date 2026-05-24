// MUSA-0158: native MUSA port of silu_and_mul_per_block_quant.
//
// Replaces upstream csrc/quantization/fused_kernels/
// fused_silu_mul_block_quant.cu which is 30x too slow on MUSA (2.3% of GDDR6
// roofline at M=4096 N=6144 G=128: 4530 µs vs target ~150 µs).
//
// Design: copied from MUSA-0150 fused_add_rmsnorm.mu V14 + MUSA-0159 RMSNorm
// + FP8 quant pattern. One block per (token, group). Single-pass register-
// resident absmax over the GROUP. FP8 cast inline.

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <torch/all.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

#include "../vec_utils.muh"

namespace vllm_musa {

template <int BLOCK_X>
__device__ __forceinline__ float block_reduce_max_silu(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
  }
  if constexpr (BLOCK_X <= 32) {
    return value;
  }

  __shared__ float shared[BLOCK_X / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();

  value = threadIdx.x < (BLOCK_X / 32) ? shared[threadIdx.x] : 0.0f;
  if (warp == 0) {
    for (int offset = (BLOCK_X / 32) >> 1; offset > 0; offset >>= 1) {
      value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
    }
    if (threadIdx.x == 0) {
      shared[0] = value;
    }
  }
  __syncthreads();
  return shared[0];
}

// V4: multi-group-per-CTA. Each warp handles 1 group independently.
// Reductions are warp-only (no cross-warp smem sync needed since each warp
// processes a different group).
// Grid: (num_tokens, num_groups / GROUPS_PER_CTA). BLOCK_X = 32 * GROUPS_PER_CTA.
// At M=4096 N=6144 G=128 GROUPS_PER_CTA=4: 4096 × 12 = 49k CTAs (vs 196k in V3).
template <typename T, int GROUP_SIZE, int BLOCK_X, int GROUPS_PER_CTA = 1>
__global__ void __launch_bounds__(BLOCK_X, 1)
    silu_and_mul_per_block_quant_kernel(
        c10::Float8_e4m3fn* __restrict__ out,  // [num_tokens, hidden]
        float* __restrict__ scales,            // [num_tokens, num_groups]
        const T* __restrict__ input,           // [num_tokens, 2*hidden]
        int hidden_size,
        int num_groups_total) {
  constexpr int THREADS_PER_GROUP = BLOCK_X / GROUPS_PER_CTA;
  constexpr int ELEMS_PER_THREAD = GROUP_SIZE / THREADS_PER_GROUP;
  constexpr float FP8_MAX = 448.0f;
  constexpr float MIN_SCALE = 1e-12f;

  const int token_idx = blockIdx.x;
  const int local_group = threadIdx.x / THREADS_PER_GROUP;
  const int tid = threadIdx.x % THREADS_PER_GROUP;
  const int group_idx = blockIdx.y * GROUPS_PER_CTA + local_group;
  if (group_idx >= num_groups_total) return;
  const int num_groups = num_groups_total;

  const int input_stride = hidden_size * 2;
  const int group_start = group_idx * GROUP_SIZE;
  const T* gate_in = input + token_idx * input_stride + group_start;
  const T* up_in = gate_in + hidden_size;
  c10::Float8_e4m3fn* group_out =
      out + token_idx * hidden_size + group_start;

  // Pass 1: per-thread silu(gate)*up, cache to register, find local absmax
  float values[ELEMS_PER_THREAD];
  float local_max = MIN_SCALE;

#pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
    const int col = tid * ELEMS_PER_THREAD + i;
    const float gate = static_cast<float>(gate_in[col]);
    const float up = static_cast<float>(up_in[col]);
    // SiLU(gate) * up
    const float sig = 1.0f / (1.0f + expf(-gate));
    const float v = gate * sig * up;
    values[i] = v;
    local_max = fmaxf(local_max, fabsf(v));
  }

  // Warp-only reduce (each warp = independent group, no cross-warp sync).
  // THREADS_PER_GROUP is always a power of 2 in the 32-lane warp.
  float group_max = local_max;
#pragma unroll
  for (int o = THREADS_PER_GROUP / 2; o > 0; o >>= 1) {
    group_max = fmaxf(group_max, __shfl_xor_sync(0xffffffff, group_max, o));
  }
  const float group_scale = fmaxf(group_max / FP8_MAX, MIN_SCALE);

  if (tid == 0) {
    scales[token_idx * num_groups + group_idx] = group_scale;
  }

  const float inv_scale = 1.0f / group_scale;

  // Pass 2: quantize to FP8 via hardware intrinsic.
  // V2: each thread holds ELEMS_PER_THREAD=4 floats. With BLOCK_X=32 threads
  // and GROUP_SIZE=128, each thread handles 4 contiguous FP8 elements.
  // `__musa_cvt_float4_to_fp8x4` packs them into a single uint32 = 4 bytes.
  // Replaces 4× c10 software cast → 1× hw cvt instruction.
  float scaled[ELEMS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
    scaled[i] = values[i] * inv_scale;
  }
  if constexpr (ELEMS_PER_THREAD == 4) {
    const float4 f4 = make_float4(scaled[0], scaled[1], scaled[2], scaled[3]);
    const __mt_fp8x4_storage_t packed =
        __musa_cvt_float4_to_fp8x4(f4, __MT_SATFINITE, __MT_E4M3);
    *reinterpret_cast<__mt_fp8x4_storage_t*>(
        group_out + tid * ELEMS_PER_THREAD) = packed;
  } else {
    // Fallback: scalar path for other ELEMS_PER_THREAD shapes
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
      const float clamped = fmaxf(-FP8_MAX, fminf(scaled[i], FP8_MAX));
      group_out[tid * ELEMS_PER_THREAD + i] =
          static_cast<c10::Float8_e4m3fn>(clamped);
    }
  }
}

template <typename T>
void dispatch_silu_and_mul_per_block_quant(c10::Float8_e4m3fn* out,
                                            float* scales, const T* input,
                                            int num_tokens, int hidden_size,
                                            int group_size,
                                            musaStream_t stream) {
  TORCH_CHECK(group_size == 128, "group_size must be 128 for MUSA port");
  TORCH_CHECK(hidden_size % group_size == 0, "hidden_size must divide group_size");
  const int num_groups = hidden_size / group_size;
  // V4 sweet spot: GROUPS_PER_CTA=4 → 4 warps per CTA, 4× fewer CTAs.
  // V5 GROUPS_PER_CTA=8 measured at 38.64%% — flat within noise of V4
  // (38.84%%); larger CTAs saturated SM occupancy. Keep V4 GROUPS_PER_CTA=4.
  constexpr int GROUPS_PER_CTA = 4;
  constexpr int BLOCK_X = 32 * GROUPS_PER_CTA;  // 128 threads = 4 warps
  const int grid_y = (num_groups + GROUPS_PER_CTA - 1) / GROUPS_PER_CTA;
  dim3 grid(num_tokens, grid_y);
  silu_and_mul_per_block_quant_kernel<T, 128, BLOCK_X, GROUPS_PER_CTA>
      <<<grid, BLOCK_X, 0, stream>>>(out, scales, input, hidden_size, num_groups);
}

}  // namespace vllm_musa

void musa_silu_and_mul_per_block_quant(torch::Tensor& out,
                                       torch::Tensor const& input,
                                       torch::Tensor& scales,
                                       int64_t group_size) {
  TORCH_CHECK(input.device().is_privateuseone(), "input must be a MUSA tensor");
  TORCH_CHECK(out.device().is_privateuseone(), "out must be a MUSA tensor");
  TORCH_CHECK(scales.device().is_privateuseone(),
              "scales must be a MUSA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(out.dtype() == c10::ScalarType::Float8_e4m3fn,
              "out must be FP8 e4m3");
  TORCH_CHECK(scales.dtype() == c10::ScalarType::Float, "scales must be FP32");
  TORCH_CHECK(input.dim() == 2 && out.dim() == 2, "input and out must be 2-D");

  const c10::musa::OptionalMUSAGuard guard(device_of(input));
  musaStream_t stream = c10::musa::getCurrentMUSAStream().stream();

  const int num_tokens = input.size(0);
  const int hidden_size = out.size(-1);
  TORCH_CHECK(input.size(-1) == hidden_size * 2,
              "input last dim must be 2x out last dim");

  if (input.scalar_type() == c10::ScalarType::BFloat16) {
    vllm_musa::dispatch_silu_and_mul_per_block_quant<__mt_bfloat16>(
        reinterpret_cast<c10::Float8_e4m3fn*>(out.data_ptr()),
        scales.data_ptr<float>(),
        reinterpret_cast<const __mt_bfloat16*>(input.data_ptr()),
        num_tokens, hidden_size, static_cast<int>(group_size), stream);
  } else if (input.scalar_type() == c10::ScalarType::Half) {
    vllm_musa::dispatch_silu_and_mul_per_block_quant<__half>(
        reinterpret_cast<c10::Float8_e4m3fn*>(out.data_ptr()),
        scales.data_ptr<float>(),
        reinterpret_cast<const __half*>(input.data_ptr()),
        num_tokens, hidden_size, static_cast<int>(group_size), stream);
  } else {
    TORCH_CHECK(false, "Unsupported input dtype for "
                       "musa_silu_and_mul_per_block_quant");
  }
}
