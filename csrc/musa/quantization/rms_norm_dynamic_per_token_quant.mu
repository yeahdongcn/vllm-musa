// MUSA-0159: native MUSA port of rms_norm_dynamic_per_token_quant.
//
// Replaces upstream csrc/quantization/fused_kernels/
// fused_layernorm_dynamic_per_token_quant.cu which is 30-40x too slow on
// MUSA (1% of GDDR6 roofline) because the "vectorized" helpers in
// layernorm_utils.cuh don't lower well on mcc.
//
// Design: copied from MUSA-0150 fused_add_rmsnorm.mu V14 (85% roofline).
//   - One block per token (grid = num_tokens).
//   - vload16_byp_slc LSU-bypass 128-bit loads of input.
//   - __musa_memcpy_g2s async load of weight into smem.
//   - Single pass over gmem: register-resident input array.
//   - Two reductions over registers: variance for inv_rms, then
//     normed-absmax for token_scale.
//   - FP8 cast inline using static_cast<c10::Float8_e4m3fn>(float) (same as
//     MUSA-0152 per_token_group_quant).
//
// Empirical diagnostic 2026-05-24 on yeahdongcn60:
//   upstream M=4096 N=6144:    6377 µs / 11.8 GB/s / 0.99 %
//   torch-native composition:  1369 µs / 55.2 GB/s / 4.60 %
//   target (this kernel):       ~80 µs / 960 GB/s / 80 %  (matches MUSA-0150)

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <torch/all.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

#include <cstdlib>

#include "../vec_utils.muh"

extern "C" {
extern __device__ void __musa_memcpy_g2s(
    __attribute__((address_space(3))) void* dst,
    __attribute__((address_space(1))) const void* src, int size, int prefetch);
extern __device__ void __musa_memcpy_g2s_wait();
}

namespace vllm_musa {

template <int BLOCK_X>
__device__ __forceinline__ float block_reduce_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
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
      value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    if (threadIdx.x == 0) {
      shared[0] = value;
    }
  }
  __syncthreads();
  return shared[0];
}

template <int BLOCK_X>
__device__ __forceinline__ float block_reduce_max(float value) {
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

// V5: multi-row-per-CTA. Same pattern as MUSA-0156 V5 (RPB=4 sweet spot).
// Weight smem load shared across all rows. Each row runs the variance +
// absmax + quant pipeline serially. Reduces CTA count 4× at M=4096.
template <typename T, int BLOCK_X, int CHUNK, int ROWS_PER_BLOCK>
__global__ void __launch_bounds__(BLOCK_X, 1)
    rms_norm_dynamic_per_token_quant_kernel(
        c10::Float8_e4m3fn* __restrict__ out, float* __restrict__ scales,
        const T* __restrict__ input, const T* __restrict__ weight,
        int num_rows, int hidden_size, int vec_hidden_size, float epsilon) {
  constexpr int VEC_SIZE = 8;
  constexpr float FP8_MAX = 448.0f;
  constexpr float MIN_SCALE = 1e-12f;

  const int tid = threadIdx.x;

  __shared__ T shared_weight[CHUNK * BLOCK_X * VEC_SIZE];

  // Async G2S load of weight — loaded ONCE per CTA, reused across all rows
#pragma unroll
  for (int chunk = 0; chunk < CHUNK; ++chunk) {
    const int vec_idx = chunk * BLOCK_X + tid;
    if (vec_idx < vec_hidden_size) {
      __attribute__((address_space(3))) void* smem_ptr =
          (__attribute__((address_space(3))) void*)(
              shared_weight + vec_idx * VEC_SIZE);
      __attribute__((address_space(1))) const void* gmem_ptr =
          (__attribute__((address_space(1))) const void*)(
              weight + vec_idx * VEC_SIZE);
      __musa_memcpy_g2s(smem_ptr, gmem_ptr, 16, 128);
    }
  }
  __musa_memcpy_g2s_wait();
  __syncthreads();

#pragma unroll
  for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
    const int row = blockIdx.x * ROWS_PER_BLOCK + r;
    if (row >= num_rows) break;

    const T* input_row = input + static_cast<size_t>(row) * hidden_size;
    c10::Float8_e4m3fn* output_row =
        out + static_cast<size_t>(row) * hidden_size;

    // Pass 1: vload input → register, compute variance
    Vec16<T> input_local[CHUNK];
    float variance = 0.0f;
#pragma unroll
    for (int chunk = 0; chunk < CHUNK; ++chunk) {
      const int vec_idx = chunk * BLOCK_X + tid;
      if (vec_idx < vec_hidden_size) {
        Vec16<T> v = vload16_byp_slc(input_row + vec_idx * VEC_SIZE);
        input_local[chunk] = v;
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          const float val = static_cast<float>(v.x[i]);
          variance += val * val;
        }
      }
    }

    const float total_sq = block_reduce_sum<BLOCK_X>(variance);
    const float inv_rms =
        rsqrtf(total_sq / static_cast<float>(hidden_size) + epsilon);

    // Pass 2: compute normed = input * inv_rms * weight, find absmax
    float normed_local[CHUNK * VEC_SIZE];
    float local_max = MIN_SCALE;
#pragma unroll
    for (int chunk = 0; chunk < CHUNK; ++chunk) {
      const int vec_idx = chunk * BLOCK_X + tid;
      if (vec_idx < vec_hidden_size) {
        Vec16<T> w = *reinterpret_cast<Vec16<T>*>(shared_weight +
                                                  vec_idx * VEC_SIZE);
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          const float v = static_cast<float>(input_local[chunk].x[i]) *
                          inv_rms * static_cast<float>(w.x[i]);
          normed_local[chunk * VEC_SIZE + i] = v;
          const float a = fabsf(v);
          local_max = fmaxf(local_max, a);
        }
      }
    }

    const float token_max = block_reduce_max<BLOCK_X>(local_max);
    const float token_scale = fmaxf(token_max / FP8_MAX, MIN_SCALE);

    if (tid == 0) {
      scales[row] = token_scale;
    }

    const float inv_scale = 1.0f / token_scale;

    // Pass 3: quantize via hw FP8 intrinsic + uint64 store.
#pragma unroll
    for (int chunk = 0; chunk < CHUNK; ++chunk) {
      const int vec_idx = chunk * BLOCK_X + tid;
      if (vec_idx < vec_hidden_size) {
        float scaled[VEC_SIZE];
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          scaled[i] = normed_local[chunk * VEC_SIZE + i] * inv_scale;
        }
        const uint64_t packed = pack8_floats_to_fp8e4m3(scaled);
        *reinterpret_cast<uint64_t*>(output_row + vec_idx * VEC_SIZE) = packed;
      }
    }
    // syncthreads between row iterations resets smem reduction buffer state
    __syncthreads();
  }
}

template <typename T>
void dispatch_rms_norm_dynamic_per_token_quant(
    c10::Float8_e4m3fn* out, float* scales, const T* input, const T* weight,
    int rows, int hidden_size, float epsilon, musaStream_t stream) {
  const int vec_hidden_size = hidden_size / 8;
  // V6: RPB=1 keeper (V5 RPB=4 sweep showed regression because this kernel
  // has 2 reductions/row + extra absmax compared to MUSA-0156's 1
  // reduction). The extra reduction means more syncthreads per row, and
  // serializing 4 rows in a CTA compounds the cost. Best to keep RPB=1
  // and rely on hw FP8 cast intrinsic (+11 pp, V4 = 45.64%%).
  constexpr int RPB_SMALL = 1;

  int rpb, block_x;
  rpb = RPB_SMALL;
  if (rows >= 256) {
    block_x = 256;
  } else if (rows >= 8 && vec_hidden_size >= 1280) {
    block_x = 512;
  } else {
    block_x = 1024;
  }
  const int grid = (rows + rpb - 1) / rpb;

#define LAUNCH(BLOCK, CHUNK, RPB)                                              \
  rms_norm_dynamic_per_token_quant_kernel<T, BLOCK, CHUNK, RPB>                \
      <<<grid, BLOCK, 0, stream>>>(out, scales, input, weight, rows,           \
                                   hidden_size, vec_hidden_size, epsilon)

  if (block_x == 256) {
    if (vec_hidden_size <= 256) LAUNCH(256, 1, RPB_SMALL);
    else if (vec_hidden_size <= 512) LAUNCH(256, 2, RPB_SMALL);
    else if (vec_hidden_size <= 768) LAUNCH(256, 3, RPB_SMALL);
    else if (vec_hidden_size <= 1024) LAUNCH(256, 4, RPB_SMALL);
    else if (vec_hidden_size <= 1536) LAUNCH(256, 6, RPB_SMALL);
    else if (vec_hidden_size <= 2048) LAUNCH(256, 8, RPB_SMALL);
    else TORCH_CHECK(false, "BLOCK=256 hidden_size too large");
  } else if (block_x == 512) {
    if (vec_hidden_size <= 512) LAUNCH(512, 1, RPB_SMALL);
    else if (vec_hidden_size <= 1024) LAUNCH(512, 2, RPB_SMALL);
    else if (vec_hidden_size <= 1536) LAUNCH(512, 3, RPB_SMALL);
    else if (vec_hidden_size <= 2048) LAUNCH(512, 4, RPB_SMALL);
    else TORCH_CHECK(false, "BLOCK=512 hidden_size too large");
  } else {
    if (vec_hidden_size <= 1024) LAUNCH(1024, 1, RPB_SMALL);
    else if (vec_hidden_size <= 2048) LAUNCH(1024, 2, RPB_SMALL);
    else TORCH_CHECK(false, "BLOCK=1024 hidden_size too large");
  }

#undef LAUNCH
}

}  // namespace vllm_musa

void musa_rms_norm_dynamic_per_token_quant(torch::Tensor& out,
                                           torch::Tensor const& input,
                                           torch::Tensor const& weight,
                                           torch::Tensor& scales,
                                           double epsilon) {
  TORCH_CHECK(input.device().is_privateuseone(), "input must be a MUSA tensor");
  TORCH_CHECK(weight.device().is_privateuseone(),
              "weight must be a MUSA tensor");
  TORCH_CHECK(out.device().is_privateuseone(), "out must be a MUSA tensor");
  TORCH_CHECK(scales.device().is_privateuseone(),
              "scales must be a MUSA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(out.dtype() == c10::ScalarType::Float8_e4m3fn,
              "out must be FP8 e4m3");
  TORCH_CHECK(scales.dtype() == c10::ScalarType::Float, "scales must be FP32");
  TORCH_CHECK(input.dim() == 2 || input.dim() == 3, "input must be 2-D or 3-D");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1-D");
  TORCH_CHECK(input.size(-1) == weight.size(0),
              "weight size mismatch with input last dim");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
              "input/weight dtype mismatch");
  TORCH_CHECK(input.size(-1) % 8 == 0,
              "hidden_size must be a multiple of 8");
  TORCH_CHECK(input.size(-1) <= 16384,
              "hidden_size must be <= 16384");

  const c10::musa::OptionalMUSAGuard guard(device_of(input));
  musaStream_t stream = c10::musa::getCurrentMUSAStream().stream();

  const int hidden_size = input.size(-1);
  const int rows = input.numel() / hidden_size;

  if (input.scalar_type() == c10::ScalarType::BFloat16) {
    vllm_musa::dispatch_rms_norm_dynamic_per_token_quant<__mt_bfloat16>(
        reinterpret_cast<c10::Float8_e4m3fn*>(out.data_ptr()),
        scales.data_ptr<float>(),
        reinterpret_cast<const __mt_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const __mt_bfloat16*>(weight.data_ptr()),
        rows, hidden_size, static_cast<float>(epsilon), stream);
  } else if (input.scalar_type() == c10::ScalarType::Half) {
    vllm_musa::dispatch_rms_norm_dynamic_per_token_quant<__half>(
        reinterpret_cast<c10::Float8_e4m3fn*>(out.data_ptr()),
        scales.data_ptr<float>(),
        reinterpret_cast<const __half*>(input.data_ptr()),
        reinterpret_cast<const __half*>(weight.data_ptr()),
        rows, hidden_size, static_cast<float>(epsilon), stream);
  } else {
    TORCH_CHECK(false, "Unsupported input dtype for "
                       "musa_rms_norm_dynamic_per_token_quant");
  }
}
