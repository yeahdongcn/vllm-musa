// MUSA-0156: native MUSA port of rms_norm_static_fp8_quant.
//
// Replaces upstream csrc/layernorm_quant_kernels.cu (rms_norm_static_fp8_quant
// fn) which is at 17.1 % of GDDR6 roofline on MUSA (367 µs at M=4096 N=6144).
//
// Design: same as MUSA-0159 V14 pattern but SIMPLER — scale is provided
// (not computed), so we only need ONE block reduction (variance) instead of
// two. This means we should match MUSA-0150's 85 % more closely.

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <torch/all.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

#include "../vec_utils.muh"

extern "C" {
extern __device__ void __musa_memcpy_g2s(
    __attribute__((address_space(3))) void* dst,
    __attribute__((address_space(1))) const void* src, int size, int prefetch);
extern __device__ void __musa_memcpy_g2s_wait();
}

namespace vllm_musa {

template <int BLOCK_X>
__device__ __forceinline__ float static_quant_block_reduce_sum(float value) {
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

// V3: multi-row-per-CTA. Each CTA processes ROWS_PER_BLOCK consecutive rows
// in a serial loop, reusing the smem weight load across rows. The kernel
// shape is `<<<(num_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, BLOCK_X>>>` —
// at M=4096 ROWS_PER_BLOCK=4 → 1024 CTAs (4× fewer than V2).
//
// Why this works: V2 was at 53 % roofline. With M=4096 there are 4096 CTAs
// each doing ~62.5 µs/CTA of bandwidth-bound work + ~7 µs of fixed
// per-CTA cost (sync, setup, weight load). The fixed cost dominates 12 %
// of the time. Amortizing it across 4 rows cuts the fixed share to 3 %.
template <typename T, int BLOCK_X, int CHUNK, int ROWS_PER_BLOCK>
__global__ void __launch_bounds__(BLOCK_X, 1)
    rms_norm_static_fp8_quant_kernel(
        c10::Float8_e4m3fn* __restrict__ out, const T* __restrict__ input,
        const T* __restrict__ weight, const float* __restrict__ scale,
        int num_rows, int hidden_size, int vec_hidden_size, float epsilon) {
  constexpr int VEC_SIZE = 8;

  const int tid = threadIdx.x;

  __shared__ T shared_weight[CHUNK * BLOCK_X * VEC_SIZE];

  // Async G2S load of weight — loaded ONCE per CTA and reused across all rows
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

  const float inv_scale = 1.0f / (*scale);

  // Process ROWS_PER_BLOCK rows in a serial loop within the same CTA
#pragma unroll
  for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
    const int row = blockIdx.x * ROWS_PER_BLOCK + r;
    if (row >= num_rows) break;

    const T* input_row = input + static_cast<size_t>(row) * hidden_size;
    c10::Float8_e4m3fn* output_row =
        out + static_cast<size_t>(row) * hidden_size;

    // Pass 1: load input → register, compute variance
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

    const float total_sq = static_quant_block_reduce_sum<BLOCK_X>(variance);
    const float inv_rms =
        rsqrtf(total_sq / static_cast<float>(hidden_size) + epsilon);

    // Pass 2: normed = input * inv_rms * weight, quantize via hw FP8 intrinsic.
#pragma unroll
    for (int chunk = 0; chunk < CHUNK; ++chunk) {
      const int vec_idx = chunk * BLOCK_X + tid;
      if (vec_idx < vec_hidden_size) {
        Vec16<T> w = *reinterpret_cast<Vec16<T>*>(shared_weight +
                                                  vec_idx * VEC_SIZE);
        float scaled[VEC_SIZE];
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          const float v = static_cast<float>(input_local[chunk].x[i]) *
                          inv_rms * static_cast<float>(w.x[i]);
          scaled[i] = v * inv_scale;
        }
        const uint64_t packed = pack8_floats_to_fp8e4m3(scaled);
        *reinterpret_cast<uint64_t*>(output_row + vec_idx * VEC_SIZE) = packed;
      }
    }
    // syncthreads between row iterations is needed because next row's
    // block_reduce uses the same smem reduction buffer.
    __syncthreads();
  }
}

template <typename T>
void dispatch_rms_norm_static_fp8_quant(c10::Float8_e4m3fn* out, const T* input,
                                        const T* weight, const float* scale,
                                        int rows, int hidden_size,
                                        float epsilon, musaStream_t stream) {
  const int vec_hidden_size = hidden_size / 8;
  // V5: ROWS_PER_BLOCK=4 at large M (V3 sweep found 4 > 8 > 2 > 1).
  // V4 result @ M=4096: RPB=8 → 56.03%% (regressed from V3 RPB=4 → 59.67%%).
  // RPB=4 hits the sweet spot of launch-overhead amortization vs per-row
  // smem sync overhead. Block_x must be 256 for the RPB>1 path.
  constexpr int RPB_LARGE = 4;
  constexpr int RPB_MID = 2;
  constexpr int RPB_SMALL = 1;

  int rpb, block_x;
  if (rows >= 256) {
    rpb = RPB_LARGE;
    block_x = 256;
  } else if (rows >= 32) {
    rpb = RPB_MID;
    block_x = 256;
  } else if (rows >= 8 && vec_hidden_size >= 1280) {
    rpb = RPB_SMALL;
    block_x = 512;
  } else {
    rpb = RPB_SMALL;
    block_x = 1024;
  }
  const int grid = (rows + rpb - 1) / rpb;

#define LAUNCH(BLOCK, CHUNK, RPB)                                             \
  rms_norm_static_fp8_quant_kernel<T, BLOCK, CHUNK, RPB>                      \
      <<<grid, BLOCK, 0, stream>>>(out, input, weight, scale, rows,           \
                                   hidden_size, vec_hidden_size, epsilon)

  if (rpb == RPB_LARGE) {
    // V6 sweep (BLOCK_X=128 vs 256): 59.28%% vs 60.17%% within noise — block
    // geometry is not the lever. The 20 pp gap to 80%% requires structural
    // change (128-bit FP8 store via uint4 + 16 elem/thread restructure, or
    // cross-kernel fusion). Keep V5 BLOCK_X=256 CHUNK=N/(256) RPB=4.
    if (vec_hidden_size <= 256) LAUNCH(256, 1, RPB_LARGE);
    else if (vec_hidden_size <= 512) LAUNCH(256, 2, RPB_LARGE);
    else if (vec_hidden_size <= 768) LAUNCH(256, 3, RPB_LARGE);
    else if (vec_hidden_size <= 1024) LAUNCH(256, 4, RPB_LARGE);
    else if (vec_hidden_size <= 1536) LAUNCH(256, 6, RPB_LARGE);
    else if (vec_hidden_size <= 2048) LAUNCH(256, 8, RPB_LARGE);
    else TORCH_CHECK(false, "BLOCK=256 hidden_size too large");
  } else if (rpb == RPB_MID) {
    if (vec_hidden_size <= 256) LAUNCH(256, 1, RPB_MID);
    else if (vec_hidden_size <= 512) LAUNCH(256, 2, RPB_MID);
    else if (vec_hidden_size <= 768) LAUNCH(256, 3, RPB_MID);
    else if (vec_hidden_size <= 1024) LAUNCH(256, 4, RPB_MID);
    else if (vec_hidden_size <= 1536) LAUNCH(256, 6, RPB_MID);
    else if (vec_hidden_size <= 2048) LAUNCH(256, 8, RPB_MID);
    else TORCH_CHECK(false, "BLOCK=256 hidden_size too large");
  } else {
    // Small-M fallback (single row per CTA) — uses larger BLOCK_X for occupancy
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
  }
#undef LAUNCH
}

}  // namespace vllm_musa

void musa_rms_norm_static_fp8_quant(torch::Tensor& out,
                                    torch::Tensor const& input,
                                    torch::Tensor const& weight,
                                    torch::Tensor const& scale,
                                    double epsilon) {
  TORCH_CHECK(input.device().is_privateuseone(), "input must be a MUSA tensor");
  TORCH_CHECK(out.dtype() == c10::ScalarType::Float8_e4m3fn,
              "out must be FP8 e4m3");
  TORCH_CHECK(scale.dtype() == c10::ScalarType::Float, "scale must be FP32");
  TORCH_CHECK(input.is_contiguous() && out.is_contiguous(),
              "input/out must be contiguous");
  TORCH_CHECK(input.dim() == 2 || input.dim() == 3, "input must be 2-D or 3-D");
  TORCH_CHECK(weight.dim() == 1 && input.size(-1) == weight.size(0),
              "weight shape mismatch");
  TORCH_CHECK(input.size(-1) % 8 == 0,
              "hidden_size must be a multiple of 8");
  TORCH_CHECK(input.size(-1) <= 16384,
              "hidden_size must be <= 16384");

  const c10::musa::OptionalMUSAGuard guard(device_of(input));
  musaStream_t stream = c10::musa::getCurrentMUSAStream().stream();
  const int hidden_size = input.size(-1);
  const int rows = input.numel() / hidden_size;

  if (input.scalar_type() == c10::ScalarType::BFloat16) {
    vllm_musa::dispatch_rms_norm_static_fp8_quant<__mt_bfloat16>(
        reinterpret_cast<c10::Float8_e4m3fn*>(out.data_ptr()),
        reinterpret_cast<const __mt_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const __mt_bfloat16*>(weight.data_ptr()),
        scale.data_ptr<float>(), rows, hidden_size,
        static_cast<float>(epsilon), stream);
  } else if (input.scalar_type() == c10::ScalarType::Half) {
    vllm_musa::dispatch_rms_norm_static_fp8_quant<__half>(
        reinterpret_cast<c10::Float8_e4m3fn*>(out.data_ptr()),
        reinterpret_cast<const __half*>(input.data_ptr()),
        reinterpret_cast<const __half*>(weight.data_ptr()),
        scale.data_ptr<float>(), rows, hidden_size,
        static_cast<float>(epsilon), stream);
  } else {
    TORCH_CHECK(false, "Unsupported dtype for musa_rms_norm_static_fp8_quant");
  }
}
