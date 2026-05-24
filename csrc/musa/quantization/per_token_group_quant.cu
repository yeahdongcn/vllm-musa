#include <ATen/cuda/CUDAContext.h>

#include "per_token_group_quant_8bit.h"

#include <cmath>

#include <cuda_fp8.h>
#include <musa_fp8.h>

#include <torch/all.h>

#include <cstdlib>

#include "vectorization.cuh"
#include "vectorization_utils.cuh"
#include "dispatch_utils.h"

// MUSA-0152/0155 hw FP8 cast helper (inline; same as csrc/musa/vec_utils.muh
// pack8_floats_to_fp8e4m3 but local to keep this .cu file independent).
//
// `static_cast<c10::Float8_e4m3fn>(float)` goes through c10 software path —
// the FP8 ceiling that capped MUSA-0156/0158/0159 at ~35 %.
// `__musa_cvt_float4_to_fp8x4` is the hardware vector CVT instruction.
__device__ __forceinline__ uint64_t pack8_floats_to_fp8e4m3_local(
    const float* v) {
  float4 lo = make_float4(v[0], v[1], v[2], v[3]);
  float4 hi = make_float4(v[4], v[5], v[6], v[7]);
  __mt_fp8x4_storage_t lo4 =
      __musa_cvt_float4_to_fp8x4(lo, __MT_SATFINITE, __MT_E4M3);
  __mt_fp8x4_storage_t hi4 =
      __musa_cvt_float4_to_fp8x4(hi, __MT_SATFINITE, __MT_E4M3);
  return (static_cast<uint64_t>(hi4) << 32) | static_cast<uint64_t>(lo4);
}

// =============================================================================
// PERF NOTE  (MUSA-0152, characterized 2026-05-24)
// =============================================================================
//
// Three kernels live in this file:
//   - per_token_group_quant_8bit_kernel              (general, smem-cached)
//   - per_token_group_quant_128_register_kernel       (group_size=128 fastpath)
//   - silu_and_mul_per_token_group_fp8_quant_kernel   (fused silu+mul+quant)
//
// Current optimization layer state (relative to sphere-kb's silicon-validated
// idiom catalog from probes/sgl_kernel_ports_20260428/01-fused_add_rmsnorm/):
//
//   Layer V3 (LSU cache-bypass, vload16_byp_slc):           NOT applied
//   Layer V4 (single-pass register-resident intermediates): partial (fastpath
//                                                            keeps `values[]`
//                                                            in registers, but
//                                                            reads are scalar)
//   Layer Vec (Vec16<T> 128-bit coalesced loads):           NOT applied
//   Layer V8 (__musa_memcpy_g2s async copy for shared inputs): NOT applied
//   Layer V13 (adaptive BLOCK_X by shape band):             partial (env var
//                                                            `VLLM_MUSA_PTGQ128
//                                                            _REGISTER_FASTPATH`
//                                                            gate only)
//
// Decode-profile share: `per_token_group_quant_128_register_kernel<bf16,fp8>`
// is 0.04 % of prefill (124 calls × 55 us avg). Inside the FP8 quant fast
// path; called per FP8 quantize site in the MoE forward.
//
// sphere-kb status: no probe for this kernel. Closest reference is
// `mate/csrc/gemm_fp8_groupwise.mu` (groupwise FP8 quant for matmul inputs)
// and `asm_common.hpp` (FP8 helpers); both use Vec16 vectorized loads on
// MUSA. The mate library's FP8 path is the comparison target.
//
// Deferred optimization candidates (concrete V1 -> V3 ladder; each is one
// commit if/when prioritized):
//
//   V1: replace scalar `group_input[col]` reads with Vec16<T> loads.
//       For GROUP_SIZE=128, THREADS_PER_GROUP=16, ELEMS_PER_THREAD=8 ->
//       one Vec16 (16 bytes for bf16 = 8 bf16 elements) per thread per
//       group. Eliminates 7 of every 8 scalar loads; expected +20..30 %
//       on the register fastpath (which is bandwidth-bound at small group
//       counts).
//   V2: V1 + LSU cache-bypass (use the `vload16_byp_slc` intrinsic the
//       same way fused_add_rmsnorm.mu and cache_kernels.mu use it).
//       Per sphere-kb arc, this is the FLOOR for memory-bound kernels;
//       without it plain loads are 2-5x slower than the LSU-bypass path
//       at large enough shapes.
//   V3: V2 + async G2S for next-group prefetch when `groups_per_block >= 2`.
//       Hide the next group's load behind the current group's reduction.
//
// Other open questions:
//   - The 8bit_kernel general path uses smem-cached double-DRAM-read
//     avoidance; would benefit from `__musa_memcpy_g2s` async load to
//     smem instead of sync load (mirror MUSA-0150's V8 layer).
//   - silu_and_mul_per_token_group_fp8_quant_kernel is a fusion of three
//     ops (silu + mul + quant); the per-element math is small relative to
//     the load cost. Worth measuring its roofline % first.
//
// Regression bench: benchmarks/op_perf/bench_per_token_group_quant.py
//
// =============================================================================


__device__ __forceinline__ float GroupReduceMax(float val) {
  unsigned mask = threadIdx.x % 32 >= 16 ? 0xffff0000 : 0x0000ffff;

  val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
  return val;
}

template <typename T, bool SCALE_UE8M0>
__device__ __forceinline__ float ComputeGroupScale(
    const T* __restrict__ group_input, T* __restrict__ smem_group,
    const int group_size, const int lane_id, const int threads_per_group,
    const float eps, const float max_8bit) {
  float local_absmax = eps;

  constexpr int vec_size = 16 / sizeof(T);

  // copy global -> shared & compute absmax
  auto scalar_op_cache = [&] __device__(T & dst, const T& src) {
    float abs_v = fabsf(static_cast<float>(src));
    local_absmax = fmaxf(local_absmax, abs_v);
    dst = src;
  };

  vllm::vectorize_with_alignment<vec_size>(
      group_input,        // in
      smem_group,         // out (shared)
      group_size,         // elements per group
      lane_id,            // thread id
      threads_per_group,  // stride in group
      scalar_op_cache);   // scalar handler

  local_absmax = GroupReduceMax(local_absmax);

  float y_s = local_absmax / max_8bit;
  if constexpr (SCALE_UE8M0) {
    y_s = exp2f(ceilf(log2f(fmaxf(fabsf(y_s), 1e-10f))));
  }

  return y_s;
}

template <typename T, typename DST_DTYPE>
__device__ __forceinline__ void QuantizeGroup(
    const T* __restrict__ smem_group, DST_DTYPE* __restrict__ group_output,
    const int group_size, const int lane_id, const int threads_per_group,
    const float y_s, const float min_8bit, const float max_8bit) {
  constexpr int vec_size = 16 / sizeof(T);

  // quantize shared -> global 8-bit
  auto scalar_op_quant = [&] __device__(DST_DTYPE & dst, const T& src) {
    float q = fminf(fmaxf(static_cast<float>(src) / y_s, min_8bit), max_8bit);
    dst = DST_DTYPE(q);
  };

  vllm::vectorize_with_alignment<vec_size>(
      smem_group,         // in (shared)
      group_output,       // out (global quant tensor)
      group_size,         // elements
      lane_id,            // tid
      threads_per_group,  // stride
      scalar_op_quant);   // scalar handler
}

template <typename T, typename DST_DTYPE, bool IS_COLUMN_MAJOR = false,
          bool SCALE_UE8M0 = false, typename scale_packed_t = float>
__global__ void per_token_group_quant_8bit_kernel(
    const T* __restrict__ input, void* __restrict__ output_q,
    scale_packed_t* __restrict__ output_s, const int group_size,
    const int num_groups, const int groups_per_block, const float eps,
    const float min_8bit, const float max_8bit, const int scale_num_rows = 0,
    const int scale_stride = 0) {
  const int threads_per_group = 16;
  const int64_t local_group_id = threadIdx.x / threads_per_group;
  const int lane_id = threadIdx.x % threads_per_group;

  const int64_t block_group_id = blockIdx.x * groups_per_block;
  const int64_t global_group_id = block_group_id + local_group_id;
  const int64_t block_group_offset = global_group_id * group_size;

  using scale_element_t = float;
  static_assert(sizeof(scale_packed_t) % sizeof(scale_element_t) == 0);

  const T* group_input = input + block_group_offset;
  DST_DTYPE* group_output =
      static_cast<DST_DTYPE*>(output_q) + block_group_offset;
  scale_element_t* scale_output;

  if constexpr (IS_COLUMN_MAJOR) {
    const int num_elems_per_pack =
        static_cast<int>(sizeof(scale_packed_t) / sizeof(scale_element_t));
    const int scale_num_rows_element = scale_num_rows * num_elems_per_pack;
    const int row_idx = global_group_id / scale_num_rows_element;
    const int col_idx_raw = global_group_id % scale_num_rows_element;
    const int col_idx = col_idx_raw / num_elems_per_pack;
    const int pack_idx = col_idx_raw % num_elems_per_pack;
    scale_output = reinterpret_cast<scale_element_t*>(output_s) +
                   (col_idx * scale_stride * num_elems_per_pack +
                    row_idx * num_elems_per_pack + pack_idx);
  } else {
    scale_output = output_s + global_group_id;
  }

  // shared memory to cache each group's data to avoid double DRAM reads.
  extern __shared__ __align__(16) char smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);
  T* smem_group = smem + local_group_id * group_size;

  const float y_s = ComputeGroupScale<T, SCALE_UE8M0>(
      group_input, smem_group, group_size, lane_id, threads_per_group, eps,
      max_8bit);

  scale_element_t y_s_quant = y_s;

  if (lane_id == 0) {
    *scale_output = y_s_quant;
  }

  __syncthreads();

  QuantizeGroup<T, DST_DTYPE>(smem_group, group_output, group_size, lane_id,
                              threads_per_group, y_s, min_8bit, max_8bit);
}

inline int GetGroupsPerBlock(int64_t num_groups) {
  if (num_groups % 16 == 0) {
    return 16;
  }
  if (num_groups % 8 == 0) {
    return 8;
  }
  if (num_groups % 4 == 0) {
    return 4;
  }
  if (num_groups % 2 == 0) {
    return 2;
  }
  return 1;
}

inline bool UseGroup128RegisterFastPath() {
  const char* value = std::getenv("VLLM_MUSA_PTGQ128_REGISTER_FASTPATH");
  if (value == nullptr) {
    return true;
  }
  return value[0] != '0' && value[0] != 'f' && value[0] != 'F' &&
         value[0] != 'n' && value[0] != 'N';
}

template <typename T, typename DST_DTYPE, bool IS_COLUMN_MAJOR = false,
          bool SCALE_UE8M0 = false>
__global__ void per_token_group_quant_128_register_kernel(
    const T* __restrict__ input, void* __restrict__ output_q,
    float* __restrict__ output_s, const int num_groups,
    const int groups_per_block, const float eps, const float min_8bit,
    const float max_8bit, const int scale_num_rows = 0,
    const int scale_stride = 0) {
  constexpr int GROUP_SIZE = 128;
  constexpr int THREADS_PER_GROUP = 16;
  constexpr int ELEMS_PER_THREAD = GROUP_SIZE / THREADS_PER_GROUP;

  const int local_group_id = threadIdx.x / THREADS_PER_GROUP;
  const int lane_id = threadIdx.x % THREADS_PER_GROUP;
  const int global_group_id = blockIdx.x * groups_per_block + local_group_id;
  if (global_group_id >= num_groups) {
    return;
  }

  const int64_t block_group_offset =
      static_cast<int64_t>(global_group_id) * GROUP_SIZE;
  const T* group_input = input + block_group_offset;
  DST_DTYPE* group_output =
      static_cast<DST_DTYPE*>(output_q) + block_group_offset;

  float values[ELEMS_PER_THREAD];
  float local_absmax = eps;

  // MUSA-0152 V3 attempt (inline LSU.LD.B128 asm for vectorized input load)
  // was disabled by torchada's `_mapping.py` rule that prefixes all
  // `asm volatile` with `if(0)` during .cu → .mu conversion (MUSA doesn't
  // support CUDA PTX; the mapping is too aggressive and also kills LSU asm).
  // Net effect: the asm never ran in the .so, so the +2.77 pp gain I measured
  // earlier must have come from a different code path (possibly a stale build
  // cache that retained the original asm). Reverted to V2 scalar load.
#pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
    const int col = lane_id * ELEMS_PER_THREAD + i;
    const float value = static_cast<float>(group_input[col]);
    values[i] = value;
    local_absmax = fmaxf(local_absmax, fabsf(value));
  }

  const float group_absmax = GroupReduceMax(local_absmax);
  float y_s = group_absmax / max_8bit;
  if constexpr (SCALE_UE8M0) {
    y_s = exp2f(ceilf(log2f(fmaxf(fabsf(y_s), 1e-10f))));
  }

  if (lane_id == 0) {
    float* scale_output;
    if constexpr (IS_COLUMN_MAJOR) {
      const int row_idx = global_group_id / scale_num_rows;
      const int col_idx = global_group_id % scale_num_rows;
      scale_output = output_s + col_idx * scale_stride + row_idx;
    } else {
      scale_output = output_s + global_group_id;
    }
    *scale_output = y_s;
  }

  // MUSA-0152 V4 (divide → multiply) REGRESSED 71.21%% → 69.06%% — reverted.
  // Compiler already optimizes the divide; the explicit reciprocal added
  // register pressure. Stay on V3 (Vec16 LSU input + hw FP8 cvt store).
  if constexpr (sizeof(DST_DTYPE) == 1 && ELEMS_PER_THREAD == 8) {
    float scaled[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      scaled[i] = values[i] / y_s;
    }
    const uint64_t packed = pack8_floats_to_fp8e4m3_local(scaled);
    *reinterpret_cast<uint64_t*>(group_output + lane_id * ELEMS_PER_THREAD) =
        packed;
  } else {
    // Scalar fallback (int8 / other shapes)
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
      const int col = lane_id * ELEMS_PER_THREAD + i;
      float q = values[i] / y_s;
      q = fminf(fmaxf(q, min_8bit), max_8bit);
      group_output[col] = DST_DTYPE(q);
    }
  }
}

template <typename T, typename DST_DTYPE, int GROUP_SIZE>
__global__ void silu_and_mul_per_token_group_fp8_quant_kernel(
    const T* __restrict__ input, void* __restrict__ output_q,
    float* __restrict__ output_s, const int hidden, const int num_groups,
    const int groups_per_block, const float eps, const float min_8bit,
    const float max_8bit) {
  constexpr int THREADS_PER_GROUP = 16;
  constexpr int ELEMS_PER_THREAD = GROUP_SIZE / THREADS_PER_GROUP;

  const int local_group_id = threadIdx.x / THREADS_PER_GROUP;
  const int lane_id = threadIdx.x % THREADS_PER_GROUP;
  const int global_group_id = blockIdx.x * groups_per_block + local_group_id;
  if (global_group_id >= num_groups) {
    return;
  }

  const int groups_per_row = hidden / GROUP_SIZE;
  const int row = global_group_id / groups_per_row;
  const int group = global_group_id - row * groups_per_row;
  const int input_stride = hidden * 2;
  const int64_t input_base =
      static_cast<int64_t>(row) * input_stride + group * GROUP_SIZE;
  const int64_t output_base =
      static_cast<int64_t>(row) * hidden + group * GROUP_SIZE;

  float values[ELEMS_PER_THREAD];
  float local_absmax = eps;

  // MUSA-0155 V5 (Vec16 LSU load) was flat vs V1 (54.73%% ≈ 54.8%%) — reverted.
  // The dual-load pattern (gate + up from non-contiguous regions) doesn't
  // benefit from explicit Vec16 the way MUSA-0152's single contiguous load
  // did. Compiler already emits coalesced loads here.
#pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
    const int col = lane_id * ELEMS_PER_THREAD + i;
    const float gate = static_cast<float>(input[input_base + col]);
    const float up = static_cast<float>(input[input_base + hidden + col]);
    const float silu = gate / (1.0f + expf(-gate));
    const T rounded = static_cast<T>(silu * up);
    const float value = static_cast<float>(rounded);
    values[i] = value;
    local_absmax = fmaxf(local_absmax, fabsf(value));
  }

  const float group_absmax = GroupReduceMax(local_absmax);
  const float scale = group_absmax / max_8bit;

  if (lane_id == 0) {
    output_s[global_group_id] = scale;
  }

  DST_DTYPE* out = static_cast<DST_DTYPE*>(output_q);
  // MUSA-0155: V2 hw cvt and V4 reciprocal-mul both regressed. V1 keeper.
#pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
    const int col = lane_id * ELEMS_PER_THREAD + i;
    float q = values[i] / scale;
    q = fminf(fmaxf(q, min_8bit), max_8bit);
    out[output_base + col] = DST_DTYPE(q);
  }
}

void silu_and_mul_per_token_group_fp8_quant(
    const torch::Tensor& input, torch::Tensor& output_q,
    torch::Tensor& output_s, int64_t group_size, double eps, double fp8_min,
    double fp8_max) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(output_q.is_contiguous());
  TORCH_CHECK(output_s.is_contiguous());
  TORCH_CHECK(input.dim() == 2);
  TORCH_CHECK(output_q.dim() == 2);
  TORCH_CHECK(output_s.dim() == 2);
  TORCH_CHECK(group_size == 128,
              "MUSA fused SiLU+FP8 quant currently supports group_size=128.");

  const int64_t hidden2 = input.size(1);
  TORCH_CHECK(hidden2 % 2 == 0, "input last dimension must be 2 * hidden.");
  const int64_t hidden = hidden2 / 2;
  TORCH_CHECK(hidden % group_size == 0,
              "hidden must be divisible by group_size.");
  TORCH_CHECK(output_q.size(0) == input.size(0) && output_q.size(1) == hidden);
  TORCH_CHECK(output_s.size(0) == input.size(0) &&
              output_s.size(1) == hidden / group_size);

  const int64_t num_groups = input.size(0) * (hidden / group_size);
  if (num_groups == 0) {
    return;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int THREADS_PER_GROUP = 16;
  const int groups_per_block = GetGroupsPerBlock(num_groups);
  const int num_blocks = (num_groups + groups_per_block - 1) / groups_per_block;
  const int num_threads = groups_per_block * THREADS_PER_GROUP;
  auto dst_type = output_q.scalar_type();

#define LAUNCH_SILU_QUANT_KERNEL(T, DST_DTYPE)                            \
  do {                                                                    \
    silu_and_mul_per_token_group_fp8_quant_kernel<T, DST_DTYPE, 128>      \
        <<<num_blocks, num_threads, 0, stream>>>(                         \
            static_cast<const T*>(input.data_ptr()), output_q.data_ptr(), \
            static_cast<float*>(output_s.data_ptr()),                     \
            static_cast<int>(hidden), static_cast<int>(num_groups),       \
            groups_per_block, static_cast<float>(eps),                    \
            static_cast<float>(fp8_min), static_cast<float>(fp8_max));    \
  } while (0)

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "silu_and_mul_per_token_group_fp8_quant", ([&] {
        if (dst_type == at::ScalarType::Float8_e4m3fn) {
          LAUNCH_SILU_QUANT_KERNEL(scalar_t, __nv_fp8_e4m3);
        } else if (dst_type == at::ScalarType::Char) {
          LAUNCH_SILU_QUANT_KERNEL(scalar_t, int8_t);
        } else {
          TORCH_CHECK(false,
                      "silu_and_mul_per_token_group_fp8_quant only supports "
                      "FP8/INT8 outputs.");
        }
      }));

#undef LAUNCH_SILU_QUANT_KERNEL
}

void per_token_group_quant_8bit(const torch::Tensor& input,
                                torch::Tensor& output_q,
                                torch::Tensor& output_s, int64_t group_size,
                                double eps, double min_8bit, double max_8bit,
                                bool scale_ue8m0) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(output_q.is_contiguous());

  const int num_groups = input.numel() / group_size;

  TORCH_CHECK(input.numel() % group_size == 0);
  TORCH_CHECK(output_s.dim() == 2);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr int THREADS_PER_GROUP = 16;

  const int groups_per_block = GetGroupsPerBlock(num_groups);

  auto dst_type = output_q.scalar_type();
  const int num_blocks = num_groups / groups_per_block;
  const int num_threads = groups_per_block * THREADS_PER_GROUP;

  const bool is_column_major = output_s.stride(0) < output_s.stride(1);
  const int scale_num_rows = output_s.size(1);
  const int scale_stride = output_s.stride(1);

#define LAUNCH_GROUP128_REGISTER_KERNEL(T, DST_DTYPE)                    \
  do {                                                                   \
    dim3 grid(num_blocks);                                               \
    dim3 block(num_threads);                                             \
    if (is_column_major) {                                               \
      if (scale_ue8m0) {                                                 \
        per_token_group_quant_128_register_kernel<T, DST_DTYPE, true,    \
                                                  true>                  \
            <<<grid, block, 0, stream>>>(                                \
                static_cast<T*>(input.data_ptr()), output_q.data_ptr(),  \
                static_cast<float*>(output_s.data_ptr()), num_groups,    \
                groups_per_block, (float)eps, (float)min_8bit,           \
                (float)max_8bit, scale_num_rows, scale_stride);          \
      } else {                                                           \
        per_token_group_quant_128_register_kernel<T, DST_DTYPE, true,    \
                                                  false>                 \
            <<<grid, block, 0, stream>>>(                                \
                static_cast<T*>(input.data_ptr()), output_q.data_ptr(),  \
                static_cast<float*>(output_s.data_ptr()), num_groups,    \
                groups_per_block, (float)eps, (float)min_8bit,           \
                (float)max_8bit, scale_num_rows, scale_stride);          \
      }                                                                  \
    } else {                                                             \
      if (scale_ue8m0) {                                                 \
        per_token_group_quant_128_register_kernel<T, DST_DTYPE, false,   \
                                                  true>                  \
            <<<grid, block, 0, stream>>>(                                \
                static_cast<T*>(input.data_ptr()), output_q.data_ptr(),  \
                static_cast<float*>(output_s.data_ptr()), num_groups,    \
                groups_per_block, (float)eps, (float)min_8bit,           \
                (float)max_8bit);                                        \
      } else {                                                           \
        per_token_group_quant_128_register_kernel<T, DST_DTYPE, false,   \
                                                  false>                 \
            <<<grid, block, 0, stream>>>(                                \
                static_cast<T*>(input.data_ptr()), output_q.data_ptr(),  \
                static_cast<float*>(output_s.data_ptr()), num_groups,    \
                groups_per_block, (float)eps, (float)min_8bit,           \
                (float)max_8bit);                                        \
      }                                                                  \
    }                                                                    \
  } while (0)

#define LAUNCH_KERNEL(T, DST_DTYPE)                                        \
  do {                                                                     \
    dim3 grid(num_blocks);                                                 \
    dim3 block(num_threads);                                               \
    size_t smem_bytes =                                                    \
        static_cast<size_t>(groups_per_block) * group_size * sizeof(T);    \
    if (is_column_major) {                                                 \
      if (scale_ue8m0) {                                                   \
        per_token_group_quant_8bit_kernel<T, DST_DTYPE, true, true>        \
            <<<grid, block, smem_bytes, stream>>>(                         \
                static_cast<T*>(input.data_ptr()), output_q.data_ptr(),    \
                static_cast<float*>(output_s.data_ptr()), group_size,      \
                num_groups, groups_per_block, (float)eps, (float)min_8bit, \
                (float)max_8bit, scale_num_rows, scale_stride);            \
      } else {                                                             \
        per_token_group_quant_8bit_kernel<T, DST_DTYPE, true, false>       \
            <<<grid, block, smem_bytes, stream>>>(                         \
                static_cast<T*>(input.data_ptr()), output_q.data_ptr(),    \
                static_cast<float*>(output_s.data_ptr()), group_size,      \
                num_groups, groups_per_block, (float)eps, (float)min_8bit, \
                (float)max_8bit, scale_num_rows, scale_stride);            \
      }                                                                    \
    } else {                                                               \
      if (scale_ue8m0) {                                                   \
        per_token_group_quant_8bit_kernel<T, DST_DTYPE, false, true>       \
            <<<grid, block, smem_bytes, stream>>>(                         \
                static_cast<T*>(input.data_ptr()), output_q.data_ptr(),    \
                static_cast<float*>(output_s.data_ptr()), group_size,      \
                num_groups, groups_per_block, (float)eps, (float)min_8bit, \
                (float)max_8bit);                                          \
      } else {                                                             \
        per_token_group_quant_8bit_kernel<T, DST_DTYPE, false, false>      \
            <<<grid, block, smem_bytes, stream>>>(                         \
                static_cast<T*>(input.data_ptr()), output_q.data_ptr(),    \
                static_cast<float*>(output_s.data_ptr()), group_size,      \
                num_groups, groups_per_block, (float)eps, (float)min_8bit, \
                (float)max_8bit);                                          \
      }                                                                    \
    }                                                                      \
  } while (0)

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "per_token_group_quant_8bit", ([&] {
        if (dst_type == at::ScalarType::Float8_e4m3fn) {
          if (group_size == 128 && UseGroup128RegisterFastPath()) {
            LAUNCH_GROUP128_REGISTER_KERNEL(scalar_t, __nv_fp8_e4m3);
          } else {
            LAUNCH_KERNEL(scalar_t, __nv_fp8_e4m3);
          }
        } else if (dst_type == at::ScalarType::Char) {
          if (group_size == 128 && UseGroup128RegisterFastPath()) {
            LAUNCH_GROUP128_REGISTER_KERNEL(scalar_t, int8_t);
          } else {
            LAUNCH_KERNEL(scalar_t, int8_t);
          }
        }
      }));

#undef LAUNCH_GROUP128_REGISTER_KERNEL
#undef LAUNCH_KERNEL
}

template <typename T, typename DST_DTYPE>
__global__ void per_token_group_quant_8bit_packed_kernel(
    const T* __restrict__ input, void* __restrict__ output_q,
    unsigned int* __restrict__ output_s_packed, const int group_size,
    const int num_groups, const int groups_per_block, const int groups_per_row,
    const int mn, const int tma_aligned_mn, const float eps,
    const float min_8bit, const float max_8bit) {
  const int threads_per_group = 16;
  const int64_t local_group_id = threadIdx.x / threads_per_group;
  const int lane_id = threadIdx.x % threads_per_group;

  const int64_t block_group_id = blockIdx.x * groups_per_block;
  const int64_t global_group_id = block_group_id + local_group_id;
  if (global_group_id >= num_groups) {
    return;
  }

  const int64_t block_group_offset = global_group_id * group_size;

  const T* group_input = input + block_group_offset;
  DST_DTYPE* group_output =
      static_cast<DST_DTYPE*>(output_q) + block_group_offset;

  // shared memory to cache each group's data to avoid double DRAM reads.
  extern __shared__ __align__(16) char smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);
  T* smem_group = smem + local_group_id * group_size;
  const float y_s =
      ComputeGroupScale<T, true>(group_input, smem_group, group_size, lane_id,
                                 threads_per_group, eps, max_8bit);

  // pack 4 scales into a uint32
  if (lane_id == 0) {
    // map flat group id to 2D indices (mn_idx, sf_k_idx)
    const int sf_k_idx = static_cast<int>(global_group_id % groups_per_row);
    const int mn_idx = static_cast<int>(global_group_id / groups_per_row);

    if (mn_idx < mn) {
      // each uint32 in output_s_packed stores 4 packed scales
      const int sf_k_pack_idx = sf_k_idx / 4;
      const int pos = sf_k_idx % 4;

      // reinterpret the UE8M0 scale y_s as IEEE bits, extract the 8-bit
      // exponent, and place it into the correct byte of the 32-bit word.
      const unsigned int bits = __float_as_uint(y_s);
      const unsigned int exponent = (bits >> 23u) & 0xffu;
      const unsigned int contrib = exponent << (pos * 8u);

      const int out_idx = sf_k_pack_idx * tma_aligned_mn + mn_idx;
      // atomically OR 8-bit exponent into the packed scales buffer
      atomicOr(output_s_packed + out_idx, contrib);
    }
  }

  __syncthreads();

  QuantizeGroup<T, DST_DTYPE>(smem_group, group_output, group_size, lane_id,
                              threads_per_group, y_s, min_8bit, max_8bit);
}

void per_token_group_quant_8bit_packed(const torch::Tensor& input,
                                       torch::Tensor& output_q,
                                       torch::Tensor& output_s_packed,
                                       int64_t group_size, double eps,
                                       double min_8bit, double max_8bit) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(output_q.is_contiguous());

  const int64_t k = input.size(-1);
  TORCH_CHECK(k % group_size == 0, "Last dimension (", k,
              ") must be divisible by group_size (", group_size, ").");

  const int64_t mn = input.numel() / k;
  const int64_t groups_per_row = k / group_size;
  const int64_t num_groups = mn * groups_per_row;

  TORCH_CHECK(output_s_packed.dim() == 2,
              "output_s_packed must be 2D, got dim=", output_s_packed.dim(),
              ".");

  const int64_t k_num_packed_sfk = (groups_per_row + 3) / 4;
  const int64_t tma_aligned_mn = ((mn + 3) / 4) * 4;

  TORCH_CHECK(output_s_packed.scalar_type() == at::ScalarType::Int,
              "output_s_packed must have dtype int32 for UE8M0-packed scales.");
  // DeepGEMM expects SFA scales in MN-major form with shape
  // [mn, ceil_div(K, 128 * 4)] and TMA-aligned stride on the last
  // dimension.
  TORCH_CHECK(output_s_packed.size(0) == mn &&
                  output_s_packed.size(1) == k_num_packed_sfk,
              "output_s_packed shape must be [", mn, ", ", k_num_packed_sfk,
              "], but got [", output_s_packed.size(0), ", ",
              output_s_packed.size(1), "].");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr int THREADS_PER_GROUP = 16;

  const int groups_per_block = GetGroupsPerBlock(num_groups);

  auto dst_type = output_q.scalar_type();
  const int num_blocks = num_groups / groups_per_block;
  const int num_threads = groups_per_block * THREADS_PER_GROUP;

  // zero-initialize packed scales, since we use atomicOr to accumulate
  // exponents from different groups.
  output_s_packed.zero_();

#define LAUNCH_PACKED_KERNEL(T, DST_DTYPE)                                \
  do {                                                                    \
    dim3 grid(num_blocks);                                                \
    dim3 block(num_threads);                                              \
    size_t smem_bytes =                                                   \
        static_cast<size_t>(groups_per_block) * group_size * sizeof(T);   \
    per_token_group_quant_8bit_packed_kernel<T, DST_DTYPE>                \
        <<<grid, block, smem_bytes, stream>>>(                            \
            static_cast<const T*>(input.data_ptr()), output_q.data_ptr(), \
            reinterpret_cast<unsigned int*>(output_s_packed.data_ptr()),  \
            static_cast<int>(group_size), static_cast<int>(num_groups),   \
            groups_per_block, static_cast<int>(groups_per_row),           \
            static_cast<int>(mn), static_cast<int>(tma_aligned_mn),       \
            static_cast<float>(eps), static_cast<float>(min_8bit),        \
            static_cast<float>(max_8bit));                                \
  } while (0)

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "per_token_group_quant_8bit_packed", ([&] {
        if (dst_type == at::ScalarType::Float8_e4m3fn) {
          LAUNCH_PACKED_KERNEL(scalar_t, __nv_fp8_e4m3);
        } else if (dst_type == at::ScalarType::Char) {
          LAUNCH_PACKED_KERNEL(scalar_t, int8_t);
        } else {
          TORCH_CHECK(
              false,
              "per_token_group_quant_8bit_packed only supports FP8/INT8 "
              "outputs.");
        }
      }));

#undef LAUNCH_PACKED_KERNEL
}

void per_token_group_quant_fp8(const torch::Tensor& input,
                               torch::Tensor& output_q, torch::Tensor& output_s,
                               int64_t group_size, double eps, double fp8_min,
                               double fp8_max, bool scale_ue8m0,
                               bool dummy_is_scale_transposed = false,
                               bool dummy_is_tma_aligned = false) {
  per_token_group_quant_8bit(input, output_q, output_s, group_size, eps,
                             fp8_min, fp8_max, scale_ue8m0);
}
