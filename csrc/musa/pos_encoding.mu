// PH1-native rotary_embedding_contiguous — full coverage (NEOX + GPT-J interleaved).
//
// Baseline: sgl-kernel csrc/musa/pos_encoding_contiguous.mu — scalar `MUSA_LDG` loads,
// no vectorization, cos/sin re-read by every thread.
//
// 4-layer pattern:
//   Layer 1: LSU.LD.B128 cache-bypass for Q/K streaming reads/writes
//   Layer 2: single-pass — read Q vec, compute rotation, write Q vec (no re-read)
//   Layer 3: async G2S cos_sin row to smem at block entry; all heads share it
//   Layer 4: adaptive BLOCK_X dispatch (T>=256 → 128; T<256 H>=64 → 512; else 256)
//
// Both NEOX (`is_neox=true`, half-rotate pairs (i, i+embed_dim)) and GPT-J interleaved
// (`is_neox=false`, pairs (2i, 2i+1)) supported with the same dispatch + smem layout.
// cos_sin_cache layout is identical for both modes: [max_pos][cos_dim_embed | sin_dim_embed].

#include <musa_runtime.h>
#include <torch/all.h>
#include <torch_musa/csrc/aten/musa/MUSAContext.h>
#include <torch_musa/csrc/core/MUSAGuard.h>
#include <cstdlib>

extern "C" {
extern __device__ void __musa_memcpy_g2s(
    __attribute__((address_space(3))) void* dst,
    __attribute__((address_space(1))) const void* src,
    int size, int prefetch);
extern __device__ void __musa_memcpy_g2s_wait();
}

namespace vllm_musa {

template <typename T>
struct Vec16 {
  static_assert(sizeof(T) == 2, "T must be 16-bit");
  static constexpr int VLEN = 8;
  T x[VLEN];
};

template <typename T>
__device__ __forceinline__ Vec16<T> vload16_byp_slc(const T* ptr) {
  Vec16<T> dst;
  asm volatile(
      "LSU.LD.B128 %0, %1, _, 16, 1, 1, inner_persist=0, "
      "outer_persist=2, chrnt=l2_l3, slc=byp, persist=0, "
      "stride_add_first=0"
      : "=R"(dst)
      : "R"(ptr));
  return dst;
}

template <typename T>
__device__ __forceinline__ void vstore16(T* p, const Vec16<T>& v) {
  *reinterpret_cast<Vec16<T>*>(p) = v;
}

// Apply NEOX rotation to one head: pairs are (i, i + embed_dim).
// num_pairs == num_heads * embed_dim total. Each thread processes VLEN=8 pairs of (x, y).
template <typename T, int BLOCK_X>
__device__ __forceinline__ void apply_neox_block(
    T* qk_token, int num_heads, int embed_dim,
    int64_t token_stride, int64_t head_stride, int64_t dim_stride,
    const T* smem_cos, const T* smem_sin) {
  constexpr int VLEN = 8;
  const int total_pairs = num_heads * embed_dim;
  const int tid = threadIdx.x;

  // Strided over pairs in vec chunks of VLEN=8.
  for (int base = tid * VLEN; base < total_pairs; base += BLOCK_X * VLEN) {
    int head_idx = base / embed_dim;
    int rot_off = base % embed_dim;
    // The chunk may straddle a head boundary. Bail to scalar if so (rare for embed_dim multiple of 8).
    if (rot_off + VLEN > embed_dim) {
      // Scalar fallback for boundary.
      #pragma unroll
      for (int k = 0; k < VLEN; ++k) {
        int p = base + k;
        if (p >= total_pairs) break;
        int h = p / embed_dim;
        int r = p % embed_dim;
        T* head_ptr = qk_token + h * head_stride;
        T cos = smem_cos[r];
        T sin = smem_sin[r];
        T x = head_ptr[r * dim_stride];
        T y = head_ptr[(embed_dim + r) * dim_stride];
        head_ptr[r * dim_stride] = (T)((float)x * (float)cos - (float)y * (float)sin);
        head_ptr[(embed_dim + r) * dim_stride] = (T)((float)y * (float)cos + (float)x * (float)sin);
      }
      continue;
    }
    T* head_ptr = qk_token + head_idx * head_stride;
    // Vec load x (offset rot_off) and y (offset embed_dim + rot_off).
    Vec16<T> vx = vload16_byp_slc(head_ptr + rot_off);
    Vec16<T> vy = vload16_byp_slc(head_ptr + embed_dim + rot_off);
    Vec16<T> vcos = *reinterpret_cast<const Vec16<T>*>(smem_cos + rot_off);
    Vec16<T> vsin = *reinterpret_cast<const Vec16<T>*>(smem_sin + rot_off);
    Vec16<T> ox, oy;
    #pragma unroll
    for (int k = 0; k < VLEN; ++k) {
      float c = (float)vcos.x[k], s = (float)vsin.x[k];
      float x = (float)vx.x[k], y = (float)vy.x[k];
      ox.x[k] = (T)(x * c - y * s);
      oy.x[k] = (T)(y * c + x * s);
    }
    vstore16(head_ptr + rot_off, ox);
    vstore16(head_ptr + embed_dim + rot_off, oy);
  }
}

// Apply GPT-J interleaved rotation: pairs are (2i, 2i+1) — adjacent in memory.
// Per thread: VLEN=8 pairs = 16 contiguous elements (two B128 loads).
// cos_sin_cache layout same as NEOX: cos[0..embed_dim-1] | sin[0..embed_dim-1].
template <typename T, int BLOCK_X>
__device__ __forceinline__ void apply_interleave_block(
    T* qk_token, int num_heads, int embed_dim,
    int64_t token_stride, int64_t head_stride, int64_t dim_stride,
    const T* smem_cos, const T* smem_sin) {
  constexpr int VLEN = 8;
  const int total_pairs = num_heads * embed_dim;
  const int tid = threadIdx.x;

  for (int base = tid * VLEN; base < total_pairs; base += BLOCK_X * VLEN) {
    int head_idx = base / embed_dim;
    int rot_off = base % embed_dim;

    if (rot_off + VLEN > embed_dim) {
      // Boundary scalar fallback.
      #pragma unroll
      for (int k = 0; k < VLEN; ++k) {
        int p = base + k;
        if (p >= total_pairs) break;
        int h = p / embed_dim, r = p % embed_dim;
        T* head_ptr = qk_token + h * head_stride;
        float c = (float)smem_cos[r], s = (float)smem_sin[r];
        float x = (float)head_ptr[(2 * r) * dim_stride];
        float y = (float)head_ptr[(2 * r + 1) * dim_stride];
        head_ptr[(2 * r) * dim_stride]     = (T)(x * c - y * s);
        head_ptr[(2 * r + 1) * dim_stride] = (T)(y * c + x * s);
      }
      continue;
    }

    T* head_ptr = qk_token + head_idx * head_stride;
    // Load 16 contiguous elements = 8 (x,y) pairs starting at offset 2*rot_off.
    Vec16<T> v0 = vload16_byp_slc(head_ptr + 2 * rot_off);       // pairs 0..3 (8 elems)
    Vec16<T> v1 = vload16_byp_slc(head_ptr + 2 * rot_off + 8);   // pairs 4..7 (8 elems)
    // Vec-load cos/sin from smem in one shot (matches NEOX path; was 8 scalar reads before).
    Vec16<T> vcos = *reinterpret_cast<const Vec16<T>*>(smem_cos + rot_off);
    Vec16<T> vsin = *reinterpret_cast<const Vec16<T>*>(smem_sin + rot_off);

    Vec16<T> o0, o1;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      float c = (float)vcos.x[k], s = (float)vsin.x[k];
      float x = (float)v0.x[2 * k], y = (float)v0.x[2 * k + 1];
      o0.x[2 * k]     = (T)(x * c - y * s);
      o0.x[2 * k + 1] = (T)(y * c + x * s);
    }
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      float c = (float)vcos.x[4 + k], s = (float)vsin.x[4 + k];
      float x = (float)v1.x[2 * k], y = (float)v1.x[2 * k + 1];
      o1.x[2 * k]     = (T)(x * c - y * s);
      o1.x[2 * k + 1] = (T)(y * c + x * s);
    }
    vstore16(head_ptr + 2 * rot_off, o0);
    vstore16(head_ptr + 2 * rot_off + 8, o1);
  }
}

// V1: BLOCK_X=256, __launch_bounds__(256, 1).
template <typename T, int BLOCK_X>
__global__ void __launch_bounds__(BLOCK_X, 1) sphere_rope_neox_kernel_v1(
    const int64_t* __restrict__ positions,
    T* __restrict__ query, T* __restrict__ key,
    const T* __restrict__ cos_sin_cache,
    int rot_dim,  // = 2 * embed_dim
    int64_t q_token_stride, int64_t q_head_stride, int64_t q_dim_stride,
    int64_t k_token_stride, int64_t k_head_stride, int64_t k_dim_stride,
    int num_heads, int num_kv_heads) {
  const int token_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int embed_dim = rot_dim / 2;

  // Async G2S the entire cos_sin_cache row to smem.
  // Footprint: rot_dim * sizeof(T) = up to 256 bytes for rot_dim=128 fp16.
  extern __shared__ char smem_raw[];
  T* smem_cs = reinterpret_cast<T*>(smem_raw);

  // Per-thread issue 16B atoms. rot_dim halves = rot_dim*2 bytes total.
  // Threads with tid * 8 < rot_dim issue.
  const int n_vec = rot_dim / 8;  // number of 16B vectors in cos_sin row
  if (tid < n_vec) {
    int64_t pos = positions[token_idx];
    const T* gptr_src = cos_sin_cache + pos * rot_dim + tid * 8;
    T* sptr_dst = smem_cs + tid * 8;
    __attribute__((address_space(3))) void* sptr =
        (__attribute__((address_space(3))) void*)sptr_dst;
    __attribute__((address_space(1))) const void* gptr =
        (__attribute__((address_space(1))) const void*)gptr_src;
    __musa_memcpy_g2s(sptr, gptr, 16, 128);
  }
  __musa_memcpy_g2s_wait();
  __syncthreads();

  const T* smem_cos = smem_cs;
  const T* smem_sin = smem_cs + embed_dim;

  // Q
  T* q_token = query + token_idx * q_token_stride;
  apply_neox_block<T, BLOCK_X>(
      q_token, num_heads, embed_dim,
      q_token_stride, q_head_stride, q_dim_stride,
      smem_cos, smem_sin);
  // K
  T* k_token = key + token_idx * k_token_stride;
  apply_neox_block<T, BLOCK_X>(
      k_token, num_kv_heads, embed_dim,
      k_token_stride, k_head_stride, k_dim_stride,
      smem_cos, smem_sin);
}

// V2: NO launch_bounds — let compiler optimize for higher concurrent CTAs/MP.
// Templated on INTERLEAVE: false = NEOX (Llama), true = GPT-J interleaved.
template <typename T, int BLOCK_X, bool INTERLEAVE>
__global__ void sphere_rope_kernel_v2(
    const int64_t* __restrict__ positions,
    T* __restrict__ query, T* __restrict__ key,
    const T* __restrict__ cos_sin_cache,
    int rot_dim,
    int64_t q_token_stride, int64_t q_head_stride, int64_t q_dim_stride,
    int64_t k_token_stride, int64_t k_head_stride, int64_t k_dim_stride,
    int num_heads, int num_kv_heads) {
  const int token_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int embed_dim = rot_dim / 2;
  extern __shared__ char smem_raw[];
  T* smem_cs = reinterpret_cast<T*>(smem_raw);
  const int n_vec = rot_dim / 8;
  if (tid < n_vec) {
    int64_t pos = positions[token_idx];
    const T* gptr_src = cos_sin_cache + pos * rot_dim + tid * 8;
    T* sptr_dst = smem_cs + tid * 8;
    __attribute__((address_space(3))) void* sptr =
        (__attribute__((address_space(3))) void*)sptr_dst;
    __attribute__((address_space(1))) const void* gptr =
        (__attribute__((address_space(1))) const void*)gptr_src;
    __musa_memcpy_g2s(sptr, gptr, 16, 128);
  }
  __musa_memcpy_g2s_wait();
  __syncthreads();
  const T* smem_cos = smem_cs;
  const T* smem_sin = smem_cs + embed_dim;
  T* q_token = query + token_idx * q_token_stride;
  T* k_token = key + token_idx * k_token_stride;
  if constexpr (INTERLEAVE) {
    apply_interleave_block<T, BLOCK_X>(q_token, num_heads, embed_dim,
        q_token_stride, q_head_stride, q_dim_stride, smem_cos, smem_sin);
    apply_interleave_block<T, BLOCK_X>(k_token, num_kv_heads, embed_dim,
        k_token_stride, k_head_stride, k_dim_stride, smem_cos, smem_sin);
  } else {
    apply_neox_block<T, BLOCK_X>(q_token, num_heads, embed_dim,
        q_token_stride, q_head_stride, q_dim_stride, smem_cos, smem_sin);
    apply_neox_block<T, BLOCK_X>(k_token, num_kv_heads, embed_dim,
        k_token_stride, k_head_stride, k_dim_stride, smem_cos, smem_sin);
  }
}

void musa_rotary_embedding_impl(
    torch::Tensor& positions,
    torch::Tensor& query,
    torch::Tensor& key,
    int64_t head_size,
    torch::Tensor& cos_sin_cache,
    bool is_neox) {
  constexpr int64_t variant = 5;  // V5 default (silicon-validated SOTA per sphere-kb probe)
  // vllm's rotary_embedding contract: query/key may be 2D [T, H*D] or 3D [T, H, D].
  // Normalize to 3D via .view() — strides remain contiguous so the kernel reads
  // [T, H, D] correctly.
  TORCH_CHECK(query.dim() == 2 || query.dim() == 3,
              "musa_rotary_embedding: query must be 2D [T, H*D] or 3D [T, H, D]");
  TORCH_CHECK(key.dim() == query.dim());
  int num_tokens = positions.size(0);
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous(),
              "musa_rotary_embedding: query and key must be contiguous");
  int num_heads, num_kv_heads;
  if (query.dim() == 2) {
    TORCH_CHECK(query.size(1) % head_size == 0,
                "musa_rotary_embedding: 2D query last dim must be divisible by head_size");
    num_heads = query.size(1) / head_size;
    num_kv_heads = key.size(1) / head_size;
  } else {
    num_heads = query.size(1);
    num_kv_heads = key.size(1);
  }
  int rot_dim = cos_sin_cache.size(1);

  const c10::musa::OptionalMUSAGuard guard(device_of(query));
  auto stream = c10::musa::getCurrentMUSAStream().stream();
  size_t smem_bytes = (size_t)rot_dim * query.element_size();

  auto dt = query.scalar_type();
  #define LAUNCH_KERNEL_TYPED(KERN_T, BLOCK_X_VAL, INTERLEAVE_VAL) \
    sphere_rope_kernel_v2<KERN_T, BLOCK_X_VAL, INTERLEAVE_VAL>     \
        <<<num_tokens, BLOCK_X_VAL, smem_bytes, stream>>>(         \
        positions.data_ptr<int64_t>(),                             \
        static_cast<KERN_T*>(query.data_ptr()),                    \
        static_cast<KERN_T*>(key.data_ptr()),                      \
        static_cast<const KERN_T*>(cos_sin_cache.data_ptr()),      \
        rot_dim, (int64_t)(num_heads * head_size), (int64_t)head_size, (int64_t)1, \
        (int64_t)(num_kv_heads * head_size), (int64_t)head_size, (int64_t)1, \
        num_heads, num_kv_heads);

  #define LAUNCH_KERNEL(BLOCK_X_VAL)                                  \
    do {                                                              \
      if (dt == at::ScalarType::Half) {                               \
        if (is_neox) LAUNCH_KERNEL_TYPED(__half, BLOCK_X_VAL, false)  \
        else         LAUNCH_KERNEL_TYPED(__half, BLOCK_X_VAL, true)   \
      } else if (dt == at::ScalarType::BFloat16) {                    \
        if (is_neox) LAUNCH_KERNEL_TYPED(__mt_bfloat16, BLOCK_X_VAL, false) \
        else         LAUNCH_KERNEL_TYPED(__mt_bfloat16, BLOCK_X_VAL, true)  \
      } else { TORCH_CHECK(false, "only fp16/bf16"); }                \
    } while (0);

  // Diagnostic variants (V1 = neox-only NEOX baseline kept for evidence trail; only callable
  // when is_neox=true — error otherwise so callers can't accidentally hit V1 with interleaved).
  // V2..V4 = is_neox-aware via INTERLEAVE template.
  // V5 (default ship) = adaptive BLOCK_X dispatch + is_neox-aware + env override.
  // Refined heuristic from per-shape sweep (S5000 64 MPs):
  //   T >= 512 → BLOCK=128 (≥8 CTAs/MP, max occupancy)
  //   T ∈ [256, 512) → BLOCK=256 (4 CTAs/MP — 128 too aggressive, 512 too fat)
  //   T < 256 AND num_heads >= 64 → BLOCK=512 (per-token work big enough to fatten)
  //   T < 128 → BLOCK=512 (launch-bound, fatten single CTA)
  //   T == 128 AND num_heads < 64 → BLOCK=256 (sweet spot)
  int block_x_v5;
  if (const char* e = std::getenv("SPHERE_ROPE_BLOCK_X")) {
    block_x_v5 = std::atoi(e);
  } else if (num_tokens >= 512) {
    block_x_v5 = 128;
  } else if (num_tokens >= 256) {
    block_x_v5 = 256;
  } else if (num_heads >= 64) {
    block_x_v5 = 512;
  } else if (num_tokens < 128) {
    block_x_v5 = 512;
  } else {
    block_x_v5 = 256;  // T == 128, H < 64
  }
  switch (variant) {
    case 1: {
      // V1 was implemented before INTERLEAVE template — keep neox-only diagnostic.
      TORCH_CHECK(is_neox, "variant=1 (V1 baseline) only supports is_neox=true");
      if (dt == at::ScalarType::Half) {
        sphere_rope_neox_kernel_v1<__half, 256><<<num_tokens, 256, smem_bytes, stream>>>(
            positions.data_ptr<int64_t>(),
            static_cast<__half*>(query.data_ptr()),
            static_cast<__half*>(key.data_ptr()),
            static_cast<const __half*>(cos_sin_cache.data_ptr()),
            rot_dim, (int64_t)(num_heads * head_size), (int64_t)head_size, (int64_t)1,
            (int64_t)(num_kv_heads * head_size), (int64_t)head_size, (int64_t)1, num_heads, num_kv_heads);
      } else if (dt == at::ScalarType::BFloat16) {
        sphere_rope_neox_kernel_v1<__mt_bfloat16, 256><<<num_tokens, 256, smem_bytes, stream>>>(
            positions.data_ptr<int64_t>(),
            static_cast<__mt_bfloat16*>(query.data_ptr()),
            static_cast<__mt_bfloat16*>(key.data_ptr()),
            static_cast<const __mt_bfloat16*>(cos_sin_cache.data_ptr()),
            rot_dim, (int64_t)(num_heads * head_size), (int64_t)head_size, (int64_t)1,
            (int64_t)(num_kv_heads * head_size), (int64_t)head_size, (int64_t)1, num_heads, num_kv_heads);
      } else { TORCH_CHECK(false, "only fp16/bf16"); }
      break;
    }
    case 2: LAUNCH_KERNEL(256); break;  // no launch_bounds, interleave-aware
    case 3: LAUNCH_KERNEL(512); break;  // BLOCK_X=512
    case 4: LAUNCH_KERNEL(128); break;  // BLOCK_X=128
    case 5: {
      if (block_x_v5 == 128) { LAUNCH_KERNEL(128) }
      else if (block_x_v5 == 256) { LAUNCH_KERNEL(256) }
      else { LAUNCH_KERNEL(512) }
      break;
    }
    default: TORCH_CHECK(false, "unknown variant");
  }
  #undef LAUNCH_KERNEL
  #undef LAUNCH_KERNEL_TYPED
}

}  // namespace vllm_musa

// Public C++ wrapper exposed to torch.ops via csrc/musa/torch_bindings.cpp.
// musa_ops_register binds this GLOBAL symbol (vllm-musa convention — see
// fused_add_rmsnorm.mu, cache_kernels.mu host wrappers).
void musa_rotary_embedding(
    torch::Tensor& positions,
    torch::Tensor& query,
    torch::Tensor& key,
    int64_t head_size,
    torch::Tensor& cos_sin_cache,
    bool is_neox) {
  vllm_musa::musa_rotary_embedding_impl(
      positions, query, key, head_size, cos_sin_cache, is_neox);
}
