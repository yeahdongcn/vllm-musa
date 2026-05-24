#include "cache.h"
#include "cuda_utils.h"
#include "musa_ops.h"
#include "core/registration.h"

#include <torch/library.h>
#include <torch/version.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _musa_ops), musa_ops) {
#ifdef USE_MUSA
  musa_ops.def(
      "musa_fused_gemv_moe(Tensor! A, Tensor! B, Tensor! C, Tensor? A_scale, Tensor? B_scale,"
      "Tensor! topk_weights, Tensor! topk_ids, bool mul_routed_weight, int topk, bool use_int4_w4a16,"
      "bool use_swigelu) -> ()");
  musa_ops.impl("musa_fused_gemv_moe", torch::kMUSA, &musa_fused_gemv_moe);

  musa_ops.def(
      "musa_fused_gemv(Tensor! A, Tensor! B, Tensor! C, Tensor? A_scale, Tensor? B_scale,"
      "bool use_int4_w4a16, bool use_swigelu, bool use_rms_norm, Tensor? gamma,"
      "float eps) -> ()");
  musa_ops.impl("musa_fused_gemv", torch::kMUSA, &musa_fused_gemv);

  musa_ops.def(
      "musa_fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, "
      "float eps) -> ()");
  musa_ops.impl("musa_fused_add_rms_norm", torch::kMUSA,
                &musa_fused_add_rms_norm);

  musa_ops.def(
      "musa_reshape_and_cache_flash_nhd(Tensor key, Tensor value, "
      "Tensor! key_cache, Tensor! value_cache, Tensor slot_mapping) -> ()");
  musa_ops.impl("musa_reshape_and_cache_flash_nhd", torch::kMUSA,
                &musa_reshape_and_cache_flash_nhd);

  musa_ops.def(
      "per_token_group_fp8_quant(Tensor input, Tensor! output_q, Tensor! "
      "output_s, "
      "int group_size, float eps, float fp8_min, float fp8_max, bool "
      "scale_ue8m0, bool dummy_is_scale_transposed, bool dummy_is_tma_aligned "
      ") -> ()");
  musa_ops.impl("per_token_group_fp8_quant", torch::kMUSA,
           &per_token_group_quant_fp8);

  musa_ops.def(
      "musa_rotary_embedding(Tensor positions, Tensor! query, Tensor! key, "
      "int head_size, Tensor cos_sin_cache, bool is_neox) -> ()");
  musa_ops.impl("musa_rotary_embedding", torch::kMUSA, &musa_rotary_embedding);

  musa_ops.def(
      "silu_and_mul_per_token_group_fp8_quant(Tensor input, Tensor! output_q, "
      "Tensor! output_s, int group_size, float eps, float fp8_min, "
      "float fp8_max) -> ()");
  musa_ops.impl("silu_and_mul_per_token_group_fp8_quant", torch::kMUSA,
                &silu_and_mul_per_token_group_fp8_quant);

  musa_ops.def(
      "musa_rms_norm_dynamic_per_token_quant(Tensor! out, Tensor input, "
      "Tensor weight, Tensor! scales, float epsilon) -> ()");
  musa_ops.impl("musa_rms_norm_dynamic_per_token_quant", torch::kMUSA,
                &musa_rms_norm_dynamic_per_token_quant);

  musa_ops.def(
      "musa_silu_and_mul_per_block_quant(Tensor! out, Tensor input, "
      "Tensor! scales, int group_size) -> ()");
  musa_ops.impl("musa_silu_and_mul_per_block_quant", torch::kMUSA,
                &musa_silu_and_mul_per_block_quant);

  musa_ops.def(
      "musa_rms_norm_static_fp8_quant(Tensor! out, Tensor input, "
      "Tensor weight, Tensor scale, float epsilon) -> ()");
  musa_ops.impl("musa_rms_norm_static_fp8_quant", torch::kMUSA,
                &musa_rms_norm_static_fp8_quant);

  musa_ops.def(
      "musa_merge_attn_states(Tensor! output, Tensor!? output_lse, "
      "Tensor prefix_output, Tensor prefix_lse, "
      "Tensor suffix_output, Tensor suffix_lse) -> ()");
  musa_ops.impl("musa_merge_attn_states", torch::kMUSA, &musa_merge_attn_states);

  musa_ops.def(
      "musa_top_k_top_p_sampling_from_probs(Tensor probs, Tensor! output, "
      "Tensor!? maybe_indices, Tensor? maybe_top_k_arr, float top_k_val, "
      "Tensor? maybe_top_p_arr, float top_p_val, bool deterministic, "
      "Generator? gen) -> ()");
  musa_ops.impl("musa_top_k_top_p_sampling_from_probs", torch::kMUSA,
                &musa_top_k_top_p_sampling_impl);

  musa_ops.def(
      "musa_fused_qk_norm_rope(Tensor! qkv, int num_heads_q, "
      "int num_heads_k, int num_heads_v, int head_dim, float eps, "
      "Tensor q_weight, Tensor k_weight, Tensor cos_sin_cache, "
      "bool is_neox, Tensor position_ids, "
      "int forced_token_heads_per_warp=-1) -> ()");
  musa_ops.impl("musa_fused_qk_norm_rope", torch::kMUSA,
                &musa_fused_qk_norm_rope);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
