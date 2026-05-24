#include <optional>
#include <torch/library.h>

#include "core/scalar_type.hpp"

#include <vector>

void musa_fused_gemv_moe(
    torch::Tensor &A,
    torch::Tensor &B,
    torch::Tensor &C,
    const c10::optional<torch::Tensor> &A_scale,
    const c10::optional<torch::Tensor> &B_scale,
    torch::Tensor &topk_weights,
    torch::Tensor &topk_ids,
    bool mul_routed_weight,
    int64_t topk,
    bool use_int4_w4a16,
    bool use_swigelu);

void musa_fused_gemv(
    torch::Tensor &A,
    torch::Tensor &B,
    torch::Tensor &C,
    const c10::optional<torch::Tensor> &A_scale,
    const c10::optional<torch::Tensor> &B_scale,
    bool use_int4_w4a16,
    bool use_swigelu,
    bool use_rms_norm,
    const c10::optional<torch::Tensor> &gamma,
    double eps);

void musa_fused_add_rms_norm(
    torch::Tensor &input,
    torch::Tensor &residual,
    torch::Tensor &weight,
    double eps);

void musa_rotary_embedding(
    torch::Tensor &positions,
    torch::Tensor &query,
    torch::Tensor &key,
    int64_t head_size,
    torch::Tensor &cos_sin_cache,
    bool is_neox);

void musa_reshape_and_cache_flash_nhd(
    torch::Tensor &key,
    torch::Tensor &value,
    torch::Tensor &key_cache,
    torch::Tensor &value_cache,
    torch::Tensor &slot_mapping);

void per_token_group_quant_fp8(
    const torch::Tensor& input,
    torch::Tensor& output_q, torch::Tensor& output_s,
    int64_t group_size, double eps, double fp8_min,
    double fp8_max, bool scale_ue8m0,
    bool dummy_is_scale_transposed = false,
    bool dummy_is_tma_aligned = false);

void silu_and_mul_per_token_group_fp8_quant(
    const torch::Tensor& input,
    torch::Tensor& output_q, torch::Tensor& output_s,
    int64_t group_size, double eps, double fp8_min,
    double fp8_max);

void musa_silu_and_mul_per_block_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor& scales,
    int64_t group_size);

void musa_rms_norm_static_fp8_quant(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& scale,
    double epsilon);

void musa_fused_qk_norm_rope(
    torch::Tensor& qkv,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
    double eps,
    torch::Tensor& q_weight,
    torch::Tensor& k_weight,
    torch::Tensor& cos_sin_cache,
    bool is_neox,
    torch::Tensor& position_ids,
    int64_t forced_token_heads_per_warp);
