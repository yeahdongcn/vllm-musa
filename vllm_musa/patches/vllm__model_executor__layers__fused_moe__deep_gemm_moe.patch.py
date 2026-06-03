# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.model_executor.layers.fused_moe.deep_gemm_moe.DeepGemmExperts
"""

# MUSA-0301: v0.22 moved this module to `fused_moe/experts/deep_gemm_moe.py`
# (patched separately by the experts__deep_gemm_moe patch). This old-path patch
# matches no module on v0.22 and silently no-ops, so mark it optional +
# version-gated: patch_report() then reports it as a clean optional skip rather
# than a required-missing failure. Retire once no supported vLLM snapshot < v0.22
# remains. (PATCHES below is untouched — application behaviour is unchanged.)
PATCH_REQUIRED = False
PATCH_VERSION_RANGE = "<v0.22"
PATCH_REMOVAL_CONDITION = (
    "superseded by fused_moe/experts/deep_gemm_moe; retire when no supported "
    "vLLM snapshot predates v0.22"
)

PATCHES = [
    # Patch DeepGemmExperts.apply where m_grouped_fp8_gemm_nt_contiguous need a2q_scale is_contiguous
    (
        """        mm2_out = _resize_cache(workspace2, (M_sum, K))
        m_grouped_fp8_gemm_nt_contiguous(
            (a2q, a2q_scale), (w2, self.w2_scale), mm2_out, expert_ids
        )
""",
        """        mm2_out = _resize_cache(workspace2, (M_sum, K))
        a2q_scale = a2q_scale.contiguous()
        m_grouped_fp8_gemm_nt_contiguous(
            (a2q, a2q_scale), (w2, self.w2_scale), mm2_out, expert_ids
        )
""",
    ),
]
