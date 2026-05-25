# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MUSA-0164 (v2): replace upstream ``torch.ops._C.<op>`` / ``torch.ops._moe_C.<op>``
impls with MUSA-native kernels via ``torchada.replace_op_impl``.

This supersedes the earlier ``_custom_ops_override.py`` Python-wrapper monkey
patch. That approach worked for runtime correctness but Dynamo traced our
``_C_musa_ops.musa_*`` symbols into the FX graph, breaking Inductor fusion
patterns that key on the upstream ``_C.<op>`` symbol. Measured regression:

    BS=1  wash    BS=4  -14%    BS=16  -5%    BS=64  -7%

Using ``torchada.replace_op_impl`` keeps the upstream op name in the FX graph
(``qk_norm_rope_fusion.py``, ``act_quant_fusion.py``, ``rms_quant_fusion.py``
all still match) and only swaps the kernel behind the dispatcher table entry.

## Coverage

Wired (when gate is on):
    - rotary_embedding              → musa_rotary_embedding (MUSA-0154)
    - fused_qk_norm_rope            → musa_fused_qk_norm_rope (MUSA-0157)
    - silu_and_mul_per_block_quant  → musa_silu_and_mul_per_block_quant (MUSA-0158)
    - rms_norm_dynamic_per_token_quant
                                    → musa_rms_norm_dynamic_per_token_quant (MUSA-0159)
    - merge_attn_states             → musa_merge_attn_states (MUSA-0161)
    - topk_softmax                  → musa_topk_softmax (MUSA-0162) [_moe_C]

Out of scope:
    - rms_norm_static_fp8_quant (MUSA-0156) — exists only as a fusion-pass
      target, not a call site. Address by patching the fusion pass's op map
      (MUSA-0166).
    - top_k_top_p_sampling_from_probs (MUSA-0160) — class-level sampler
      integration (MUSA-0165).

## Why each shim is a Python wrapper

For each op the upstream schema includes optional args (``scale_ub``,
``residual``, ``rope_dim_offset``, ``bias``, etc.) that our native kernels
do not implement. Each shim asserts those args are unused on the M2.5 hot
path and forwards to the OpOverload, so the dispatch tax is one Python
frame per call. For the *fused* ops (which Inductor emits to replace 2-3
upstream calls), this frame is amortized over the work that the unfused
path would have done.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torchada

logger = logging.getLogger(__name__)

_OVERRIDES_APPLIED = False

_GATE_ENV = "VLLM_MUSA_OP_PERF_OVERRIDES"
_SKIP_ENV = "VLLM_MUSA_OVERRIDE_SKIP"


def _gate_enabled() -> bool:
    """Default OFF.

    Final today-vs-today A/B on yeahdongcn60 (4k/1k cookbook SOTA stack):

      BS  baseline (3984816b7)  v2 (current)  delta
      1   35.42                 35.14         wash
      4   72.76                 75.65         +4.0%  ← real win
      16  89.53                 85.53         -4.5%  ← real regression
      64  103.78                98.95         -4.7%  ← real regression

    BS=16 and BS=64 are the higher-throughput production regime. The
    +4% BS=4 win does not outweigh the BS>=16 loss for typical serving
    workloads, so the gate ships default OFF. Opt in for BS=1-4
    latency-optimized workloads via VLLM_MUSA_OP_PERF_OVERRIDES=1.
    Per-op skip via VLLM_MUSA_OVERRIDE_SKIP=op1,op2,... still available.

    BS>=16 regression is tracked as MUSA-0169 — measure the op-perf
    kernels at BS=16/64 shapes specifically and identify which shim(s)
    regress at those shapes (microbenches only covered BS=1-4 shapes).
    """
    value = os.environ.get(_GATE_ENV, "0")
    return value.lower() in ("1", "true", "yes", "on")


def _skip_set() -> set[str]:
    """Per-op opt-out — comma-separated op names in ``VLLM_MUSA_OVERRIDE_SKIP``.

    Names are the upstream op names without namespace, e.g.
    ``VLLM_MUSA_OVERRIDE_SKIP=silu_and_mul_per_block_quant,topk_softmax``.
    Used by the BS=16 bisect to identify which shim is the regressor.
    """
    value = os.environ.get(_SKIP_ENV, "")
    return {name.strip() for name in value.split(",") if name.strip()}


def _musa_op(name: str) -> Any | None:
    """Look up a MUSA-native op on ``_C_musa_ops``, returning None if missing."""
    ns = getattr(torch.ops, "_C_musa_ops", None)
    if ns is None:
        return None
    op = getattr(ns, name, None)
    if op is None:
        return None
    return op.default


# --- shim factories ----------------------------------------------------------
#
# Each factory returns a callable with the *upstream* op's full schema. The
# dispatcher passes us every positional/optional arg the upstream schema
# declares; we forward the subset our MUSA kernel supports and fail loudly
# if the unsupported optional args are actually set.


def _shim_rotary_embedding() -> Any:
    """Upstream schema (vllm/csrc/torch_bindings.cpp)::

        rotary_embedding(Tensor positions, Tensor! query, Tensor!? key,
                         int head_size, Tensor cos_sin_cache, bool is_neox,
                         int rope_dim_offset=0, bool inverse=False) -> ()

    Our musa_rotary_embedding takes only the first six args.
    """
    musa = _musa_op("musa_rotary_embedding")

    def shim(
        positions, query, key, head_size, cos_sin_cache, is_neox,
        rope_dim_offset=0, inverse=False,
    ):
        assert rope_dim_offset == 0, (
            "MUSA musa_rotary_embedding does not implement rope_dim_offset!=0"
        )
        assert not inverse, (
            "MUSA musa_rotary_embedding does not implement inverse=True"
        )
        assert key is not None, (
            "MUSA musa_rotary_embedding does not implement key=None"
        )
        return musa(positions, query, key, head_size, cos_sin_cache, is_neox)

    return shim


def _shim_fused_qk_norm_rope() -> Any:
    """Upstream schema::

        fused_qk_norm_rope(Tensor! qkv, int num_heads_q, int num_heads_k,
            int num_heads_v, int head_dim, float eps,
            Tensor q_weight, Tensor k_weight, Tensor cos_sin_cache,
            bool is_neox, Tensor position_ids,
            int forced_token_heads_per_warp=-1) -> ()

    Schema matches our musa_fused_qk_norm_rope exactly.
    """
    musa = _musa_op("musa_fused_qk_norm_rope")

    def shim(
        qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps,
        q_weight, k_weight, cos_sin_cache, is_neox, position_ids,
        forced_token_heads_per_warp=-1,
    ):
        return musa(
            qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps,
            q_weight, k_weight, cos_sin_cache, is_neox, position_ids,
            forced_token_heads_per_warp,
        )

    return shim


def _shim_silu_and_mul_per_block_quant() -> Any:
    """Upstream schema::

        silu_and_mul_per_block_quant(Tensor! out, Tensor input,
            Tensor! scales, int group_size,
            Tensor? scale_ub=None, bool is_scale_transposed=False) -> ()
    """
    musa = _musa_op("musa_silu_and_mul_per_block_quant")

    def shim(out, input, scales, group_size, scale_ub=None,
             is_scale_transposed=False):
        assert scale_ub is None, (
            "MUSA musa_silu_and_mul_per_block_quant does not implement scale_ub"
        )
        assert not is_scale_transposed, (
            "MUSA musa_silu_and_mul_per_block_quant does not implement "
            "is_scale_transposed=True"
        )
        return musa(out, input, scales, group_size)

    return shim


def _shim_rms_norm_dynamic_per_token_quant() -> Any:
    """Upstream schema::

        rms_norm_dynamic_per_token_quant(Tensor! result, Tensor input,
            Tensor weight, Tensor! scale, float epsilon,
            Tensor? scale_ub, Tensor!? residual) -> ()
    """
    musa = _musa_op("musa_rms_norm_dynamic_per_token_quant")

    def shim(out, input, weight, scales, epsilon,
             scale_ub=None, residual=None):
        assert scale_ub is None, (
            "MUSA musa_rms_norm_dynamic_per_token_quant does not implement scale_ub"
        )
        assert residual is None, (
            "MUSA musa_rms_norm_dynamic_per_token_quant does not implement residual"
        )
        return musa(out, input, weight, scales, epsilon)

    return shim


def _shim_merge_attn_states() -> Any:
    """Upstream schema::

        merge_attn_states(Tensor! output, Tensor!? output_lse,
            Tensor prefix_output, Tensor prefix_lse,
            Tensor suffix_output, Tensor suffix_lse,
            int!? prefill_tokens_with_context,
            Tensor? output_scale=None) -> ()
    """
    musa = _musa_op("musa_merge_attn_states")

    def shim(output, output_lse, prefix_output, prefix_lse,
             suffix_output, suffix_lse,
             prefill_tokens_with_context=None, output_scale=None):
        assert prefill_tokens_with_context is None, (
            "MUSA musa_merge_attn_states does not implement prefill_tokens_with_context"
        )
        assert output_scale is None, (
            "MUSA musa_merge_attn_states does not implement output_scale"
        )
        return musa(output, output_lse, prefix_output, prefix_lse,
                    suffix_output, suffix_lse)

    return shim


def _shim_topk_softmax() -> Any:
    """Upstream schema (vllm/csrc/moe/torch_bindings.cpp)::

        topk_softmax(Tensor! topk_weights, Tensor! topk_indices,
            Tensor! token_expert_indices, Tensor gating_output,
            bool renormalize, Tensor? bias) -> ()
    """
    musa = _musa_op("musa_topk_softmax")

    def shim(topk_weights, topk_indices, token_expert_indices,
             gating_output, renormalize, bias=None):
        assert bias is None, (
            "MUSA musa_topk_softmax does not implement bias"
        )
        return musa(topk_weights, topk_indices, token_expert_indices,
                    gating_output, renormalize)

    return shim


# Ordered to match the load sequence; each tuple is (namespace, op, factory).
_REGISTRATIONS = (
    ("_C", "rotary_embedding", _shim_rotary_embedding),
    ("_C", "fused_qk_norm_rope", _shim_fused_qk_norm_rope),
    ("_C", "silu_and_mul_per_block_quant", _shim_silu_and_mul_per_block_quant),
    ("_C", "rms_norm_dynamic_per_token_quant", _shim_rms_norm_dynamic_per_token_quant),
    ("_C", "merge_attn_states", _shim_merge_attn_states),
    ("_moe_C", "topk_softmax", _shim_topk_softmax),
)


def apply_overrides() -> None:
    """Register MUSA-native shims as the dispatcher impl for the upstream
    op names. Idempotent — safe to call multiple times."""
    global _OVERRIDES_APPLIED
    if _OVERRIDES_APPLIED:
        return

    if not _gate_enabled():
        logger.debug(
            "MUSA-0164: dispatcher overrides are gated off "
            "(set %s=1 to opt in).", _GATE_ENV,
        )
        _OVERRIDES_APPLIED = True
        return

    skip = _skip_set()
    applied: list[str] = []
    skipped: list[str] = []
    for namespace, op_name, factory in _REGISTRATIONS:
        if op_name in skip:
            skipped.append(f"{namespace}::{op_name}=env_skip")
            continue
        if _musa_op(f"musa_{op_name}") is None:
            skipped.append(f"{namespace}::{op_name}=no_musa_op")
            continue
        shim = factory()
        torchada.replace_op_impl(namespace, op_name, shim, dispatch_key="CUDA")
        applied.append(f"{namespace}::{op_name}")

    _OVERRIDES_APPLIED = True
    logger.info(
        "MUSA-0164: applied %d dispatcher overrides (%s)%s",
        len(applied), ", ".join(applied),
        f"; skipped: {skipped}" if skipped else "",
    )
