# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA-0087: Inductor template heuristics for device_type='musa'.

Without this registration, vllm/Inductor logs (per rank, per `mm` op):

    No template heuristic found - template_name=triton::mm,
    device_type=musa, op_name=mm. ... Using fallback
    TemplateConfigHeuristics instance.

The fallback returns an empty Triton config iterator, so Inductor
never attempts Triton autotune for `mm` on MUSA. With this
registration MUSA inherits the CUDA tile-size configs as a safe
starting point; per-shape MUSA tuning can be added incrementally.

Disable with: ``VLLM_MUSA_DISABLE_INDUCTOR_HEURISTICS=1``
"""

from __future__ import annotations

import os

from vllm.logger import init_logger

# vllm.logger.init_logger returns a Logger that supports info_once /
# warning_once. The previous use of logging.getLogger here raised
# AttributeError at call sites, which was caught by the broad except
# in vllm_musa/__init__.py and silently disabled MUSA-0087 heuristic
# registration — see PR #40 review comment.
logger = init_logger(__name__)

# Idempotency guard
_REGISTERED = False


def maybe_register_musa_template_heuristics() -> None:
    """Idempotent registration of MUSA Inductor template heuristics.

    Safe to call multiple times. Opportunistic — silently no-ops on old
    torch versions or when the disable env var is set.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    # MUSA-0087: registration is DEFAULT-OFF because the MUSA mm/bmm
    # heuristic surfaces a `RuntimeError: MUSA error: unknown error` in
    # Eagle3 draft compile (scalar_tensor lowering) at TP=8. Reproduced
    # in the M2.5 SOTA perf-sweep on yeahdongcn60 (2026-05-18). Opt in
    # via VLLM_MUSA_ENABLE_INDUCTOR_HEURISTICS=1 for benchmark probes
    # that don't trigger the crash.
    if os.getenv("VLLM_MUSA_ENABLE_INDUCTOR_HEURISTICS", "0") != "1":
        logger.info_once(
            "MUSA-0087: Inductor template heuristic registration is "
            "default-off (Eagle3 draft compile crash on TP=8 M2.5). "
            "Set VLLM_MUSA_ENABLE_INDUCTOR_HEURISTICS=1 to opt in for "
            "non-Eagle3 workloads."
        )
        _REGISTERED = True
        return

    try:
        from torch._inductor.kernel.bmm import bmm_template
        from torch._inductor.kernel.mm import mm_template
        from torch._inductor.template_heuristics.registry import (
            register_template_heuristic,
        )
        from torch._inductor.template_heuristics.triton import (
            AddMMConfigMixin,
            CUDAConfigHeuristic,
            MMTemplateConfigMixin,
        )
    except ImportError as exc:
        logger.warning_once(
            "MUSA-0087: torch._inductor template-heuristics API not "
            "available (%s); skipping MUSA registration. Compiled `mm` "
            "ops on MUSA will fall through to ATen via torchada.",
            exc,
        )
        _REGISTERED = True
        return

    # MUSAConfigHeuristic inherits all CUDA configs as a safe starting
    # point. Per-shape MUSA tuning can override defaults in future PRs.
    class MUSAConfigHeuristic(CUDAConfigHeuristic):
        """MUSA template heuristic.

        Phase-1: inherit CUDA tile-size configs verbatim. MTT S5000 has
        the same warp size (32) and max threads/block (1024) as CUDA
        Hopper-class hardware so the CUDA defaults are a reasonable
        starting point. Phase-2 can override `default_mm_config` etc.
        with MUSA-tuned tile sizes once profiled.
        """

        def __init__(self) -> None:
            super().__init__()

    # Register the four common (template, device, op) combinations.
    # The decorator-based pattern matches CUDAMMTemplateConfigHeuristic
    # registration in torch._inductor.template_heuristics.triton.
    @register_template_heuristic(mm_template.uid, "musa")
    @register_template_heuristic(bmm_template.uid, "musa")
    class MUSAMMTemplateConfigHeuristic(  # noqa: F841 — register-only
        MMTemplateConfigMixin, MUSAConfigHeuristic
    ):
        """Standard MM template heuristic for MUSA."""

    @register_template_heuristic(mm_template.uid, "musa", op_name="addmm")
    @register_template_heuristic(bmm_template.uid, "musa", op_name="baddbmm")
    class MUSAAddMMTemplateConfigHeuristic(  # noqa: F841 — register-only
        AddMMConfigMixin, MUSAConfigHeuristic
    ):
        """Addmm/baddbmm template heuristic for MUSA."""

    # scaled_mm (FP8 path) — the M2.5 FP8 layers compile through this.
    @register_template_heuristic(mm_template.uid, "musa", op_name="scaled_mm")
    class MUSAScaledMMTemplateConfigHeuristic(  # noqa: F841 — register-only
        MMTemplateConfigMixin, MUSAConfigHeuristic
    ):
        """Scaled MM (FP8) template heuristic for MUSA."""

    _REGISTERED = True
    logger.info_once(
        "MUSA-0087: Inductor template heuristics registered for "
        "device_type='musa' (mm + bmm + addmm + baddbmm + scaled_mm)."
    )
