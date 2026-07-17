# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MUSA repetition-penalty path.

vLLM applies the repetition penalty through
``vllm._custom_ops.apply_repetition_penalties``, which dispatches to the native
``_C::apply_repetition_penalties_`` kernel whenever the logits report ``is_cuda``.
On MUSA that kernel is not registered and torchada reports ``is_cuda=True`` for
musa tensors, so the native call raises ``NotImplementedError`` and the upstream
torch fallback is never reached.

This module installs a musa implementation that is numerically identical to the
upstream torch fallback, does the scaling in a single vocab-wide pass (no
``[num_seqs, vocab]`` penalty materialization), and is CUDAGraph-capture safe (no
host-to-device scalar copies; the ``1.0`` no-op constant is cached per
device/dtype at first use).
"""

import torch
import vllm._custom_ops as vllm_custom_ops
from vllm.platforms import current_platform

_one_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
_original_apply_repetition_penalties = None


def _one_like(tensor: torch.Tensor) -> torch.Tensor:
    key = (tensor.device, tensor.dtype)
    one = _one_cache.get(key)
    if one is None:
        one = torch.ones((), device=tensor.device, dtype=tensor.dtype)
        _one_cache[key] = one
    return one


def apply_repetition_penalties_musa(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    """Apply repetition penalties in place on MUSA.

    For a token that appears in the prompt or output, a positive logit is divided
    by the penalty and a non-positive logit is multiplied by it; unseen tokens are
    left unchanged. Equivalent to ``apply_repetition_penalties_torch`` without the
    ``[num_seqs, vocab]`` ``repeat``.
    """
    rep = repetition_penalties.unsqueeze(dim=1)
    appeared = prompt_mask | output_mask
    scale = torch.where(logits > 0, 1.0 / rep, rep)
    scale = torch.where(appeared, scale, _one_like(logits))
    logits.mul_(scale)


def _apply_repetition_penalties(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    if current_platform.is_musa() and logits.device.type == "musa":
        apply_repetition_penalties_musa(
            logits, prompt_mask, output_mask, repetition_penalties
        )
        return
    _original_apply_repetition_penalties(
        logits, prompt_mask, output_mask, repetition_penalties
    )


def install_penalty_hook() -> None:
    global _original_apply_repetition_penalties
    if _original_apply_repetition_penalties is not None:
        return
    _original_apply_repetition_penalties = vllm_custom_ops.apply_repetition_penalties
    vllm_custom_ops.apply_repetition_penalties = _apply_repetition_penalties
