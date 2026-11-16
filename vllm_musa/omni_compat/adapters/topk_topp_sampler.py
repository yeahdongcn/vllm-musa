# SPDX-License-Identifier: Apache-2.0
"""``apply_top_k_top_p`` on a pin that predates its ``has_topp_only_rows`` hint.

vLLM 0.33 added a fourth parameter::

    def apply_top_k_top_p(logits, k, p, has_topp_only_rows: bool = True)

documented as "must be True if any row has p < 1", and forwarded to the Triton
kernel so it can skip the top-p pass when no row needs it. vLLM-Omni main passes
it by keyword from the Qwen3-Omni and MiniCPM-o 4.5 talker samplers, so on the
pinned three-parameter function every talker sampling step raises TypeError.

Dropping the argument is safe in both directions, which is why this is an
adapter rather than a declared-unsupported path:

* ``True`` is the parameter's default and means "take the full top-p path",
  which is exactly and only what the pinned function does.
* ``False`` means no row needs top-p. The pinned function still runs the top-p
  masking, and masking with ``p == 1`` keeps the whole distribution, so the
  result is identical; the pin merely does work 0.33 would have skipped.

Delete this adapter when ``third_party/PINS`` reaches a vLLM whose
``apply_top_k_top_p`` takes the parameter itself.
"""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

ADAPTED = "__vllm_musa_drops_has_topp_only_rows__"


def apply(module: ModuleType) -> list[str]:
    """Let the pinned ``apply_top_k_top_p`` accept and ignore the newer hint."""
    original = getattr(module, "apply_top_k_top_p", None)
    if original is None or getattr(original, ADAPTED, False):
        return []

    try:
        parameters = inspect.signature(original).parameters
    except (TypeError, ValueError):  # pragma: no cover - C callables
        parameters = {}
    if "has_topp_only_rows" in parameters:
        return []  # the pin already takes it; nothing to adapt

    @functools.wraps(original)
    def apply_top_k_top_p(*args, has_topp_only_rows=None, **kwargs):
        # The pin always takes the full top-p path, which is what the hint's
        # True value selects and is equivalent to its False value.
        del has_topp_only_rows
        return original(*args, **kwargs)

    setattr(apply_top_k_top_p, ADAPTED, True)
    module.apply_top_k_top_p = apply_top_k_top_p
    return ["apply_top_k_top_p"]
