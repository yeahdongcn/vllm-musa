# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MUSA-0109 / MUSA-0090 layer-3 attempt: bypass `draft_token_ids_copy_stream`
on MUSA so the H2D copy of draft tokens runs on the default stream (same
stream the runner's CUDAGraph replay used).

Why: with VLLM_MUSA_EAGLE_RUNNER=1, vllm's `_copy_draft_token_ids_to_cpu`
fails with "MUDNN err 999 = unknown error" because it copies from
runner-output (potentially CUDAGraph pool memory) on a DEDICATED stream
(`self.draft_token_ids_copy_stream`). Forcing default-stream avoids the
cross-stream interaction.

This is a non-destructive change: same-stream copy is semantically
identical to cross-stream copy + event-record + wait. The async-overlap
benefit goes away, but at BS=1 the copy is ~12 bytes (int32 [bs=1, 3])
so it doesn't matter.

Gated via VLLM_MUSA_DRAFT_COPY_DEFAULT_STREAM=1 (default OFF).
Enable only when running with VLLM_MUSA_EAGLE_RUNNER=1.
"""

PATCHES: list = []  # Monkey-patch only.

import logging
import os

_log = logging.getLogger(__name__)

_ENABLED = os.environ.get("VLLM_MUSA_DRAFT_COPY_DEFAULT_STREAM", "0") == "1"

if not _ENABLED:
    _log.info(
        "MUSA-0109: VLLM_MUSA_DRAFT_COPY_DEFAULT_STREAM=0; "
        "draft_token_ids_copy_stream stays on a dedicated stream (default vllm)"
    )
else:
    try:
        import torch
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError as exc:
        _log.warning(
            "MUSA-0109 draft_copy_stream: import failed (%s); patch disabled",
            exc,
        )
        GPUModelRunner = None

    if GPUModelRunner is not None and not getattr(
        GPUModelRunner, "_musa_draft_copy_stream_patched", False
    ):
        _ORIGINAL_INIT = GPUModelRunner.__init__

        def _musa_patched_init(self, *args, **kwargs):
            _ORIGINAL_INIT(self, *args, **kwargs)
            # Override the dedicated copy stream with the current default stream
            # AFTER the original init has constructed everything else.
            if getattr(self, "draft_token_ids_copy_stream", None) is not None:
                # Replace with current stream so the H2D copy stays on it.
                self.draft_token_ids_copy_stream = torch.cuda.current_stream()
                _log.info(
                    "MUSA-0109: draft_token_ids_copy_stream redirected to "
                    "torch.cuda.current_stream() to avoid CUDAGraph pool "
                    "cross-stream interaction"
                )
            if getattr(self, "valid_sampled_token_count_copy_stream", None) is not None:
                self.valid_sampled_token_count_copy_stream = torch.cuda.current_stream()
                _log.info(
                    "MUSA-0109: valid_sampled_token_count_copy_stream redirected "
                    "to torch.cuda.current_stream()"
                )

        GPUModelRunner.__init__ = _musa_patched_init
        GPUModelRunner._musa_draft_copy_stream_patched = True
        _log.info(
            "MUSA-0109: GPUModelRunner.__init__ monkey-patched to redirect "
            "copy streams to default stream (gate "
            "VLLM_MUSA_DRAFT_COPY_DEFAULT_STREAM=1)"
        )
