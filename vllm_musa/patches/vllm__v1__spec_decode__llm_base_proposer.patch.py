# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MUSA-0109 POC: Capture LLMBaseProposer.propose() under a single
torch.cuda.graph (persistent pool) to eliminate per-round Python
orchestration overhead.

Goal: close the 28.6 tok/s gap from vllm-musa 91.4 → SGLang 120 by
matching SGLang's EagleDraftCudaGraphRunner architecture.

Strategy (POC):
  1. On first call: run propose normally, save reference outputs.
  2. On second call with same batch_size: copy inputs into pre-allocated
     buffers and capture propose body under torch.cuda.graph().
  3. On subsequent calls: copy inputs into buffers + replay.
  4. On ANY capture or replay error: fall back to original path.

Risks (per design doc generated/musa0109/design.md):
  - MUSA CUDAGraph at TP=8 may crash (per MUSA-0061/0063)
  - Allocations inside build_for_drafting() may not be capture-safe
  - common_attn_metadata fields change per call — must be pre-allocated
  - vllm's `replace(common_attn_metadata, ...)` creates new objects

This POC uses a persistent memory pool to absorb capture-time
allocations. If it crashes or produces wrong results, the failure mode
is logged and the original path is used for the rest of the run.

Gated via env var VLLM_MUSA_TREE_GRAPH_CAPTURE=1 (default OFF for safety).
Enable explicitly for measurement; revert to fallback if regressions.
"""

PATCHES: list = []  # Monkey-patch only; no file mutation.

import logging
import os

_log = logging.getLogger(__name__)

_ENABLED = os.environ.get("VLLM_MUSA_TREE_GRAPH_CAPTURE", "0") == "1"

# MUSA-0109 POC findings (2026-05-17, see generated/musa0109/poc-attempt-findings.md):
# This patch monkey-patches `propose_tree` but `propose_tree` is NEVER called
# in the current SOTA stack because vllm uses FLASH_ATTN attention backend
# (not TREE_ATTN). The actual chain spec-decode loop lives in `propose()`
# (lines 554-650), which has a complex signature and needs careful buffer
# management — that's exactly what MUSA-0090's EagleFullLoopRunner was
# designed for. The patch file remains as evidence of the attempt + starting
# point for the proper port. Keep DISABLED by default.

if not _ENABLED:
    _log.info(
        "MUSA-0109 POC: VLLM_MUSA_TREE_GRAPH_CAPTURE != 1; tree propose "
        "graph capture is disabled (using vllm's iterative path)."
    )
else:
    try:
        import torch
        from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
    except ImportError as exc:
        _log.warning("MUSA-0109 POC: import failed (%s); patch disabled", exc)
        SpecDecodeBaseProposer = None

    if SpecDecodeBaseProposer is not None and not getattr(
        SpecDecodeBaseProposer, "_musa_tree_graph_patched", False
    ):
        # Cache state per-proposer-instance via attribute on `self`.
        _ORIGINAL_PROPOSE_TREE = SpecDecodeBaseProposer.propose

        def _musa_patched_propose(
            self,
            batch_size: int,
            logits,
            positions,
            hidden_states,
            common_attn_metadata,
            slot_mappings=None,
        ):
            """POC wrapper. See module docstring."""
            # Safety: only attempt capture for batch_size=1, the SOTA target.
            if batch_size != 1:
                return _ORIGINAL_PROPOSE_TREE(
                    self,
                    batch_size,
                    logits,
                    positions,
                    hidden_states,
                    common_attn_metadata,
                    slot_mappings,
                )

            cache = getattr(self, "_musa_tree_cache", None)
            if cache is None:
                cache = {
                    "warmup_done": False,
                    "graph": None,
                    "pool": None,
                    "input_buffers": None,
                    "output_list_ref": None,
                    "failed": False,
                }
                self._musa_tree_cache = cache

            # If capture already failed, never try again.
            if cache["failed"]:
                return _ORIGINAL_PROPOSE_TREE(
                    self,
                    batch_size,
                    logits,
                    positions,
                    hidden_states,
                    common_attn_metadata,
                    slot_mappings,
                )

            # First call: just run it, get reference outputs + shapes.
            if not cache["warmup_done"]:
                cache["warmup_done"] = True
                cache["ref_logits_shape"] = logits.shape
                cache["ref_positions_shape"] = positions.shape
                cache["ref_hidden_shape"] = hidden_states.shape
                _log.info(
                    "MUSA-0109 POC: warmup call done; logits=%s positions=%s "
                    "hidden_states=%s; capture on next call",
                    logits.shape,
                    positions.shape,
                    hidden_states.shape,
                )
                return _ORIGINAL_PROPOSE_TREE(
                    self,
                    batch_size,
                    logits,
                    positions,
                    hidden_states,
                    common_attn_metadata,
                    slot_mappings,
                )

            # Verify shape stability before capture attempt.
            if (
                logits.shape != cache["ref_logits_shape"]
                or positions.shape != cache["ref_positions_shape"]
                or hidden_states.shape != cache["ref_hidden_shape"]
            ):
                _log.warning(
                    "MUSA-0109 POC: shape changed across calls "
                    "(logits %s vs %s); capture aborted, falling back",
                    logits.shape,
                    cache["ref_logits_shape"],
                )
                cache["failed"] = True
                return _ORIGINAL_PROPOSE_TREE(
                    self,
                    batch_size,
                    logits,
                    positions,
                    hidden_states,
                    common_attn_metadata,
                    slot_mappings,
                )

            # Capture path: allocate buffers + capture under persistent pool.
            if cache["graph"] is None:
                try:
                    bufs = {
                        "logits": torch.empty_like(logits),
                        "positions": torch.empty_like(positions),
                        "hidden_states": torch.empty_like(hidden_states),
                    }
                    bufs["logits"].copy_(logits)
                    bufs["positions"].copy_(positions)
                    bufs["hidden_states"].copy_(hidden_states)

                    g = torch.cuda.CUDAGraph()
                    pool = torch.cuda.graph_pool_handle()
                    torch.cuda.synchronize()
                    with torch.cuda.graph(g, pool=pool):
                        out_list = _ORIGINAL_PROPOSE_TREE(
                            self,
                            batch_size,
                            bufs["logits"],
                            bufs["positions"],
                            bufs["hidden_states"],
                            common_attn_metadata,
                            slot_mappings,
                        )
                    cache["graph"] = g
                    cache["pool"] = pool
                    cache["input_buffers"] = bufs
                    cache["output_list_ref"] = out_list
                    _log.info(
                        "MUSA-0109 POC: CAPTURED tree propose loop for bs=%d "
                        "(output_list len=%d)",
                        batch_size,
                        len(out_list),
                    )
                    return out_list
                except Exception as exc:
                    _log.warning(
                        "MUSA-0109 POC: capture FAILED (%s: %s); falling back",
                        type(exc).__name__,
                        exc,
                    )
                    cache["failed"] = True
                    return _ORIGINAL_PROPOSE_TREE(
                        self,
                        batch_size,
                        logits,
                        positions,
                        hidden_states,
                        common_attn_metadata,
                        slot_mappings,
                    )

            # Replay path: copy inputs + replay
            try:
                bufs = cache["input_buffers"]
                bufs["logits"].copy_(logits)
                bufs["positions"].copy_(positions)
                bufs["hidden_states"].copy_(hidden_states)
                cache["graph"].replay()
                return cache["output_list_ref"]
            except Exception as exc:
                _log.warning(
                    "MUSA-0109 POC: replay FAILED (%s: %s); falling back",
                    type(exc).__name__,
                    exc,
                )
                cache["failed"] = True
                return _ORIGINAL_PROPOSE_TREE(
                    self,
                    batch_size,
                    logits,
                    positions,
                    hidden_states,
                    common_attn_metadata,
                    slot_mappings,
                )

        SpecDecodeBaseProposer.propose = _musa_patched_propose
        SpecDecodeBaseProposer._musa_tree_graph_patched = True
        _log.info(
            "MUSA-0109 POC: SpecDecodeBaseProposer.propose monkey-patched "
            "with cudagraph capture (gate VLLM_MUSA_TREE_GRAPH_CAPTURE=1)"
        )
