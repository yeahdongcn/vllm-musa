#!/usr/bin/env python3
"""MUSA-0161 bench: merge_attn_states (csrc/attention/merge_attn_states.cu).

Chunked-prefill merge of two partial attention outputs (prefix + suffix)
into a single output via LSE-based combine. Section 2.2 of arxiv 2501.01005.

Usage:  python3 benchmarks/op_perf/bench_merge-attn-states.py
"""
from __future__ import annotations
import json, os, statistics, time, torch
import torchada  # noqa: F401
import torch_musa  # noqa: F401
import os as _os
for _so in ("/ws/vllm_musa/_C.cpython-310-x86_64-linux-gnu.so",
            "/ws/vllm/_C.cpython-310-x86_64-linux-gnu.so"):
    if _os.path.exists(_so):
        torch.ops.load_library(_so)
del _os

_DEVICE = "musa"


def _sync(): torch.musa.synchronize()
def _peak(): return float(os.environ.get("OP_PERF_PEAK_GDDR6_GBPS", "1200")) * 1e9


def _io_bytes(s):
    T, H, D = s["T"], s["H"], s["D"]
    # read 2 partial outputs (T*H*D*2 bf16) + 2 LSEs (T*H*4) + write output + write output_lse
    return 2 * (T * H * D * 2) + 2 * (T * H * 4) + (T * H * D * 2) + (T * H * 4)


def _build(s):
    T, H, D = s["T"], s["H"], s["D"]
    output = torch.empty(T, H, D, device=_DEVICE, dtype=torch.bfloat16)
    output_lse = torch.empty(T, H, device=_DEVICE, dtype=torch.float32)
    prefix_output = torch.randn(T, H, D, device=_DEVICE, dtype=torch.bfloat16)
    prefix_lse = torch.randn(T, H, device=_DEVICE, dtype=torch.float32)
    suffix_output = torch.randn(T, H, D, device=_DEVICE, dtype=torch.bfloat16)
    suffix_lse = torch.randn(T, H, device=_DEVICE, dtype=torch.float32)
    return output, output_lse, prefix_output, prefix_lse, suffix_output, suffix_lse


def _op_upstream(t):
    output, output_lse, p_out, p_lse, s_out, s_lse = t
    torch.ops._C.merge_attn_states(output, output_lse, p_out, p_lse, s_out, s_lse, None, None)


def _op_musa(t):
    output, output_lse, p_out, p_lse, s_out, s_lse = t
    torch.ops._C_musa_ops.musa_merge_attn_states(output, output_lse, p_out, p_lse, s_out, s_lse)


_op = _op_upstream  # legacy alias


def _bench(name, s, op, nw=20, ni=100):
    t = _build(s)
    for _ in range(nw): op(t)
    _sync()
    durs = []
    for _ in range(ni):
        _sync(); t0 = time.perf_counter_ns(); op(t); _sync()
        durs.append((time.perf_counter_ns() - t0) / 1e3)
    med = statistics.median(durs)
    io = _io_bytes(s)
    bps = io / (med * 1e-6) if med > 0 else 0
    pct = bps / _peak() * 100
    print(f"{name:<40s} med_us={med:>8.2f}  GB/s={bps/1e9:>7.1f}  roofline={pct:>6.2f}%")
    return {"name": name, "shape": s, "median_us": med, "achieved_gbps": bps/1e9, "roofline_pct": pct}


SHAPES = [
    {"T": 32,   "H": 32, "D": 128},
    {"T": 512,  "H": 32, "D": 128},
    {"T": 4096, "H": 32, "D": 128},
]


def main():
    have_upstream = getattr(torch.ops._C, "merge_attn_states", None) is not None
    have_musa = (hasattr(torch.ops, "_C_musa_ops") and
                 getattr(torch.ops._C_musa_ops, "musa_merge_attn_states", None) is not None)
    print(f"MUSA-0161 merge_attn_states (peak GDDR6 = {_peak()/1e9:.0f} GB/s)")
    print(f"Available impls: upstream={have_upstream}  musa_port={have_musa}\n")
    results = []
    for s in SHAPES:
        name = f"T{s['T']:>4d}_H{s['H']:>2d}_D{s['D']}"
        if have_upstream:
            try: results.append(_bench(f"upstream_{name}", s, _op_upstream))
            except Exception as e: print(f"upstream_{name}: FAIL ({type(e).__name__}: {str(e)[:120]})")
        if have_musa:
            try: results.append(_bench(f"musa-port_{name}", s, _op_musa))
            except Exception as e: print(f"musa-port_{name}: FAIL ({type(e).__name__}: {str(e)[:120]})")
    out = os.environ.get("MUSA_0161_BENCH_OUT", "/tmp/musa_0161_merge_attn_states.json")
    json.dump(results, open(out, "w"), indent=2, default=str)
    print(f"\nResults JSON: {out}")


if __name__ == "__main__":
    main()
