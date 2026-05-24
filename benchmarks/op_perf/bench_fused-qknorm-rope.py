#!/usr/bin/env python3
"""MUSA-0157 bench: fused_qk_norm_rope (csrc/fused_qknorm_rope_kernel.cu).

Note: MUSA-0105 documented that this kernel triggers a mcc 'h' (16-bit)
register-allocation crash for bf16 inline asm on MUSA. This bench WILL FAIL
or CRASH on MUSA until that compiler blocker is resolved. Captured here so
the baseline is documented and re-running becomes a one-line check.

Usage:  python3 benchmarks/op_perf/bench_fused-qknorm-rope.py
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
    T, NH_Q, NH_K, NH_V, D, R = s["T"], s["NH_Q"], s["NH_K"], s["NH_V"], s["D"], s["R"]
    # qkv read+write (bf16) + q/k weight read + cos_sin read + position_ids read
    qkv_elem = (NH_Q + NH_K + NH_V) * D
    return 2 * T * qkv_elem * 2 + (NH_Q + NH_K) * D * 2 + T * R * 2 + T * 8


def _build(s):
    T, NH_Q, NH_K, NH_V, D, R = s["T"], s["NH_Q"], s["NH_K"], s["NH_V"], s["D"], s["R"]
    qkv = torch.randn(T, (NH_Q + NH_K + NH_V) * D, device=_DEVICE, dtype=torch.bfloat16)
    q_weight = torch.randn(D, device=_DEVICE, dtype=torch.bfloat16) * 0.1 + 1.0
    k_weight = torch.randn(D, device=_DEVICE, dtype=torch.bfloat16) * 0.1 + 1.0
    cos_sin_cache = torch.randn(T, R, device=_DEVICE, dtype=torch.bfloat16)
    position_ids = torch.arange(T, device=_DEVICE, dtype=torch.int64)
    return qkv, NH_Q, NH_K, NH_V, D, q_weight, k_weight, cos_sin_cache, position_ids


def _op(t):
    qkv, nhq, nhk, nhv, D, q_w, k_w, cs, pos = t
    # Prefer MUSA-native (upstream is bf16 no-op on MUSA — would give meaningless % roofline)
    op = getattr(torch.ops._C_musa_ops, "musa_fused_qk_norm_rope", None)
    if op is None:
        op = torch.ops._C.fused_qk_norm_rope
    op(qkv, nhq, nhk, nhv, D, 1e-5, q_w, k_w, cs, True, pos, -1)


def _bench(name, s, nw=10, ni=50):
    t = _build(s)
    for _ in range(nw): _op(t)
    _sync()
    durs = []
    for _ in range(ni):
        _sync(); t0 = time.perf_counter_ns(); _op(t); _sync()
        durs.append((time.perf_counter_ns() - t0) / 1e3)
    med = statistics.median(durs)
    io = _io_bytes(s)
    bps = io / (med * 1e-6) if med > 0 else 0
    pct = bps / _peak() * 100
    print(f"{name:<32s} med_us={med:>8.2f}  GB/s={bps/1e9:>7.1f}  roofline={pct:>6.2f}%")
    return {"name": name, "shape": s, "median_us": med, "achieved_gbps": bps/1e9, "roofline_pct": pct}


SHAPES = [
    {"T": 32,   "NH_Q": 32, "NH_K": 8, "NH_V": 8, "D": 128, "R": 128},
    {"T": 512,  "NH_Q": 32, "NH_K": 8, "NH_V": 8, "D": 128, "R": 128},
    {"T": 4096, "NH_Q": 32, "NH_K": 8, "NH_V": 8, "D": 128, "R": 128},
]


def main():
    if getattr(torch.ops._C, "fused_qk_norm_rope", None) is None:
        print("fused_qk_norm_rope binding not available — SKIP")
        return
    print(f"MUSA-0157 fused_qk_norm_rope baseline (NOTE: MUSA-0105 bf16 mcc blocker may CRASH)\n")
    results = []
    for s in SHAPES:
        name = f"T{s['T']:>4d}_QKV{s['NH_Q']}_{s['NH_K']}_{s['NH_V']}_D{s['D']}"
        try: results.append(_bench(name, s))
        except Exception as e: print(f"{name}: FAIL ({type(e).__name__}: {str(e)[:120]})")
    out = os.environ.get("MUSA_0157_BENCH_OUT", "/tmp/musa_0157_fused_qknorm_rope.json")
    json.dump(results, open(out, "w"), indent=2, default=str)
    print(f"\nResults JSON: {out}")


if __name__ == "__main__":
    main()
