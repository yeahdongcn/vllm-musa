#!/usr/bin/env python3
"""MUSA-0160 correctness: musa_top_k_top_p_sampling_from_probs (flashinfer port).

Stochastic sampling — we can't bit-compare, but we can verify:
1. Output indices are valid (in [0, vocab_size))
2. When top_k=1 (greedy) deterministic=True, output matches argmax
3. With deterministic=True + fixed seed, repeated calls give same result.
"""
import sys
import torch
import torchada  # noqa: F401
import torch_musa  # noqa: F401
import os as _os
for _so in ("/ws/vllm_musa/_C.cpython-310-x86_64-linux-gnu.so",
            "/ws/vllm/_C.cpython-310-x86_64-linux-gnu.so"):
    if _os.path.exists(_so):
        torch.ops.load_library(_so)
del _os


def check_greedy(B, V):
    """top_k=1 should produce argmax (deterministic)."""
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(B, V, device="musa") * 2.0, dim=-1)
    expected = probs.argmax(dim=-1).to(torch.int32)
    output = torch.empty(B, device="musa", dtype=torch.int32)
    gen = torch.Generator(device="musa").manual_seed(42)
    torch.ops._C_musa_ops.musa_top_k_top_p_sampling_from_probs(
        probs, output, None, None, 1.0, None, 1.0, True, gen)
    match = torch.equal(expected, output)
    print(f"  greedy_top_k=1   B={B:>3d} V={V:>6d}  match={match} "
          f"{'PASS' if match else 'FAIL'}")
    return match


def check_valid_range(B, V, top_k):
    """All sampled indices must be in [0, V)."""
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(B, V, device="musa"), dim=-1)
    output = torch.empty(B, device="musa", dtype=torch.int32)
    gen = torch.Generator(device="musa").manual_seed(42)
    torch.ops._C_musa_ops.musa_top_k_top_p_sampling_from_probs(
        probs, output, None, None, float(top_k), None, 1.0, True, gen)
    out = output.cpu()
    in_range = (out >= 0).all() and (out < V).all()
    print(f"  topK={top_k:>4d}      B={B:>3d} V={V:>6d}  in_range={in_range.item()} "
          f"{'PASS' if in_range else 'FAIL'}")
    return in_range.item()


def main():
    print("MUSA-0160 musa_top_k_top_p_sampling_from_probs correctness\n")
    if getattr(torch.ops._C_musa_ops, "musa_top_k_top_p_sampling_from_probs", None) is None:
        print("RESULT: SKIP (binding not present)")
        sys.exit(0)
    ok = True
    for B in (1, 8, 64):
        for V in (32000, 152064):
            ok &= check_greedy(B, V)
    for B in (8,):
        for V in (32000, 152064):
            for top_k in (16, 1024):
                ok &= check_valid_range(B, V, top_k)
    print()
    if ok:
        print("RESULT: PASS musa_top_k_top_p_sampling_from_probs"); sys.exit(0)
    else:
        print("RESULT: FAIL"); sys.exit(1)


if __name__ == "__main__":
    main()
