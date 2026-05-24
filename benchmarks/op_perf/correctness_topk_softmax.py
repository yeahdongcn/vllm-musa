#!/usr/bin/env python3
"""MUSA-0162 correctness: musa_topk_softmax vs torch reference.

Comparing on (topk_weights, topk_indices). For ties on softmax values we
allow either ordering. The kernel emits indices in (value, lower-index-first)
order which matches torch.topk's tiebreak.
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


def reference(gating, K, renormalize=True):
    """Softmax + top-K + optional renormalize."""
    softmax = torch.softmax(gating, dim=-1)
    weights, indices = torch.topk(softmax, K, dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights, indices.to(torch.int32)


def check_shape(T, E, K):
    torch.manual_seed(0)
    gating = torch.randn(T, E, device="musa", dtype=torch.float32)
    ref_w, ref_i = reference(gating, K, renormalize=True)

    w_k = torch.empty(T, K, device="musa", dtype=torch.float32)
    i_k = torch.empty(T, K, device="musa", dtype=torch.int32)
    te_k = torch.empty(T, K, device="musa", dtype=torch.int32)
    torch.ops._C_musa_ops.musa_topk_softmax(w_k, i_k, te_k, gating, True)

    # Sort both by indices for comparison (since both should pick the same K
    # experts but maybe in different sort order)
    ref_sorted_i, ref_perm = torch.sort(ref_i, dim=-1)
    ref_sorted_w = torch.gather(ref_w, -1, ref_perm)
    k_sorted_i, k_perm = torch.sort(i_k, dim=-1)
    k_sorted_w = torch.gather(w_k, -1, k_perm)

    idx_match = torch.equal(ref_sorted_i, k_sorted_i)
    w_max_diff = (ref_sorted_w - k_sorted_w).abs().max().item()
    # Softmax weights sum to 1 (renormalized); diff tolerance 1e-3 (FP32 rounding + exp diffs)
    passed = idx_match and w_max_diff <= 1e-3
    status = "PASS" if passed else "FAIL"
    print(f"  T={T:>4d} E={E:>3d} K={K} "
          f"indices_match={idx_match} w_max_diff={w_max_diff:>8.5f} {status}")
    return passed


def main():
    print("MUSA-0162 musa_topk_softmax correctness\n")
    ok = True
    for T in (32, 512, 4096):
        ok &= check_shape(T, 128, 8)
    print()
    if ok:
        print("RESULT: PASS musa_topk_softmax shapes=all"); sys.exit(0)
    else:
        print("RESULT: FAIL"); sys.exit(1)


if __name__ == "__main__":
    main()
