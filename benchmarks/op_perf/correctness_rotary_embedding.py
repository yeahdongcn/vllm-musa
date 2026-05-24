#!/usr/bin/env python3
"""MUSA-0154 correctness: musa_rotary_embedding vs upstream torch.ops._C.rotary_embedding."""
import os
import sys

import torch
import torchada  # noqa: F401
import torch_musa  # noqa: F401

for _so in (
    "/ws/vllm_musa/_C.cpython-310-x86_64-linux-gnu.so",
    "/ws/vllm/_C.cpython-310-x86_64-linux-gnu.so",
):
    if os.path.exists(_so):
        torch.ops.load_library(_so)


def check_shape(T, H, D, is_neox=True, atol=1e-2, rtol=1e-2):
    torch.manual_seed(0)
    positions = torch.arange(T, device="musa", dtype=torch.int64)
    q_ref = torch.randn(T, H * D, device="musa", dtype=torch.bfloat16)
    k_ref = torch.randn(T, H * D, device="musa", dtype=torch.bfloat16)
    cs = torch.randn(T, D, device="musa", dtype=torch.bfloat16)

    q_a = q_ref.clone(); k_a = k_ref.clone()
    q_b = q_ref.clone(); k_b = k_ref.clone()

    musa_op = getattr(torch.ops._C_musa_ops, "musa_rotary_embedding", None)
    upstream_op = getattr(torch.ops._C, "rotary_embedding", None)
    if musa_op is None:
        print("SKIP: musa_rotary_embedding not registered")
        sys.exit(2)
    if upstream_op is None:
        print("SKIP: upstream rotary_embedding not registered")
        sys.exit(2)

    musa_op(positions, q_a, k_a, D, cs, is_neox)
    upstream_op(positions, q_b, k_b, D, cs, is_neox)

    q_diff = (q_a.float() - q_b.float()).abs().max().item()
    k_diff = (k_a.float() - k_b.float()).abs().max().item()
    passed = (torch.allclose(q_a, q_b, atol=atol, rtol=rtol)
              and torch.allclose(k_a, k_b, atol=atol, rtol=rtol))
    status = "PASS" if passed else "FAIL"
    print(f"  T={T:>4d} H={H:>3d} D={D:>3d} is_neox={is_neox} "
          f"q_max_abs={q_diff:.5g} k_max_abs={k_diff:.5g} {status}")
    return passed


def main():
    all_passed = True
    for T in (1, 6, 32, 128, 512, 2048):
        all_passed &= check_shape(T, H=32, D=128, is_neox=True)
    all_passed &= check_shape(T=256, H=64, D=128, is_neox=True)
    if all_passed:
        print("RESULT: PASS musa_rotary_embedding shapes=all")
        sys.exit(0)
    else:
        print("RESULT: FAIL musa_rotary_embedding")
        sys.exit(1)


if __name__ == "__main__":
    main()
