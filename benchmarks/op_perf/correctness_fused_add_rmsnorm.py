#!/usr/bin/env python3
"""MUSA-0150 correctness: musa_fused_add_rms_norm vs a fp32 torch reference.

Tolerance for bf16/fp16 accumulation: rtol=2e-2, atol=2e-3 (matches the
sphere-kb fused_add_rmsnorm probe's accuracy budget).

Usage:
  python3 benchmarks/op_perf/correctness_fused_add_rmsnorm.py
"""
import sys
import torch
import torchada  # noqa: F401
import torch_musa  # noqa: F401  -- registers MUSA backend so torch.ops.load_library can resolve MUSA symbols
# Load torch.ops directly from the .so files. Avoids the
# vllm-plugin circular-import error that fires when `import vllm`
# triggers `vllm.plugins.load_plugins_by_group()` against a
# partially-initialized vllm_musa module.
import os as _os
for _so in (
    "/ws/vllm_musa/_C.cpython-310-x86_64-linux-gnu.so",
    "/ws/vllm/_C.cpython-310-x86_64-linux-gnu.so",
):
    if _os.path.exists(_so):
        torch.ops.load_library(_so)
del _os


def reference(input_, residual, weight, eps):
    """Pure-torch fp32 reference: x = input + residual; norm = x * rsqrt(mean(x^2) + eps) * weight."""
    x = (input_.float() + residual.float())
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps) * weight.float()
    return normed.to(input_.dtype), x.to(residual.dtype)


def check_shape(M, N, dtype, atol=2e-3, rtol=2e-2):
    torch.manual_seed(0)
    inp_ref = torch.randn(M, N, device="musa", dtype=dtype) * 0.1
    res_ref = torch.randn(M, N, device="musa", dtype=dtype) * 0.1
    w = torch.randn(N, device="musa", dtype=dtype) * 0.1 + 1.0
    eps = 1e-5

    # Reference (out-of-place)
    ref_normed, ref_sum = reference(inp_ref, res_ref, w, eps)

    # Kernel (in-place: input -> normed; residual -> sum)
    inp_k = inp_ref.clone()
    res_k = res_ref.clone()
    torch.ops._C_musa_ops.musa_fused_add_rms_norm(inp_k, res_k, w, eps)

    max_abs_normed = (ref_normed.float() - inp_k.float()).abs().max().item()
    max_abs_sum = (ref_sum.float() - res_k.float()).abs().max().item()
    passed = (torch.allclose(ref_normed, inp_k, atol=atol, rtol=rtol)
              and torch.allclose(ref_sum, res_k, atol=atol, rtol=rtol))
    status = "PASS" if passed else "FAIL"
    print(f"  M={M:>4d} N={N:>5d} {str(dtype):<20s} "
          f"max_abs_normed={max_abs_normed:.5g} max_abs_sum={max_abs_sum:.5g} {status}")
    return passed


def main():
    print("MUSA-0150 fused_add_rms_norm correctness vs fp32 torch reference")
    print("Tolerance: rtol=2e-2, atol=2e-3")
    print()
    all_passed = True
    for M in (1, 6, 32, 128, 512, 4096):
        for N in (4096, 6144, 8192):
            all_passed &= check_shape(M, N, torch.bfloat16)
    print()
    if all_passed:
        print("RESULT: PASS musa_fused_add_rms_norm shapes=all")
        sys.exit(0)
    else:
        print("RESULT: FAIL musa_fused_add_rms_norm")
        sys.exit(1)


if __name__ == "__main__":
    main()
