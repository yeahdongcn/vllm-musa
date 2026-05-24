#!/usr/bin/env python3
"""MUSA-0156 correctness: musa_rms_norm_static_fp8_quant vs fp32 reference."""
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


def reference(inp, weight, scale, eps=1e-5):
    fp32 = inp.float()
    rms = torch.rsqrt(fp32.pow(2).mean(-1, keepdim=True) + eps)
    normed = fp32 * rms * weight.float()
    scaled = normed / scale
    out = scaled.clamp(-448, 448).to(torch.float8_e4m3fn)
    return out


def check_shape(M, N, dtype=torch.bfloat16):
    torch.manual_seed(0)
    inp = torch.randn(M, N, device="musa", dtype=dtype) * 0.1
    weight = torch.randn(N, device="musa", dtype=dtype) * 0.1 + 1.0
    scale = torch.tensor(0.01, device="musa", dtype=torch.float32)
    ref = reference(inp, weight, scale)

    out_k = torch.empty(M, N, device="musa", dtype=torch.float8_e4m3fn)
    torch.ops._C_musa_ops.musa_rms_norm_static_fp8_quant(
        out_k, inp, weight, scale, 1e-5)

    ref_f = ref.float()
    k_f = out_k.float()
    abs_diff = (ref_f - k_f).abs()
    out_max_diff = abs_diff.max().item()
    rel_diff = (abs_diff / (ref_f.abs() + 1e-3)).mean().item()
    passed = out_max_diff <= 64.0 and rel_diff <= 0.02
    status = "PASS" if passed else "FAIL"
    print(f"  M={M:>4d} N={N:>5d} dtype={str(dtype):<22s} "
          f"out_max_diff={out_max_diff:>6.2f} mean_rel={rel_diff:>7.4f} {status}")
    return passed


def main():
    print("MUSA-0156 musa_rms_norm_static_fp8_quant correctness\n")
    ok = True
    for M in (1, 32, 128, 4096):
        for N in (4096, 6144, 8192):
            ok &= check_shape(M, N, torch.bfloat16)
    print()
    if ok:
        print("RESULT: PASS musa_rms_norm_static_fp8_quant shapes=all"); sys.exit(0)
    else:
        print("RESULT: FAIL"); sys.exit(1)


if __name__ == "__main__":
    main()
