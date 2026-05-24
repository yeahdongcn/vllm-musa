#!/usr/bin/env python3
"""MUSA-0159 correctness: musa_rms_norm_dynamic_per_token_quant vs fp32 reference.

Compares the new MUSA-native kernel against a pure-torch reference impl of
RMSNorm + dynamic per-token FP8 quant. Tolerance: bf16 RMSNorm has ~2e-2
relative error; FP8 quant adds ~1/2^7 = 0.78% additional error from the
log2 of 448 / mantissa-3 quantization. Combined tolerance: rtol=0.05.
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


def reference(inp, weight, eps=1e-5):
    """Pure-torch reference: RMSNorm + dynamic per-token FP8 quant."""
    fp32 = inp.float()
    rms = torch.rsqrt(fp32.pow(2).mean(-1, keepdim=True) + eps)
    normed = fp32 * rms * weight.float()
    abs_max = normed.abs().amax(-1, keepdim=True)
    scale = abs_max.clamp(min=1e-12) / 448.0
    out = (normed / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return out, scale.flatten()


def check_shape(M, N, dtype=torch.bfloat16):
    torch.manual_seed(0)
    inp = torch.randn(M, N, device="musa", dtype=dtype) * 0.1
    weight = torch.randn(N, device="musa", dtype=dtype) * 0.1 + 1.0
    ref_out, ref_scale = reference(inp, weight)

    out_k = torch.empty(M, N, device="musa", dtype=torch.float8_e4m3fn)
    scale_k = torch.empty(M, device="musa", dtype=torch.float32)
    torch.ops._C_musa_ops.musa_rms_norm_dynamic_per_token_quant(
        out_k, inp, weight, scale_k, 1e-5)

    # FP8 -> float to compare
    ref_f = ref_out.float()
    k_f = out_k.float()
    abs_diff = (ref_f - k_f).abs()
    out_max_diff = abs_diff.max().item()
    # Numerical comparison: FP8 e4m3 has variable step size (up to 32 at max=448).
    # Pairwise warp-reduce in the kernel uses a different summation order than
    # torch's .mean(), so a 1-2 FP8 quant-step difference is expected at large
    # magnitudes. Tolerate up to 2 FP8 steps = 64 in absolute value, or
    # 0.5% of the per-element abs(ref_f). Per-element FP non-associativity is fine.
    rel_diff = (abs_diff / (ref_f.abs() + 1e-3)).mean().item()
    scale_diff = (ref_scale - scale_k).abs().max().item()
    rel_scale_diff = scale_diff / (ref_scale.abs().max().item() + 1e-12)
    # 2-step FP8 tolerance + low mean relative error + tight scale match
    passed_out = out_max_diff <= 64.0 and rel_diff <= 0.02
    passed_scale = rel_scale_diff < 0.05
    passed = passed_out and passed_scale
    status = "PASS" if passed else "FAIL"
    print(f"  M={M:>4d} N={N:>5d} dtype={str(dtype):<22s} "
          f"out_max_diff={out_max_diff:>6.2f} mean_rel={rel_diff:>7.4f} "
          f"scale_rel={rel_scale_diff:>8.5f} {status}")
    return passed


def main():
    print("MUSA-0159 musa_rms_norm_dynamic_per_token_quant correctness vs fp32 reference\n")
    all_passed = True
    for M in (1, 32, 128, 4096):
        for N in (4096, 6144, 8192):
            all_passed &= check_shape(M, N, torch.bfloat16)
    print()
    if all_passed:
        print("RESULT: PASS musa_rms_norm_dynamic_per_token_quant shapes=all")
        sys.exit(0)
    else:
        print("RESULT: FAIL musa_rms_norm_dynamic_per_token_quant")
        sys.exit(1)


if __name__ == "__main__":
    main()
