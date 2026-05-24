#!/usr/bin/env python3
"""MUSA-0155 correctness: silu_and_mul_per_token_group_fp8_quant vs fp32 reference.

Computes silu(gate) * up over the gated half of the input, then per-128-group FP8 quant.
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


def reference(inp, group_size=128, fp8_min=-448.0, fp8_max=448.0, eps=1e-12):
    gate, up = inp.chunk(2, dim=-1)
    activated = gate.float() * torch.sigmoid(gate.float()) * up.float()
    M, N = activated.shape
    blocks = activated.view(M, N // group_size, group_size)
    abs_max = blocks.abs().amax(-1, keepdim=True)
    scale = (abs_max / fp8_max).clamp(min=eps)
    quantized = (blocks / scale).clamp(fp8_min, fp8_max).to(torch.float8_e4m3fn)
    return quantized.view(M, N), scale.squeeze(-1)


def check_shape(M, N, G=128, dtype=torch.bfloat16):
    torch.manual_seed(0)
    inp = torch.randn(M, 2 * N, device="musa", dtype=dtype) * 0.1
    ref_out, ref_scale = reference(inp, G)

    out_k = torch.empty(M, N, device="musa", dtype=torch.float8_e4m3fn)
    scale_k = torch.empty(M, N // G, device="musa", dtype=torch.float32)
    torch.ops._C_musa_ops.silu_and_mul_per_token_group_fp8_quant(
        inp, out_k, scale_k, G, 1e-12, -448.0, 448.0)

    ref_f = ref_out.float()
    k_f = out_k.float()
    abs_diff = (ref_f - k_f).abs()
    out_max_diff = abs_diff.max().item()
    rel_diff = (abs_diff / (ref_f.abs() + 1e-3)).mean().item()
    scale_max = ref_scale.abs().max().item() + 1e-12
    scale_diff = (ref_scale - scale_k).abs().max().item()
    rel_scale_diff = scale_diff / scale_max
    passed = out_max_diff <= 64.0 and rel_diff <= 0.05 and rel_scale_diff <= 0.05
    status = "PASS" if passed else "FAIL"
    print(f"  M={M:>4d} N={N:>5d} G={G} dtype={str(dtype):<22s} "
          f"out_max_diff={out_max_diff:>6.2f} mean_rel={rel_diff:>7.4f} "
          f"scale_rel={rel_scale_diff:>8.5f} {status}")
    return passed


def main():
    print("MUSA-0155 silu_and_mul_per_token_group_fp8_quant correctness vs fp32 reference\n")
    ok = True
    for M in (1, 32, 128, 4096):
        for N in (4096, 6144, 8192):
            ok &= check_shape(M, N, 128, torch.bfloat16)
    print()
    if ok:
        print("RESULT: PASS silu_and_mul_per_token_group_fp8_quant shapes=all"); sys.exit(0)
    else:
        print("RESULT: FAIL"); sys.exit(1)


if __name__ == "__main__":
    main()
