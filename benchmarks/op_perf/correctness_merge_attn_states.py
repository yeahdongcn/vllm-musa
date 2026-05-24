#!/usr/bin/env python3
"""MUSA-0161 correctness: musa_merge_attn_states vs fp32 reference (LSE-weighted merge)."""
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


def reference(p_out, p_lse, s_out, s_lse):
    """LSE-weighted merge of two partial attention outputs (arxiv 2501.01005 §2.2)."""
    pm = torch.maximum(p_lse, s_lse)
    ps = torch.exp(p_lse - pm)
    ss = torch.exp(s_lse - pm)
    z = ps + ss
    p_norm = (ps / z).unsqueeze(-1)
    s_norm = (ss / z).unsqueeze(-1)
    out = (p_out.float() * p_norm + s_out.float() * s_norm).to(p_out.dtype)
    out_lse = torch.log(z) + pm
    return out, out_lse


def check_shape(T, H, D, dtype=torch.bfloat16):
    torch.manual_seed(0)
    p_out = torch.randn(T, H, D, device="musa", dtype=dtype)
    p_lse = torch.randn(T, H, device="musa", dtype=torch.float32)
    s_out = torch.randn(T, H, D, device="musa", dtype=dtype)
    s_lse = torch.randn(T, H, device="musa", dtype=torch.float32)
    ref_out, ref_lse = reference(p_out, p_lse, s_out, s_lse)

    out_k = torch.empty(T, H, D, device="musa", dtype=dtype)
    lse_k = torch.empty(T, H, device="musa", dtype=torch.float32)
    torch.ops._C_musa_ops.musa_merge_attn_states(
        out_k, lse_k, p_out, p_lse, s_out, s_lse)

    out_max_diff = (ref_out.float() - out_k.float()).abs().max().item()
    lse_max_diff = (ref_lse - lse_k).abs().max().item()
    # bf16 mantissa + exp arithmetic noise — generous tolerance
    passed = out_max_diff <= 0.05 and lse_max_diff <= 0.01
    status = "PASS" if passed else "FAIL"
    print(f"  T={T:>4d} H={H:>2d} D={D:>3d} dtype={str(dtype):<22s} "
          f"out_max_diff={out_max_diff:>8.5f} lse_max_diff={lse_max_diff:>8.5f} {status}")
    return passed


def main():
    print("MUSA-0161 musa_merge_attn_states correctness\n")
    ok = True
    for T in (32, 512, 4096):
        ok &= check_shape(T, 32, 128, torch.bfloat16)
    print()
    if ok:
        print("RESULT: PASS musa_merge_attn_states shapes=all"); sys.exit(0)
    else:
        print("RESULT: FAIL"); sys.exit(1)


if __name__ == "__main__":
    main()
