#!/usr/bin/env python3
"""MUSA-0157 ULTRA-MINIMAL debug: all-ones input + all-ones weights + identity RoPE.

With these inputs:
- Pre-RMSNorm: x = 1 for all positions
- variance = mean(1*1) = 1
- inv_rms = 1/sqrt(1 + 1e-5) ≈ 0.999995
- Post-RMSNorm: x * inv_rms * w = 1 * 0.999995 * 1 ≈ 0.999995
- After identity RoPE (cos=1, sin=0): unchanged

Expected kernel output: all ~0.999995 (or 1.0 within bf16 precision).

If kernel produces this, my random-data test's bug is somewhere in scale-dependent
math. If kernel produces something else, there's a fundamental setup bug.

We also test single-token, single-head (minimal complexity).
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


def check_all_ones(T=1, NH_Q=1, NH_K=1, NH_V=1, D=128, dtype=torch.bfloat16):
    qkv = torch.ones(T, (NH_Q + NH_K + NH_V) * D, device="musa", dtype=dtype)
    q_w = torch.ones(D, device="musa", dtype=dtype)
    k_w = torch.ones(D, device="musa", dtype=dtype)
    # Identity RoPE: cos=1, sin=0
    cos_sin_cache = torch.zeros(T, D, device="musa", dtype=dtype)
    cos_sin_cache[:, :D // 2] = 1.0  # cos = 1
    # second half stays 0 = sin
    position_ids = torch.arange(T, device="musa", dtype=torch.int64)

    qkv_k = qkv.clone()
    # Use MUSA-native binding (avoids upstream's bf16 no-op guard)
    op = getattr(torch.ops._C_musa_ops, "musa_fused_qk_norm_rope", None)
    if op is None:
        # Fallback to upstream (will silently no-op for bf16 on MUSA)
        op = torch.ops._C.fused_qk_norm_rope
    try:
        op(qkv_k, NH_Q, NH_K, NH_V, D, 1e-5,
           q_w, k_w, cos_sin_cache, True,
           position_ids, -1)
    except Exception as e:
        print(f"  CRASH ({type(e).__name__}: {str(e)[:120]})")
        return False

    v = qkv_k.view(T, NH_Q + NH_K + NH_V, D)
    q_out = v[:, :NH_Q].float().flatten()
    k_out = v[:, NH_Q:NH_Q+NH_K].float().flatten()
    v_out = v[:, NH_Q+NH_K:].float().flatten()
    # Expected: 1.0 (within bf16 precision)
    expected_value = 1.0 / (1.0 + 1e-5) ** 0.5  # ≈ 0.999995
    print(f"  T={T} NH_Q={NH_Q} NH_K={NH_K} NH_V={NH_V} D={D}")
    print(f"    expected value: {expected_value:.6f}")
    print(f"    Q output: min={q_out.min().item():.6f} max={q_out.max().item():.6f} "
          f"mean={q_out.mean().item():.6f}")
    print(f"    K output: min={k_out.min().item():.6f} max={k_out.max().item():.6f} "
          f"mean={k_out.mean().item():.6f}")
    print(f"    V output: min={v_out.min().item():.6f} max={v_out.max().item():.6f} "
          f"mean={v_out.mean().item():.6f}")
    # Show first 16 values of Q for inspection
    print(f"    Q first 16 values: {q_out[:16].tolist()}")
    print(f"    K first 16 values: {k_out[:16].tolist()}")
    return True


def check_input_scale(scale, T=1, NH_Q=1, NH_K=1, NH_V=1, D=128, dtype=None):
    if dtype is None:
        dtype = torch.bfloat16
    """Use uniform input value=scale to test that RMSNorm normalizes it to ~1."""
    qkv = torch.full((T, (NH_Q + NH_K + NH_V) * D), scale,
                     device="musa", dtype=dtype)
    q_w = torch.ones(D, device="musa", dtype=dtype)
    k_w = torch.ones(D, device="musa", dtype=dtype)
    cos_sin_cache = torch.zeros(T, D, device="musa", dtype=dtype)
    cos_sin_cache[:, :D // 2] = 1.0
    position_ids = torch.arange(T, device="musa", dtype=torch.int64)

    qkv_k = qkv.clone()
    op = getattr(torch.ops._C_musa_ops, "musa_fused_qk_norm_rope", None)
    if op is None:
        op = torch.ops._C.fused_qk_norm_rope
    op(qkv_k, NH_Q, NH_K, NH_V, D, 1e-5,
       q_w, k_w, cos_sin_cache, True,
       position_ids, -1)
    v = qkv_k.view(T, NH_Q + NH_K + NH_V, D)
    q_out = v[:, :NH_Q].float().flatten()
    # With input=scale, variance = scale². rms = 1/sqrt(scale² + eps) ≈ 1/scale.
    # Output: scale * (1/scale) * 1 = 1.0. (sign of scale preserved by RoPE-identity)
    expected = scale / abs(scale) if scale != 0 else 0
    expected_norm = expected / (1.0 + 1e-5 / (scale ** 2 if scale != 0 else 1)) ** 0.5
    print(f"  scale={scale:>6.3f}  expected={expected_norm:.6f}  "
          f"Q[0]={q_out[0].item():.6f} K[0]={v[:,NH_Q:NH_Q+NH_K].float().flatten()[0].item():.6f}")


def main():
    print("=" * 70)
    print("MUSA-0157 trivial debug")
    print("=" * 70)
    print()
    print("# BF16 PATH (kernel has #if __CUDA_ARCH__<800 early-return for bf16):")
    print()
    print("Test 1: bf16, T=4, NH_Q=4, NH_K=4, NH_V=4, D=128 — uniform scale=0.5")
    check_input_scale(0.5, dtype=torch.bfloat16)
    print()
    print("Test 2: bf16 — uniform scale=10.0 (expected ~1.0 if RMSNorm runs)")
    check_input_scale(10.0, dtype=torch.bfloat16)
    print()
    print("=" * 70)
    print("# FP16 PATH (kernel guard does NOT apply to fp16):")
    print("=" * 70)
    print()
    print("Test 3: fp16 — uniform scale=0.5 (expected ~1.0)")
    check_input_scale(0.5, dtype=torch.float16)
    print()
    print("Test 4: fp16 — uniform scale=10.0 (expected ~1.0)")
    check_input_scale(10.0, dtype=torch.float16)
    print()
    print("Test 5: fp16 — uniform scale=2.0 (expected ~1.0)")
    check_input_scale(2.0, dtype=torch.float16)


if __name__ == "__main__":
    main()
