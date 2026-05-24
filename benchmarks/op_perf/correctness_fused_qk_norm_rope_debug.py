#!/usr/bin/env python3
"""MUSA-0157 DEBUG: isolate the source of correctness FAIL.

Run with cos=1, sin=0 — RoPE becomes identity, so kernel output should equal
RMSNorm(input). If still mismatched, it's RMSNorm or layout. If matches, RoPE is fine.
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


def rms_norm(x, weight, eps):
    fp32 = x.float()
    rms = torch.rsqrt(fp32.pow(2).mean(-1, keepdim=True) + eps)
    return (fp32 * rms * weight.float()).to(x.dtype)


def reference_identity_rope(qkv, nhq, nhk, nhv, D, eps, q_w, k_w):
    """Reference with identity RoPE (cos=1, sin=0) — just RMSNorm Q and K."""
    T = qkv.shape[0]
    v = qkv.view(T, nhq + nhk + nhv, D)
    q = v[:, :nhq, :].contiguous()
    k = v[:, nhq:nhq + nhk, :].contiguous()
    vv = v[:, nhq + nhk:, :].contiguous()
    q = rms_norm(q, q_w, eps)
    k = rms_norm(k, k_w, eps)
    return torch.cat([q.view(T, -1), k.view(T, -1), vv.view(T, -1)], dim=-1)


def check(T=32, NH_Q=32, NH_K=8, NH_V=8, D=128, dtype=torch.bfloat16):
    torch.manual_seed(0)
    qkv = torch.randn(T, (NH_Q + NH_K + NH_V) * D, device="musa", dtype=dtype) * 0.05
    q_w = (torch.randn(D, device="musa", dtype=dtype) * 0.05 + 1.0).contiguous()
    k_w = (torch.randn(D, device="musa", dtype=dtype) * 0.05 + 1.0).contiguous()
    # IDENTITY ROPE: cos=1, sin=0
    cos_sin_cache = torch.zeros(T, D, device="musa", dtype=dtype)
    cos_sin_cache[:, :D // 2] = 1.0  # cos block = 1
    # sin block stays 0
    position_ids = torch.arange(T, device="musa", dtype=torch.int64)

    ref_out = reference_identity_rope(qkv.clone(), NH_Q, NH_K, NH_V, D, 1e-5, q_w, k_w)
    qkv_k = qkv.clone()
    op = getattr(torch.ops._C_musa_ops, "musa_fused_qk_norm_rope", None)
    if op is None:
        op = torch.ops._C.fused_qk_norm_rope
    try:
        op(qkv_k, NH_Q, NH_K, NH_V, D, 1e-5,
           q_w, k_w, cos_sin_cache, True,
           position_ids, -1)
    except Exception as e:
        print(f"  CRASH ({type(e).__name__}: {str(e)[:80]})")
        return False

    v_ref = ref_out.view(T, NH_Q + NH_K + NH_V, D)
    v_k = qkv_k.view(T, NH_Q + NH_K + NH_V, D)
    q_diff = (v_ref[:, :NH_Q].float() - v_k[:, :NH_Q].float()).abs().max().item()
    k_diff = (v_ref[:, NH_Q:NH_Q+NH_K].float() - v_k[:, NH_Q:NH_Q+NH_K].float()).abs().max().item()
    v_diff = (v_ref[:, NH_Q+NH_K:].float() - v_k[:, NH_Q+NH_K:].float()).abs().max().item()
    print(f"T={T}  q_diff={q_diff:.5f}  k_diff={k_diff:.5f}  v_diff={v_diff:.5f}")
    print(f"  Interpretation: q_diff <0.01 → kernel = RMSNorm(input). >0.1 → kernel diverges from RMSNorm too.")
    return q_diff < 0.01 and k_diff < 0.01


def main():
    print("MUSA-0157 DEBUG: identity RoPE (cos=1, sin=0). If kernel ≈ RMSNorm,")
    print("RoPE math is the issue. If kernel ≠ RMSNorm, RMSNorm/layout is the issue.\n")
    ok = check()
    print()
    print("PASS" if ok else "FAIL — kernel disagrees with RMSNorm-only path")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
