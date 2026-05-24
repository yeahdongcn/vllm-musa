#!/usr/bin/env python3
"""MUSA-0151 correctness: musa_reshape_and_cache_flash_nhd vs pure-torch reference.

Bit-exact for the in-place gather/scatter (no math, just copy).
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


def reference(key, value, key_cache, value_cache, slot_mapping):
    """Per-token gather + scatter into the paged kv-cache."""
    T = key.shape[0]
    block_size = key_cache.shape[1]
    for t in range(T):
        slot = int(slot_mapping[t].item())
        if slot < 0:
            continue
        block_idx = slot // block_size
        block_offset = slot - block_idx * block_size
        key_cache[block_idx, block_offset] = key[t]
        value_cache[block_idx, block_offset] = value[t]


def check_shape(T, H, D, B, NB, dtype=torch.bfloat16):
    torch.manual_seed(0)
    key = torch.randn(T, H, D, device="musa", dtype=dtype)
    value = torch.randn(T, H, D, device="musa", dtype=dtype)
    kc_ref = torch.zeros(NB, B, H, D, device="musa", dtype=dtype)
    vc_ref = torch.zeros(NB, B, H, D, device="musa", dtype=dtype)
    kc_k = torch.zeros(NB, B, H, D, device="musa", dtype=dtype)
    vc_k = torch.zeros(NB, B, H, D, device="musa", dtype=dtype)
    # Slot mapping: sequential starting at offset 17 (test non-zero start).
    slot_mapping = torch.arange(T, device="musa", dtype=torch.int64) + 17

    reference(key, value, kc_ref, vc_ref, slot_mapping)
    torch.ops._C_musa_ops.musa_reshape_and_cache_flash_nhd(
        key, value, kc_k, vc_k, slot_mapping)

    k_ok = torch.equal(kc_ref, kc_k)
    v_ok = torch.equal(vc_ref, vc_k)
    status = "PASS" if (k_ok and v_ok) else "FAIL"
    print(f"  T={T:>4d} H={H:>2d} D={D:>3d} B={B:>2d} NB={NB:>4d} "
          f"k_bitexact={k_ok} v_bitexact={v_ok} {status}")
    return k_ok and v_ok


def main():
    print("MUSA-0151 reshape_and_cache_flash_nhd correctness vs pure-torch reference")
    print("Bit-exact (pure copy, no math)")
    print()
    all_passed = True
    for T in (1, 6, 64, 256, 1024):
        all_passed &= check_shape(T, H=8, D=128, B=16, NB=4096)
    # Alt config
    all_passed &= check_shape(T=128, H=32, D=128, B=16, NB=4096)
    print()
    if all_passed:
        print("RESULT: PASS musa_reshape_and_cache_flash_nhd shapes=all")
        sys.exit(0)
    else:
        print("RESULT: FAIL musa_reshape_and_cache_flash_nhd")
        sys.exit(1)


if __name__ == "__main__":
    main()
