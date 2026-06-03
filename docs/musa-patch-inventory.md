# vLLM-MUSA patch inventory (MUSA-0300)

> **Deliverable for ticket MUSA-0300.** This is the base document for the
> MUSA-0300→0309 patching-maintenance roadmap. It classifies every mechanism
> by which `vllm-musa` modifies upstream vLLM, so later tickets can migrate,
> version-gate, report, and shrink the patch surface.

| | |
|---|---|
| Generated | 2026-06-03 |
| vllm-musa branch / SHA | `v0.22.0-dev` / `aad1aca59` |
| Upstream vLLM pin (`setup.py`) | `v0.22.0` (FlashInfer pinned to `bc29697b…`) |
| Resolution base | pristine `vllm/` clone at tag `v0.22.0` (0 dirty lines) |
| Runtime patch files | **51** `*.patch.py` |
| Native build patches | 4 file overrides + **19** text-patch target files |
| Object-patch helpers | 6 in `vllm_musa/__init__.py` |

**Caveat on "target resolves":** resolution is checked against a *pristine
v0.22.0 clone*. A patch marked `private/absent` does **not** mean it is dead —
its target module may be provided at runtime by a private checkpoint's
remote-code, a vendored tree, or a path that only exists in the MUSA container.
It means the target is **not** in upstream v0.22.0 and therefore each such
patch needs per-patch confirmation (this is the single highest-value follow-up
the inventory surfaces — see §5).

---

## 1. The four patch mechanisms

`vllm-musa` modifies vLLM through **four** distinct mechanisms, not one. The
original MUSA-0300 ticket named three; the `vllm_musa/__init__.py` object-patch
family is the fourth and is the *reference pattern* the migration tickets
(MUSA-0302/0304) want to move other patches toward.

| # | Mechanism | Where | Phase | Process scope | Persistence |
|---|---|---|---|---|---|
| 1 | **Runtime source transform** (disk rewrite) | `vllm_musa/patches/*.patch.py` via `apply_patches()` | plugin load (`_register_patches`) | per-process, **writes installed vLLM `.py` on disk** (atomic tempfile+rename) | **persists across sessions & processes** |
| 2 | **Import side-effect monkey-patch** | `*.patch.py` with empty `PATCHES` | same loader, executed as import side-effect | in-memory, per-process | process-local |
| 3 | **Build-time native patch** | `setup.py` `CSRC_FILE_OVERRIDES` + `CSRC_TEXT_PATCHES` | `pip install` / build | source tree before compile | until next clean checkout |
| 4 | **Object monkey-patch (good pattern)** | `vllm_musa/__init__.py` `_patch_*` helpers | plugin load (`_register_patches`) | in-memory, idempotent via `_musa_*` sentinel attr | process-local |

**Mechanism 1 is the central maintenance liability** (MUSA-0303): it mutates
the *installed* vLLM package on disk, so environment state drifts, upgrades
race, and `apply_patches()` carries explicit hazard-mitigation for
multiprocess spawn (the atomic-rename comment at `patches/__init__.py:221-252`)
and for re-apply accumulation (the `new not in source` gate, MUSA-0089/0096).

---

## 2. Runtime Python patches (`vllm_musa/patches/`)

51 files. `kind`: **src** = non-empty `PATCHES` (mechanism 1, disk rewrite);
**side** = empty `PATCHES`, monkey-patch on import (mechanism 2). `norm` =
defines a `normalize_source()` idempotency hook. `tgt` = resolves in pristine
v0.22.0 (`up`) or not (`PRIV`). `L` = file lines.

### A — Compilation / Inductor / Triton compat (upstream)

| Module (`vllm.*`) | kind | norm | tgt | L | Purpose |
|---|---|---|---|---|---|
| `compilation.backends` | src | | up | 37 | accept backend kw-options on this snapshot |
| `compilation.caching` | src | | up | 55 | MUSA compile-cache compat |
| `compilation.compiler_interface` | src | | up | 25 | functorch config compat |
| `compilation.passes.pass_manager` | src | | up | 39 | MUSA pass-manager predicates |
| `compilation.piecewise_backend` | src | | up | 25 | piecewise backend compat |
| `triton_utils.jit_monitor` | src | | up | 15 | v0.22 JIT-monitor Triton compat |

### B — Distributed / communication (upstream)

| Module (`vllm.*`) | kind | norm | tgt | L | Purpose |
|---|---|---|---|---|---|
| `distributed.device_communicators.all2all` | src | | up | 12 | drop `explicitly_destroy` arg |
| `distributed.device_communicators.cuda_communicator` | src | | up | 61 | MUSA-0088 qr_comm gate extension |
| `distributed.device_communicators.custom_all_reduce` | src | | up | 46 | 8MB→128MB max_size; enable MUSA CAR |
| `distributed.parallel_state` | **side** | | up | 202 | MUSA-0124 Eagle3 draft at TP=1 (replicated) |

### C — Quantization / MoE / GEMM / linear (upstream)

| Module (`vllm.*`) | kind | norm | tgt | L | Purpose |
|---|---|---|---|---|---|
| `model_executor.kernels.linear` | src | | up | 131 | MUSA linear-kernel selection |
| `model_executor.layers.mhc` | src | norm | up | 250 | mHC blocks → MUSA native/JIT |
| `model_executor.layers.fused_moe.activation` | src | | up | 15 | v0.22 MoE activation helper compat |
| `model_executor.layers.fused_moe.deep_gemm_moe` | src | | **PRIV** | 22 | **DEAD — moved to `experts/` in v0.22 (see §5)** |
| `model_executor.layers.fused_moe.experts.deep_gemm_moe` | src | | up | 75 | `a2q_scale.contiguous()` for MUSA |
| `model_executor.layers.quantization.fp8` | src | | up | 18 | MUSA no-Marlin; cap 75→31 |
| `model_executor.layers.quantization.utils.fp8_utils` | src | | up | 40 | per-token quant contiguous on MUSA |
| `utils.deep_gemm` | src | | up | 59 | enable DeepGemm on MUSA; cap 90→31 |

### D — Attention / MLA / FlashMLA / sparse (upstream)

| Module (`vllm.*`) | kind | norm | tgt | L | Purpose |
|---|---|---|---|---|---|
| `model_executor.layers.attention.attention` | src | | up | 14 | attention compile compat |
| `model_executor.layers.sparse_attn_indexer` | src | norm | up | 998 | opt-in MUSA sparse-indexer correctness fallback (helpers injected as source text, NOT monkey-patches — MUSA-0302 audit) |
| `v1.attention.backends.mla.flashmla` | src | | up | 15 | reorder-batch threshold 128→1 |
| `v1.attention.backends.mla.sparse_swa` | src | | up | 89 | DeepSeek-V4 sparse SWA metadata Triton |
| `v1.attention.ops.flashmla` | src | | up | 14 | MUSA capability family |
| `v1.attention.ops.triton_turboquant_decode` | src | | up | 17 | MUSA Triton decode kernel compat |
| `v1.attention.ops.triton_unified_attention` | src | | up | 36 | PEP 526 / `tl.cast` Triton fixes |

### E — Sampler (upstream)

| Module (`vllm.*`) | kind | norm | tgt | L | Purpose |
|---|---|---|---|---|---|
| `v1.sample.ops.topk_topp_triton` | src | | up | 121 | MUSA Triton int32 / pivot refactor |
| `v1.sample.rejection_sampler` | src | | up | 200 | MUSA-safe rejection sampler |

### F — Worker / GPU model runner (upstream)

| Module (`vllm.*`) | kind | norm | tgt | L | Purpose |
|---|---|---|---|---|---|
| `v1.worker.gpu_worker` | src | | up | 30 | accept `musa` device type |
| `v1.worker.gpu_input_batch` | src | | up | 78 | MUSA-0203 (#34880) input-batch hunks |
| `v1.worker.gpu_model_runner` | src | | up | 165 | DeepSeek/MTP runner source patches (former monkey-patch "no longer reachable" per docstring — pure source-transform, MUSA-0302 audit) |
| `v1.worker.gpu.block_table` | src | norm | up | 45 | v0.22 block-table Triton compat |
| `v1.worker.gpu.sample.penalties` | src | | up | 10 | v0.22 worker penalties Triton compat |

### G — Spec-decode backports (upstream) — **highest-risk family (MUSA-0305)**

| Module (`vllm.*`) | kind | norm | tgt | L | Purpose |
|---|---|---|---|---|---|
| `v1.cudagraph_dispatcher` | src | | up | 134 | MUSA-0203 backport of vllm#34880 |
| `v1.spec_decode.llm_base_proposer` | src | norm | up | 435 | MUSA-0203 backport of vllm#34880 |
| `v1.spec_decode.utils` | src | | up | 53 | MUSA-0203 (#34880) kernel safety |
| `v1.spec_decode.eagle` | **side** | | up | 25 | MUSA spec-decode kernel monkey-patch shim |
| `v1.spec_decode.dflash` | src | | up | 144 | MUSA-0400/0402/0403 dflash + draft-loop FULL CUDAGraph |
| `distributed.parallel_state` | **side** | | up | 202 | (listed in B) Eagle3 draft TP=1 |

### H — Profiler (upstream)

| Module (`vllm.*`) | kind | norm | tgt | L | Purpose |
|---|---|---|---|---|---|
| `profiler.wrapper` | src | | up | 18 | add `MUSA` to TorchProfilerActivity literal |

### I — DeepSeek-V4, **new** v0.22 namespace `vllm.models.deepseek_v4.*` (upstream)

| Module (`vllm.*`) | kind | norm | tgt | L | Purpose |
|---|---|---|---|---|---|
| `models.deepseek_v4.attention` | src | | up | 186 | v0.22 DSv4 attention FP8 einsum for MUSA |
| `models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant` | src | | up | 53 | inverse-RoPE FP8 quant for MUSA |
| `models.deepseek_v4.nvidia.flashmla` | src | | up | 18 | DSv4 FlashMLA imports for MUSA |
| `models.deepseek_v4.nvidia.model` | src | | up | 16 | DSv4 NVIDIA model gates for MUSA |

### J — DeepSeek-V4, **old** namespace — **target absent in v0.22 (confirm/retire — §5)**

| Module (`vllm.*`) | kind | norm | tgt | L | Purpose |
|---|---|---|---|---|---|
| `model_executor.layers.deepseek_compressor` | src | | **PRIV** | 1133 | DSv4 compressor/cache MUSA gates |
| `model_executor.layers.deepseek_v4_attention` | src | norm | **PRIV** | 839 | DSv4 attention → MUSA sparse FlashMLA |
| `model_executor.models.deepseek_v4` | src | | **PRIV** | 378 | DSv4 model CUDA-only gates |
| `model_executor.models.deepseek_v4_mtp` | src | | **PRIV** | 71 | DSv4 MTP compat |
| `v1.attention.ops.deepseek_v4_ops.cache_utils` | src | | **PRIV** | 444 | DSv4 cache-util kernels (helpers injected as source text, NOT monkey-patches — MUSA-0302 audit) |
| `v1.attention.ops.deepseek_v4_ops.fused_compress_quant_cache` | src | | **PRIV** | 23 | DSv4 compressor Triton compat |
| `v1.attention.ops.deepseek_v4_ops.fused_indexer_q` | src | | **PRIV** | 282 | DSv4 sparse-indexer Q quant gate |
| `v1.attention.ops.deepseek_v4_ops.fused_inv_rope_fp8_quant` | src | norm | **PRIV** | 294 | DSv4 inverse-RoPE FP8 quant fallback |
| `v1.attention.ops.deepseek_v4_ops.fused_qk_rmsnorm` | src | | **PRIV** | 158 | DSv4 fused Q/K RMSNorm native MUSA |

---

## 3. Native build-time patches (`setup.py`) — mechanism 3 (MUSA-0306)

Applied by `_apply_file_overrides()` and `_apply_text_patches()` against the
cloned upstream csrc tree before compile. Stored as Python data, not reviewable
`.patch` files — this is the MUSA-0306 target.

### `CSRC_FILE_OVERRIDES` (4 — whole-file MUSA replacements from `csrc/`)

- `csrc/custom_all_reduce.cu`
- `csrc/custom_all_reduce.cuh`
- `csrc/mamba/mamba_ssm/selective_scan_fwd.cu`
- `csrc/quantization/activation_kernels.cu`

### `CSRC_TEXT_PATCHES` (19 target files, multiple rules each)

Categories: `USE_ROCM`→`USE_MUSA` guards; `#include` redirects to
`csrc_musa/…` MUSA headers; `cuda*`→`musa*` API renames; FP8/BF16 header swaps
to `c10/util`; MUSA-0203 `paged_attention_v1/v2` impl stripping; DeepSeek-V4
fused-kernel impl stripping.

`csrc/topk.cu` · `csrc/moe/torch_bindings.cpp` · `csrc/torch_bindings.cpp` ·
`csrc/libtorch_stable/torch_bindings.cpp` ·
`csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh` ·
`csrc/attention/merge_attn_states.cu` ·
`csrc/libtorch_stable/quantization/fp4/nvfp4_utils.cuh` ·
`csrc/libtorch_stable/quantization/vectorization.cuh` ·
`csrc/cuda_vec_utils.cuh` · `csrc/cuda_compat.h` ·
`csrc/quantization/w8a8/fp8/common.cuh` ·
`csrc/quantization/fused_kernels/quant_conversions.cuh` ·
`csrc/libtorch_stable/quantization/fused_kernels/quant_conversions.cuh` ·
`csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu` ·
`csrc/moe/moe_align_sum_kernels.cu` · `csrc/attention/attention_kernels.cuh` ·
`csrc/type_convert.cuh` · `csrc/activation_kernels.cu` ·
**`vllm/_custom_ops.py`** ← a *Python* file patched at build time (note: this
is mechanism 3 touching Python, distinct from mechanism 1's runtime rewrite).

---

## 4. Object-patch helpers (`vllm_musa/__init__.py`) — mechanism 4 (reference pattern)

Wired through `_register_patches()` / `_register_ops()`. Each is idempotent via
a `_musa_*` sentinel attribute and uses wrap-and-replace, never disk mutation —
**this is the target shape for MUSA-0302/0304 migrations.**

| Helper | Target object | Purpose |
|---|---|---|
| `_apply_vllm_patches` | → `patches.apply_patches()` | entry to mechanism 1 |
| `_patch_vllm_backend_call_options` | `vllm.compilation.backends.VllmBackend.__call__` | ignore extra backend kwargs |
| `_patch_functorch_config_patch` | `torch._functorch.config.patch` | drop missing functorch keys |
| `_patch_inductor_config_patch` | `torch._inductor.config.patch` | drop missing inductor keys |
| `_patch_vllm_functorch_config` | `vllm.compilation.compiler_interface._get_vllm_functorch_config` | filter functorch config |
| `_patch_musa_batch_defaults` | `vllm.engine.arg_utils.EngineArgs.get_batch_defaults` | keep MUSA on non-H100 scheduler defaults |
| `_patch_vllm_custom_ops_dflash_fallbacks` | dflash custom-op fallbacks | dflash op shims |

---

## 5. Key findings (drive MUSA-0301/0304/0305/0307/0309)

1. **Mechanism 1 still rewrites installed vLLM on disk.** Unchanged from the
   0.20.0 era; `apply_patches()` writes `.py` files in the installed package.
   This is MUSA-0303's whole reason to exist and remains the largest liability.

2. **Patch surface ~doubled in two weeks** (27 on 2026-05-18 → 31 → 43 → **51**
   on 2026-06-02), driven by DeepSeek-V4, sparse-attn, M3, and dflash. Growth
   rate — not absolute count — is the maintenance risk.

3. **Confirmed dead patch:**
   `vllm__model_executor__layers__fused_moe__deep_gemm_moe.patch.py` targets
   `…/fused_moe/deep_gemm_moe.py`, which v0.22 **moved** to
   `…/fused_moe/experts/deep_gemm_moe.py`. The old patch silently skips (no
   target) and a replacement (`…experts__deep_gemm_moe`) already exists. The
   old file is dead weight → retire or version-gate. This is the prototypical
   failure mode MUSA-0301 (skip reporting) and MUSA-0307 (cross-version check)
   must catch automatically.

4. **10 patches target modules absent from pristine v0.22.0** (all DeepSeek-V4
   old-namespace: `model_executor.layers.deepseek_*`,
   `model_executor.models.deepseek_v4*`, `v1.attention.ops.deepseek_v4_ops.*`).
   v0.22 relocated DeepSeek-V4 to `vllm/models/deepseek_v4/`. Each needs
   per-patch confirmation: is its target provided at runtime (private
   checkpoint remote-code / vendored tree) or is it a stale pre-v0.22 target?
   These nine large files (≈3650 lines total) are the biggest unaudited block.

5. **No manifest, no version gate, no runtime report.** Patch order is
   filesystem `glob()` order; there is no declared ID/phase/version-range, and
   no `patch_report()`. The `new not in source` re-apply guard
   (MUSA-0089/0096) is the only built-in safety. → MUSA-0301.

6. **Test coverage is anchor-based.** `tests/test_patches.py` = **78 tests**
   across ~30 per-patch classes, most asserting *"anchor string exists in
   source"* rather than *runtime behavior*. This is brittle across version
   bumps (the anchor moves → test passes or the patch silently no-ops). →
   MUSA-0304 (convert to behavior tests), MUSA-0307 (cross-version validation).

7. **Exactly two genuine import-side-effect patches existed**
   (`distributed.parallel_state`, `v1.spec_decode.eagle`) — `PATCHES = []`, the
   effect was a monkey-patch fired at module-load time, invisible to any report.
   **RESOLVED by MUSA-0302** (`65181ba2d`): both converted to a module-level
   idempotent `apply()` driven by the explicit `patches.apply_object_patches()`
   phase; `patch_report()` now flags them (`object_patch=True`). The earlier
   claim of "mixed side-effect code inside three `src` files"
   (`sparse_attn_indexer`, `deepseek_v4_ops.cache_utils`, `gpu_model_runner`)
   was **disproven by code audit** — those have zero monkey-patch installs
   (their `_musa_*` helpers are source-transform *text*; `gpu_model_runner`'s
   former monkey-patch is "no longer reachable"). They are pure source-transforms.
   Separately, `RELOAD_AFTER_PATCH` is defined in 7 DeepSeek-V4 patch files but
   **consumed nowhere** (dead metadata; the in-memory-vs-disk reload hazard it
   was meant to address is MUSA-0303's concern).

8. **`setup.py` patches a Python file at build time** (`vllm/_custom_ops.py` in
   `CSRC_TEXT_PATCHES`) — a fourth wrinkle that blurs mechanism 1 vs 3 and
   should be folded into the MUSA-0306 native patch-queue design.

9. **The migration cost is now measured.** PR #55 (v0.20→v0.22 adaptation,
   HEAD `aad1aca59`, 2026-06-02) changed **25 files / +1361 / −79**, including
   `setup.py` (+100), `tests/test_patches.py` (+123), `__init__.py` (+165), and
   10 patch files. This is the per-rebase tax the roadmap exists to reduce, and
   the concrete regression fixture for MUSA-0307.

---

## 6. Classification summary & migration disposition

| Bucket | Count | Mechanism | Disposition (target ticket) |
|---|---|---|---|
| Object monkey-patch (good pattern) | 6 | 4 | keep; use as template (MUSA-0302/0304) |
| Side-effect import patch | 2 | 2 | **DONE** — explicit `apply_object_patches()` phase (MUSA-0302 `65181ba2d`); "3 mixed" disproven (pure source-transforms) |
| Source transform, upstream target | 41 | 1 | → in-memory transform (MUSA-0303); migrate low-risk to object/registry (MUSA-0304); spec-decode subset → compat modules (MUSA-0305) |
| Source transform, private/absent target | 10 | 1 | **audit first** (§5.4); confirm runtime provider or retire |
| Native file override | 4 | 3 | → documented MUSA-owned replacements (MUSA-0306) |
| Native text patch (incl. 1 `.py`) | 19 files | 3 | → ordered `.patch` queue with `git apply --check` (MUSA-0306) |

**Removal-condition rule of thumb** (for the manifest in MUSA-0301): a patch is
removable when (a) the upstream behavior lands and the version gate excludes
all supported snapshots, (b) an upstream extension seam replaces it
(MUSA-0308), or (c) for `private/absent` rows, the providing module is itself
MUSA-owned and the override can live in that module directly.
