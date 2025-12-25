# vLLM MUSA Platform Patches

This directory contains patches that modify vLLM source files at runtime to ensure compatibility with the MUSA platform.

## Why Patches Are Needed

The MUSA platform uses a modified version of Triton that has some syntax incompatibilities with the standard Triton used by vLLM. These patches fix these incompatibilities at runtime when the MUSA platform plugin is loaded.

## Patch File Naming Convention

Patch files follow this naming convention:

```
{module_path_with_double_underscores}.patch.py
```

For example:
- `vllm__attention__ops__triton_unified_attention.patch.py` patches `vllm.attention.ops.triton_unified_attention`

The double underscores (`__`) in the filename are converted to dots (`.`) to form the target module path.

## Patch File Format

Each patch file should define a `PATCHES` list containing tuples of `(old_string, new_string)`:

```python
PATCHES = [
    ("old code to replace", "new replacement code"),
    # ... more patches
]
```

## Current Patches

### vllm__attention__ops__triton_unified_attention.patch.py

**Target:** `vllm.attention.ops.triton_unified_attention`

**Issue:** MUSA Triton doesn't support Python's annotated assignment syntax (PEP 526)

**Error:**
```
triton.compiler.errors.CompilationError:
    left: tl.int32 = 0
    ^
AttributeError("'AnnAssign' object has no attribute 'targets'")
```

**Fix:** Replace `left: tl.int32 = 0` with `left = 0`

**Affected Function:** `find_seq_idx()` - Binary search helper for the unified attention kernel

### vllm__v1__worker__gpu_worker.patch.py

**Target:** `vllm.v1.worker.gpu_worker`

**vLLM Versions:** 0.10.x and 0.13.x (all V1 engine versions)

**Issue:** The V1 GPU worker only checks for `device.type == "cuda"`, which doesn't match MUSA devices

**Fix:** Extend device type check to also accept "musa" device type

### vllm__worker__worker.patch.py

**Target:** `vllm.worker.worker`

**vLLM Versions:** 0.10.x only (V0 engine - removed in 0.13.x)

**Issue:** The V0 worker only checks for `device.type == "cuda"`, which doesn't match MUSA devices

**Fix:** Extend device type check to also accept "musa" device type

**Note:** This module doesn't exist in vLLM 0.13.0+ since the V0 engine was removed. The patch system automatically skips this patch on newer vLLM versions.

## Version-Specific Patches

Some patches are version-specific and will be automatically skipped if the target module doesn't exist:

| Patch | vLLM 0.10.x | vLLM 0.13.x |
|-------|-------------|-------------|
| `vllm__attention__ops__triton_unified_attention` | ✅ Applied | ✅ Applied |
| `vllm__v1__worker__gpu_worker` | ✅ Applied | ✅ Applied |
| `vllm__worker__worker` | ✅ Applied | ⏭️ Skipped (module doesn't exist) |

When a patch is skipped due to a missing module, a debug message is logged (not a warning), as this is expected behavior for version-specific patches.

## How Patches Are Applied

1. When the MUSA platform plugin is loaded, it calls `apply_patches()` from this module
2. The function scans for all `*.patch.py` files in this directory
3. For each patch file:
   - Extracts the target module name from the filename
   - Loads the `PATCHES` list from the patch file
   - Reads the target module's source file
   - Applies string replacements
   - Writes the patched source back to disk
   - Clears the module from `sys.modules` to force a fresh import

## Adding New Patches

1. Create a new file named `{module__path}.patch.py`
2. Add documentation explaining the issue and solution
3. Define the `PATCHES` list with your replacements
4. Test that the patch is applied correctly

Example:
```python
# vllm__some__module.patch.py
"""
Patch for vllm.some.module

Issue: Description of the problem
Solution: Description of the fix
"""

PATCHES = [
    ("problematic code", "fixed code"),
]
```

## Notes

- Patches modify files on disk, so they persist across Python sessions
- Patches are only applied once per module (tracked by `_patches_applied` flag)
- If a patch has already been applied (old string not found), it's skipped silently
- Failed patches log a warning but don't prevent the platform from loading

