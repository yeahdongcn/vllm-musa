# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patches for vLLM compatibility with MUSA platform.

This module contains patches that modify vLLM source files at runtime
to ensure compatibility with the MUSA Triton version.
"""

import importlib.util
import sys
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)

_patches_applied = False


def _get_patch_files():
    """Get all patch files in the patches directory."""
    patches_dir = Path(__file__).parent
    patch_files = []

    for patch_file in patches_dir.glob("*.patch.py"):
        # Extract module name from filename
        # Format: module.name.patch.py -> module.name
        module_name = patch_file.stem.rsplit(".patch", 1)[0]
        # Convert filename format to module format
        # vllm__attention__ops__triton_unified_attention -> vllm.attention.ops.triton_unified_attention
        module_name = module_name.replace("__", ".")
        patch_files.append((module_name, patch_file))

    return patch_files


def _load_patch_config(patch_file: Path) -> list[tuple[str, str]]:
    """Load patch configuration from a patch file.

    Patch files should define a PATCHES list of (old_str, new_str) tuples.
    """
    spec = importlib.util.spec_from_file_location("patch_config", patch_file)
    if spec is None or spec.loader is None:
        return []

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return getattr(module, "PATCHES", [])
    except Exception as e:
        logger.warning(f"Failed to load patch config from {patch_file}: {e}")
        return []


def apply_patches():
    """Apply all patches for MUSA compatibility.

    This function should be called early during platform initialization.
    """
    global _patches_applied
    if _patches_applied:
        return

    patch_files = _get_patch_files()

    for module_name, patch_file in patch_files:
        try:
            # Find the module spec
            try:
                spec = importlib.util.find_spec(module_name)
            except ModuleNotFoundError:
                # Module doesn't exist in this vLLM version (e.g., vllm.worker.worker
                # exists in vLLM 0.10.x but not in 0.13.0 where V0 engine was removed)
                logger.debug(
                    f"Module {module_name} not found in this vLLM version, "
                    "skipping patch (this is expected for version-specific patches)"
                )
                continue
            if spec is None or spec.origin is None:
                logger.debug(f"Module {module_name} not found, skipping patch")
                continue

            # Read the source file
            with open(spec.origin, "r") as f:
                source = f.read()

            # Load patches from patch file
            patches = _load_patch_config(patch_file)
            if not patches:
                continue

            # Check if any patches are needed
            needs_patch = any(old in source for old, new in patches)
            if not needs_patch:
                logger.debug(f"No patches needed for {module_name}")
                continue

            # Apply patches
            patched_source = source
            applied_count = 0
            for old, new in patches:
                if old in patched_source:
                    patched_source = patched_source.replace(old, new)
                    applied_count += 1

            # Write back the patched source
            with open(spec.origin, "w") as f:
                f.write(patched_source)

            # Remove from cache to force reload
            if module_name in sys.modules:
                del sys.modules[module_name]

            logger.info(f"Applied {applied_count} patch(es) to {module_name}")

        except Exception as e:
            logger.warning(f"Failed to apply patches to {module_name}: {e}")

    _patches_applied = True
