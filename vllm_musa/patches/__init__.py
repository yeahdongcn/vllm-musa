# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patches for vLLM compatibility with MUSA platform.

This module contains patches that modify vLLM source files at runtime
to ensure compatibility with the MUSA Triton version.
"""

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

from vllm.logger import init_logger

logger = init_logger(__name__)

_patches_applied = False


def _get_patch_files():
    """Get all patch files in the patches directory, in deterministic order.

    MUSA-0301: sort by filename so discovery/application order is reproducible
    across machines and runs (``glob`` order is filesystem-dependent). Patches
    target independent modules so order does not affect correctness, but a
    stable order makes the patch report and any import-side-effect ordering
    deterministic.
    """
    patches_dir = Path(__file__).parent
    patch_files = []

    for patch_file in sorted(patches_dir.glob("*.patch.py")):
        # Extract module name from filename
        # Format: module.name.patch.py -> module.name
        module_name = patch_file.stem.rsplit(".patch", 1)[0]
        # Convert filename format to module format
        # vllm__attention__ops__triton_unified_attention -> vllm.attention.ops.triton_unified_attention
        module_name = module_name.replace("__", ".")
        patch_files.append((module_name, patch_file))

    return patch_files


def _load_patch_module(patch_file: Path) -> ModuleType | None:
    """Load a patch module.

    Patch files may define a PATCHES list of (old_str, new_str) tuples and an
    optional normalize_source(source: str) -> str hook for idempotent cleanup.
    """
    spec = importlib.util.spec_from_file_location("patch_config", patch_file)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.warning(f"Failed to load patch config from {patch_file}: {e}")
        return None


def _load_patch_config(patch_file: Path) -> list[tuple[str, str]]:
    """Load patch configuration from a patch file.

    Patch files should define a PATCHES list of (old_str, new_str) tuples.
    """
    module = _load_patch_module(patch_file)
    if module is None:
        return []
    return getattr(module, "PATCHES", [])


def _normalize_source(
    patch_module: ModuleType,
    source: str,
    module_name: str,
) -> str:
    normalizer = getattr(patch_module, "normalize_source", None)
    if not callable(normalizer):
        return source
    try:
        normalized = normalizer(source)
    except Exception as e:
        logger.warning(f"Failed to normalize patched source for {module_name}: {e}")
        return source
    if not isinstance(normalized, str):
        logger.warning(
            "Ignoring non-string normalized source returned for %s", module_name
        )
        return source
    return normalized


def _resolve_module_origin(module_name: str) -> str | None:
    """Resolve a module source file without importing the target module."""
    parts = module_name.split(".")
    if parts and parts[0] == "vllm":
        roots: list[Path] = []
        try:
            root_spec = importlib.util.find_spec("vllm")
        except (ModuleNotFoundError, ImportError):
            root_spec = None
        if root_spec is not None:
            if root_spec.submodule_search_locations:
                roots.extend(Path(p) for p in root_spec.submodule_search_locations)
            elif root_spec.origin is not None:
                roots.append(Path(root_spec.origin).parent)
        for sys_path in sys.path:
            if not sys_path:
                sys_path = os.getcwd()
            candidate_root = Path(sys_path) / "vllm"
            if candidate_root.is_dir():
                roots.append(candidate_root)

        seen: set[Path] = set()
        for root in roots:
            root = root.resolve()
            if root in seen:
                continue
            seen.add(root)
            rel = Path(*parts[1:]) if len(parts) > 1 else Path()
            module_file = (root / rel).with_suffix(".py")
            if module_file.is_file():
                return str(module_file)
            package_file = root / rel / "__init__.py"
            if package_file.is_file():
                return str(package_file)

    try:
        spec = importlib.util.find_spec(module_name)
    except (ModuleNotFoundError, ImportError) as e:
        logger.debug(
            f"Module {module_name} not found or has import issues: {e}, "
            "skipping patch (this is expected for version-specific patches "
            "or when modules are not yet fully initialized)"
        )
        return None
    if spec is None or spec.origin is None:
        logger.debug(f"Module {module_name} not found, skipping patch")
        return None
    return spec.origin


def patch_report() -> list[dict]:
    """Report the status of every vLLM-MUSA source patch (read-only, MUSA-0301).

    Inspects the *installed* vLLM source for each ``*.patch.py`` and classifies
    it without modifying anything (safe to call any time, including before
    ``apply_patches``):

    - ``kind``: ``source-transform`` (non-empty ``PATCHES``) or ``side-effect``
      (empty ``PATCHES`` — an import-side-effect / object monkey-patch).
    - ``target_resolved``: whether the target module resolves in this env.
    - ``status``: ``applied`` / ``needs-apply`` / ``no-op`` / ``missing-target`` /
      ``unreadable-target`` / ``side-effect`` / ``load-failed`` / ``error``.

    Returned in deterministic ``_get_patch_files()`` order.
    """
    report: list[dict] = []
    for module_name, patch_file in _get_patch_files():
        entry: dict = {
            "module": module_name,
            "file": patch_file.name,
            "kind": "unknown",
            "target_resolved": False,
            "status": "unknown",
        }
        try:
            origin = _resolve_module_origin(module_name)
            entry["target_resolved"] = origin is not None
            patch_module = _load_patch_module(patch_file)
            if patch_module is None:
                entry.update(kind="load-failed", status="load-failed")
            else:
                patches = getattr(patch_module, "PATCHES", [])
                has_norm = callable(getattr(patch_module, "normalize_source", None))
                if not patches and not has_norm:
                    entry.update(kind="side-effect", status="side-effect")
                elif origin is None:
                    entry.update(kind="source-transform", status="missing-target")
                else:
                    entry["kind"] = "source-transform"
                    try:
                        with open(origin, "r") as f:
                            source = f.read()
                    except OSError:
                        source = None
                        entry["status"] = "unreadable-target"
                    if source is not None:
                        pending = sum(
                            1 for old, new in patches if old in source and new not in source
                        )
                        applied = sum(1 for _, new in patches if new in source)
                        entry["pending"] = pending
                        entry["status"] = (
                            "needs-apply"
                            if pending
                            else "applied"
                            if applied
                            else "no-op"
                        )
        except Exception as e:  # a report must never raise
            entry.update(status="error", error=str(e))
        report.append(entry)
    return report


def apply_patches(force: bool = False):
    """Apply all patches for MUSA compatibility.

    This function should be called early during platform initialization.
    """
    global _patches_applied
    if _patches_applied and not force:
        return

    patch_files = _get_patch_files()

    for module_name, patch_file in patch_files:
        try:
            # Resolve source path without importing the patched module. Several
            # DeepSeek-V4 modules import MUSA attention helpers during module
            # import, so importlib.find_spec(module_name) can trip plugin
            # circularity before the patches that would make the import safe.
            origin = _resolve_module_origin(module_name)
            if origin is None:
                continue

            # Read the source file
            try:
                with open(origin, "r") as f:
                    source = f.read()
            except (IOError, OSError) as e:
                logger.debug(f"Cannot read {origin}: {e}, skipping patch")
                continue

            # Load patches from patch file. Loading can also install
            # monkey-patches for patch modules that intentionally use an empty
            # PATCHES list.
            patch_module = _load_patch_module(patch_file)
            if patch_module is None:
                continue
            patches = getattr(patch_module, "PATCHES", [])

            patched_source = _normalize_source(patch_module, source, module_name)
            source_normalized = patched_source != source
            if not patches and not source_normalized:
                continue

            # Check if any patches are needed.
            # MUSA-0089/0096 fix: for INSERT-style patches (where `new` contains
            # `old` plus extra inserted text), `old in source` remains True after
            # first apply, causing accumulation across re-imports. Gate on
            # `new not in source` to detect "patch already applied" state.
            # Behaviour-preserving for REPLACEMENT-style patches where `new`
            # differs entirely from `old`.
            needs_patch = (
                any(
                    old in patched_source and new not in patched_source
                    for old, new in patches
                )
                or source_normalized
            )
            if not needs_patch:
                logger.debug(f"No patches needed for {module_name}")
                continue

            # Apply patches.
            # MUSA-0089/0096 fix: same `new not in patched_source` gate as
            # the outer needs_patch check. Without this, INSERT-style patches
            # re-apply on every import and accumulate (e.g., MUSA-0088's
            # cuda_communicator.py grew 10 duplicate `elif current_platform.is_musa()`
            # blocks before MUSA-0089 caught it and manual sed -i restored).
            applied_count = 0
            for old, new in patches:
                if old in patched_source and new not in patched_source:
                    patched_source = patched_source.replace(old, new)
                    applied_count += 1
            patched_source = _normalize_source(
                patch_module,
                patched_source,
                module_name,
            )
            if patched_source == source:
                logger.debug(f"No source changes produced for {module_name}")
                continue

            # Write back the patched source ATOMICALLY via tempfile + rename.
            # Rationale: vLLM's spawn-based multiproc executor starts N worker
            # processes nearly simultaneously, each of which re-imports
            # vllm_musa and therefore re-runs apply_patches(). If a worker
            # holds the file open for read (e.g. during `import
            # vllm.utils.deep_gemm`) while another process is mid-write, the
            # worker can observe a truncated/partial file and raise an
            # ImportError mid-startup (observed during MUSA-0046 MiniMax-M2.7
            # smoke #4, 2026-05-14). Atomic rename guarantees readers see
            # either the pre-patch or post-patch content, never partial.
            #
            # Do not evict an already-imported module from sys.modules: some
            # vLLM modules register torch custom ops at import time, and
            # re-importing them would register the same schema twice in the
            # current process.
            import tempfile  # noqa: I001 — local import keeps top-level imports stable

            tmp_fd, tmp_path = tempfile.mkstemp(
                prefix=os.path.basename(origin) + ".",
                dir=os.path.dirname(origin),
            )
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    f.write(patched_source)
                os.rename(tmp_path, origin)
            except Exception:
                # Best-effort cleanup on error.
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            logger.info(f"Applied {applied_count} patch(es) to {module_name}")

        except Exception as e:
            # More detailed error handling for circular imports
            if "circular import" in str(e) or "partially initialized" in str(e):
                logger.debug(
                    f"Skipping patch for {module_name} due to circular import "
                    f"during initialization: {e}"
                )
            else:
                logger.warning(f"Failed to apply patches to {module_name}: {e}")

    _patches_applied = True
