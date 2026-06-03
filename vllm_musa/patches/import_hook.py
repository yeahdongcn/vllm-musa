# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA-0303: in-memory source-transform import hook.

The legacy patch mechanism (:func:`vllm_musa.patches.apply_patches`) rewrites
the *installed* ``vllm`` source files on disk. That persists across sessions,
races package upgrades, and leaves the installed tree no longer matching its
``RECORD`` (see ``docs/musa-patch-inventory.md`` §5.1). This module provides the
replacement: a :class:`sys.meta_path` finder that applies the **same**
``(old, new)`` PATCHES anchors (and any ``normalize_source`` hook) to a target
``vllm.*`` module's source **in memory at import time** — nothing is written to
disk.

Rollout (MUSA-0303 is the keystone — a mistake changes how *every* source patch
lands, so the default does not move until the full regression matrix passes):

- **Default: legacy disk patcher** (unchanged behavior).
- ``VLLM_MUSA_INMEMORY_PATCH=1`` → opt **in** to this hook (for verification).
- Once verified, the default flips and ``VLLM_MUSA_LEGACY_DISK_PATCH=1`` becomes
  the escape hatch back to the disk patcher.

The selector lives in :func:`vllm_musa.patches.apply_patches` /
``vllm_musa.__init__._apply_vllm_patches``; this module only implements the hook.

Semantics are kept byte-for-byte compatible with ``apply_patches``: for each
target module, ``normalize_source`` (if any) runs, then every ``(old, new)`` is
applied **gated on ``new not in source``** (the MUSA-0089/0096 re-apply guard),
then ``normalize_source`` runs once more. Side-effect patches (empty ``PATCHES``)
are NOT handled here — those are the explicit object-patch phase (MUSA-0302).
"""

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from typing import Callable

from vllm.logger import init_logger

logger = init_logger(__name__)

# Module-global so install is idempotent within a process.
_INSTALLED_FINDER: "MusaInMemoryPatchFinder | None" = None


class _PatchingSourceLoader(importlib.machinery.SourceFileLoader):
    """A ``SourceFileLoader`` that rewrites source in memory before compile.

    Reads the real file via the base loader, then applies the patch module's
    ``normalize_source`` + ``(old, new)`` replacements (same order/gating as
    ``apply_patches``). The file on disk is never modified.
    """

    def __init__(
        self,
        fullname: str,
        path: str,
        patches: list[tuple[str, str]],
        normalizer: Callable[[str], str] | None,
    ) -> None:
        super().__init__(fullname, path)
        self._patches = patches
        self._normalizer = normalizer
        self.musa_transformed = False  # set True once a real change is applied

    def _normalize(self, source: str) -> str:
        if self._normalizer is None:
            return source
        try:
            out = self._normalizer(source)
        except Exception as e:  # never let a normalizer break the import
            logger.warning(
                "MUSA-0303: normalize_source failed for %s: %s", self.name, e
            )
            return source
        return out if isinstance(out, str) else source

    def get_source(self, fullname: str) -> str | None:
        source = super().get_source(fullname)
        if source is None:
            return source
        patched = self._normalize(source)
        for old, new in self._patches:
            # MUSA-0089/0096 gate: only apply when the anchor is present AND the
            # result is not already there (idempotent for INSERT-style patches).
            if old in patched and new not in patched:
                patched = patched.replace(old, new)
        patched = self._normalize(patched)
        if patched != source:
            self.musa_transformed = True
        return patched

    # get_code() in the base class calls get_source() -> source_to_code(), so the
    # patched text is what gets compiled. No further override needed.


# Sentinel: a module name whose .patch.py has not yet been lazily resolved.
_UNRESOLVED = object()


class MusaInMemoryPatchFinder(importlib.abc.MetaPathFinder):
    """meta_path finder that hands patched target modules a patching loader.

    **Lazy** (MUSA-0303 keystone refinement). At install time the finder only
    knows the *names* of modules that have a ``.patch.py`` — building that set is
    cheap and executes no ``.patch.py``. The first time a target module is
    actually imported, ``find_spec`` loads *that one* ``.patch.py`` (by then its
    own deps — ``torch``, ``vllm.triton_utils``, ``vllm_musa._custom_ops`` — are
    importable) and caches its ``(patches, normalizer)``. This lets the finder be
    installed **very early** (before the engine imports attention/MoE/FP8/...)
    without the circular-import hazard an eager patch-map build hits when a
    ``.patch.py`` imports ``vllm_musa._custom_ops`` before it is ready.

    A target whose ``.patch.py`` has empty ``PATCHES`` and no ``normalize_source``
    (a MUSA-0302 side-effect/object patch) resolves to ``None`` and is left to
    normal import. Non-target modules return ``None`` immediately. The real source
    is resolved via :class:`importlib.machinery.PathFinder` (not ``sys.meta_path``)
    so the finder never recurses into itself.
    """

    def __init__(self, name_to_file: dict[str, Path]):
        self._name_to_file = name_to_file
        # module_name -> (patches, normalizer) | None  (None = skip/side-effect/failed)
        self._cache: dict[str, tuple | None] = {}
        self._resolving: set[str] = set()
        self.transformed: set[str] = set()  # targets we actually built a loader for

    def _resolve_entry(self, fullname: str):
        """Lazily exec the target's ``.patch.py`` → ``(patches, normalizer)``.

        Returns ``None`` for a side-effect/empty patch or a load failure. Guards
        reentrancy (loading a ``.patch.py`` may import another target module).
        """
        from . import _load_patch_module

        if fullname in self._resolving:
            return None
        self._resolving.add(fullname)
        try:
            mod = _load_patch_module(self._name_to_file[fullname])
            if mod is None:
                return None
            patches = getattr(mod, "PATCHES", []) or []
            normalizer = getattr(mod, "normalize_source", None)
            if not callable(normalizer):
                normalizer = None
            if not patches and normalizer is None:
                return None  # side-effect / object patch — MUSA-0302's job
            return (patches, normalizer)
        except Exception as e:  # never break the import on a patch-load error
            logger.warning("MUSA-0303: failed to load patch for %s: %s", fullname, e)
            return None
        finally:
            self._resolving.discard(fullname)

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self._name_to_file:
            return None
        if fullname in self._resolving:
            return None  # mid-resolve for this very module; let normal import run
        entry = self._cache.get(fullname, _UNRESOLVED)
        if entry is _UNRESOLVED:
            entry = self._resolve_entry(fullname)
            self._cache[fullname] = entry
        if entry is None:
            return None  # side-effect/empty/failed -> normal import
        patches, normalizer = entry
        real_spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        origin = getattr(real_spec, "origin", None) if real_spec else None
        if real_spec is None or not origin or not str(origin).endswith(".py"):
            return None
        loader = _PatchingSourceLoader(fullname, origin, patches, normalizer)
        new_spec = importlib.util.spec_from_file_location(fullname, origin, loader=loader)
        if new_spec is None:
            return None
        # Preserve package-ness so submodules still resolve.
        new_spec.submodule_search_locations = real_spec.submodule_search_locations
        self.transformed.add(fullname)
        return new_spec


def _target_name_map() -> dict[str, Path]:
    """Cheap module-name -> ``.patch.py`` map. Executes no ``.patch.py``."""
    from . import _get_patch_files

    return {module_name: patch_file for module_name, patch_file in _get_patch_files()}


def install_import_hook() -> list[dict]:
    """Install the in-memory patch finder at the front of ``sys.meta_path``.

    Cheap and safe to call **very early** (only the module-name set is built here;
    each ``.patch.py`` loads lazily when its target imports — see the finder).
    Idempotent.

    Already-imported guard: a target already in ``sys.modules`` cannot be
    retro-patched in memory. We lazily resolve **only that already-imported
    subset** (their deps are loaded, so exec'ing the ``.patch.py`` is safe) and
    record an ``already-imported`` failure (``is_failure``) for any that is a real
    source-transform. Side-effect/object patches in that subset are not failures.

    Returns a per-patch report; never raises.
    """
    global _INSTALLED_FINDER
    if _INSTALLED_FINDER is not None:
        return [{"module": "*", "state": "already-installed"}]

    name_to_file = _target_name_map()
    finder = MusaInMemoryPatchFinder(name_to_file)

    report: list[dict] = []
    n_failures = 0
    for module_name in name_to_file:
        if module_name not in sys.modules:
            continue
        # Resolve only the imported subset (safe — their deps are loaded). A real
        # source-transform already imported is a failure; side-effect ones aren't.
        if finder._resolve_entry(module_name) is None:
            continue
        report.append(
            {"module": module_name, "state": "already-imported", "is_failure": True}
        )
        n_failures += 1
        logger.warning(
            "MUSA-0303: target %s already imported before the in-memory hook "
            "installed; its source-transform patch will NOT apply this process.",
            module_name,
        )

    sys.meta_path.insert(0, finder)
    _INSTALLED_FINDER = finder
    logger.info(
        "MUSA-0303: in-memory patch hook installed (%d candidate target(s)%s).",
        len(name_to_file),
        f", {n_failures} already-imported failure(s)" if n_failures else "",
    )
    return report


def uninstall_import_hook() -> None:
    """Remove the finder from ``sys.meta_path`` (mainly for tests)."""
    global _INSTALLED_FINDER
    if _INSTALLED_FINDER is not None:
        try:
            sys.meta_path.remove(_INSTALLED_FINDER)
        except ValueError:
            pass
        _INSTALLED_FINDER = None


def is_installed() -> bool:
    return _INSTALLED_FINDER is not None
