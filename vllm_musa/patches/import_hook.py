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
from types import ModuleType
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


class MusaInMemoryPatchFinder(importlib.abc.MetaPathFinder):
    """meta_path finder that hands patched target modules a patching loader.

    For any non-target module it returns ``None`` (normal import proceeds). For a
    target ``vllm.*`` module it resolves the real source via the standard
    :class:`importlib.machinery.PathFinder` (NOT the meta_path, to avoid
    recursing into ourselves) and wraps it with :class:`_PatchingSourceLoader`.
    """

    def __init__(self, patch_map: dict[str, tuple[list[tuple[str, str]], Callable | None]]):
        # module_name -> (patches, normalizer)
        self._patch_map = patch_map
        # which targets were actually transformed (populated lazily on import)
        self.transformed: set[str] = set()

    def find_spec(self, fullname, path=None, target=None):
        entry = self._patch_map.get(fullname)
        if entry is None:
            return None
        # Resolve the real on-disk spec WITHOUT going through sys.meta_path
        # (PathFinder is the std file-based finder), so we never recurse into
        # this finder. `path` is the parent package __path__ supplied by import.
        real_spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        origin = getattr(real_spec, "origin", None) if real_spec else None
        if real_spec is None or not origin or not str(origin).endswith(".py"):
            return None
        patches, normalizer = entry
        loader = _PatchingSourceLoader(fullname, origin, patches, normalizer)
        new_spec = importlib.util.spec_from_file_location(
            fullname, origin, loader=loader
        )
        if new_spec is None:
            return None
        # Preserve package-ness so submodules still resolve.
        new_spec.submodule_search_locations = real_spec.submodule_search_locations
        self.transformed.add(fullname)
        return new_spec


def _build_patch_map() -> tuple[dict[str, tuple], list[dict]]:
    """Build the {module: (patches, normalizer)} map for source-transform patches.

    Reuses the existing ``.patch.py`` anchors. Side-effect patches (empty
    ``PATCHES`` and no ``normalize_source``) are excluded — those are the
    MUSA-0302 object-patch phase. Returns the map plus a per-patch report list.
    """
    from . import _get_patch_files, _load_patch_module

    patch_map: dict[str, tuple] = {}
    report: list[dict] = []
    for module_name, patch_file in _get_patch_files():
        mod: ModuleType | None = _load_patch_module(patch_file)
        if mod is None:
            report.append({"module": module_name, "state": "load-failed"})
            continue
        patches = getattr(mod, "PATCHES", []) or []
        normalizer = getattr(mod, "normalize_source", None)
        if not callable(normalizer):
            normalizer = None
        if not patches and normalizer is None:
            # side-effect / object patch — not our business
            continue
        patch_map[module_name] = (patches, normalizer)
        report.append({"module": module_name, "state": "registered"})
    return patch_map, report


def install_import_hook() -> list[dict]:
    """Install the in-memory patch finder at the front of ``sys.meta_path``.

    Idempotent. For each target module already present in ``sys.modules`` BEFORE
    the hook installs, the on-disk source was already imported and the hook
    cannot retro-patch it; that is recorded as ``already-imported`` (a required
    patch in that state is a real failure — surfaced via the ``is_failure`` flag).

    Returns a per-patch report; never raises.
    """
    global _INSTALLED_FINDER
    if _INSTALLED_FINDER is not None:
        return [{"module": "*", "state": "already-installed"}]

    patch_map, report = _build_patch_map()

    # Imported-module guard: anything already imported can't be patched in memory.
    by_mod = {e["module"]: e for e in report}
    for module_name in patch_map:
        if module_name in sys.modules:
            entry = by_mod.setdefault(module_name, {"module": module_name})
            entry["state"] = "already-imported"
            entry["is_failure"] = True
            logger.warning(
                "MUSA-0303: target %s already imported before the in-memory hook "
                "installed; its source-transform patch will NOT apply this process. "
                "Install the hook earlier (platform plugin load).",
                module_name,
            )

    finder = MusaInMemoryPatchFinder(patch_map)
    sys.meta_path.insert(0, finder)
    _INSTALLED_FINDER = finder
    n_failures = sum(1 for e in report if e.get("is_failure"))
    logger.info(
        "MUSA-0303: in-memory patch hook installed for %d source-transform "
        "module(s)%s.",
        len(patch_map),
        f" ({n_failures} already-imported failure(s))" if n_failures else "",
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
