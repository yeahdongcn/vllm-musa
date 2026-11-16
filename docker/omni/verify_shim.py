#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise the Omni drift shim against the installed vLLM, without a GPU.

What this proves, in a driverless build:

* the startup hook is installed and active, through whichever mechanism;
* every ``RELOCATION`` and ``BACKPORT`` module the table declares really does
  import, and really does provide the names Omni imports from it;
* every ``ADAPTER`` applied to its pinned module;
* the unsupported paths are reported rather than silently absent.

What it deliberately does not prove: that ``import vllm_omni`` succeeds. That
import activates the MUSA platform plugin, which queries device capability
through ``torch_musa`` and raises without a driver, so it belongs to the
hardware smoke rather than the build. ``--require-omni`` runs it anyway, for use
on a leased GPU.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys

_DEVICE_MARKERS = (
    "MUSA driver initialization failed",
    "No MUSA runtime",
    "you might not have a Mthreads GPU",
)


def _needs_a_device(exc: BaseException) -> bool:
    """Whether an import failed for want of a GPU rather than for want of a shim."""
    return any(marker in str(exc) for marker in _DEVICE_MARKERS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-omni",
        action="store_true",
        help="also import vllm_omni (needs a MUSA driver)",
    )
    parser.add_argument(
        "--driverless",
        action="store_true",
        help="report a module that cannot import without a MUSA device as skipped, not failed",
    )
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    import vllm

    from vllm_musa import omni_compat
    from vllm_musa.omni_compat import table

    failures: list[str] = []
    result: dict = {
        "vllm_version": vllm.__version__,
        "hooks": omni_compat.active_hooks(),
        "installed": omni_compat.is_installed(),
        "served": {},
        "adapted": {},
        "unsupported": list(omni_compat.unsupported_paths()),
    }

    if not result["installed"]:
        failures.append(
            "no shim hook is installed; the .pth did not run and install() was not called"
        )

    for entry in table.importable_entries():
        if not omni_compat.version_matches(entry.applies_to, vllm.__version__):
            result["served"][entry.module] = "inactive (pin outside applies_to)"
            continue
        try:
            module = importlib.import_module(entry.module)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if args.driverless and _needs_a_device(exc):
                result["served"][entry.module] = f"skipped, needs a device: {detail}"
                continue
            failures.append(f"{entry.module}: {detail}")
            result["served"][entry.module] = f"import failed: {detail}"
            continue
        missing = [name for name in entry.names if not hasattr(module, name)]
        if missing:
            failures.append(f"{entry.module}: served but missing {missing}")
        shim = getattr(module, "__vllm_musa_shim__", None)
        result["served"][entry.module] = {
            "kind": shim["kind"] if shim else "provided by the pin",
            "names": list(entry.names),
            "missing": missing,
        }

    for entry in table.adapter_entries():
        if not omni_compat.version_matches(entry.applies_to, vllm.__version__):
            result["adapted"][entry.module] = "inactive (pin outside applies_to)"
            continue
        try:
            module = importlib.import_module(entry.module)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if args.driverless and _needs_a_device(exc):
                # The module itself reaches into torch_musa at import. Nothing
                # about the adapter can be checked here; the hardware run with
                # --require-omni is where it is verified.
                result["adapted"][entry.module] = f"skipped, needs a device: {detail}"
                continue
            failures.append(f"{entry.module}: {detail}")
            result["adapted"][entry.module] = f"import failed: {detail}"
            continue
        applied = getattr(module, "__vllm_musa_adapted__", None)
        if applied is None:
            failures.append(
                f"{entry.module}: the adapter did not run; the hook was installed too late"
            )
            result["adapted"][entry.module] = "not adapted"
            continue
        result["adapted"][entry.module] = applied["names"]

    if args.require_omni:
        try:
            import vllm_omni

            result["vllm_omni_version"] = vllm_omni.__version__
        except Exception as exc:
            failures.append(f"import vllm_omni: {type(exc).__name__}: {exc}")
            result["vllm_omni_version"] = None

    result["failures"] = failures
    result["verdict"] = "pass" if not failures else "fail"

    text = json.dumps(result, indent=1, sort_keys=True)
    print(text)
    if args.report:
        with open(args.report, "w") as handle:
            handle.write(text + "\n")
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
