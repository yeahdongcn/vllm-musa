#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Make this checkout's Omni compat shim available to the base image's vllm-musa.

Once the shim ships in a vLLM-MUSA release the base image already carries it and
this is a no-op. Until then -- and whenever an Omni ref needs a shim entry newer
than the base was built with -- the shim has to reach an installed vllm-musa
that predates it.

Both cases are the same operation: copy ``vllm_musa/omni_compat/`` and the
top-level ``.pth`` bootstrap into the installed package, and refuse if the files
differ in a way that suggests the base is actually newer than this checkout.
Only pure-Python files move, so nothing is rebuilt and the base's compiled
extensions are untouched.
"""

from __future__ import annotations

import argparse
import filecmp
import importlib.util
import shutil
import sys
import sysconfig
from pathlib import Path


def installed_package_dir() -> Path:
    spec = importlib.util.find_spec("vllm_musa")  # does not execute __init__
    locations = (
        list(getattr(spec, "submodule_search_locations", None) or []) if spec else []
    )
    if not locations:
        raise SystemExit("vllm_musa is not installed in this interpreter")
    return Path(locations[0])


def purelib() -> Path:
    """Where a top-level module has to go to be importable.

    Not next to the package: vllm-musa is installed editable through a PEP 660
    strict finder (``__editable___vllm_musa_*_finder.py``) that maps only the
    ``vllm_musa`` package, so its source directory is never added to
    ``sys.path`` and a top-level module dropped there is unreachable -- which
    is exactly how the ``.pth`` first failed.
    """
    return Path(sysconfig.get_paths()["purelib"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="vllm-musa checkout in the build context",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing omni_compat"
    )
    args = parser.parse_args(argv)

    src_pkg = args.source / "vllm_musa" / "omni_compat"
    src_pth = args.source / "vllm_musa_omni_compat_pth.py"
    for path in (src_pkg, src_pth):
        if not path.exists():
            raise SystemExit(f"missing in the build context: {path}")

    target_pkg = installed_package_dir() / "omni_compat"
    target_pth = purelib() / "vllm_musa_omni_compat_pth.py"

    if target_pkg.exists() and not args.force:
        same = filecmp.dircmp(str(src_pkg), str(target_pkg))
        if not (same.left_only or same.right_only or same.diff_files):
            print(f"shim already installed and identical: {target_pkg}")
            return 0
        print(
            f"shim present but different at {target_pkg}; "
            f"left_only={same.left_only} right_only={same.right_only} diff={same.diff_files}",
            file=sys.stderr,
        )
        print("refusing to overwrite without --force", file=sys.stderr)
        return 1

    if target_pkg.exists():
        shutil.rmtree(target_pkg)
    shutil.copytree(src_pkg, target_pkg, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(src_pth, target_pth)
    print(f"staged {target_pkg}")
    print(f"staged {target_pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
