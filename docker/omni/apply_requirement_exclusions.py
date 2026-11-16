#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Drop requirements that cannot coexist with the pinned MUSA foundation.

Editing the requirements files, rather than installing with ``--no-deps`` or
``--ignore-installed``, keeps the installed metadata honest: the distribution
really does not require what was removed, so ``pip check`` stays meaningful and
a later install cannot silently pull the conflict back in.

Every exclusion is listed with its reason in ``requirement-exclusions.txt`` and
echoed into the image, so ``docker run --entrypoint cat <image>
/opt/vllm-omni-requirement-exclusions.txt`` answers "what is missing from this
build and why" without the Dockerfile.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# A requirement line starts with the distribution name; everything after the
# first comparison, bracket, semicolon or whitespace is version/marker/extras.
NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def read_exclusions(path: Path) -> list[str]:
    names = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(canonical(line))
    return names


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--omni-root", required=True, type=Path)
    parser.add_argument("--exclusions", required=True, type=Path)
    parser.add_argument("--record", type=Path, default=None)
    args = parser.parse_args(argv)

    excluded = read_exclusions(args.exclusions)
    if not excluded:
        print("no requirement exclusions configured")
        return 0

    requirements_dir = args.omni_root / "requirements"
    if not requirements_dir.is_dir():
        print(f"[error] {requirements_dir} does not exist", file=sys.stderr)
        return 1

    removed: list[tuple[str, str]] = []
    for path in sorted(requirements_dir.rglob("*.txt")):
        lines = path.read_text().splitlines(keepends=True)
        kept = []
        for line in lines:
            match = NAME.match(line)
            if match and canonical(match.group(1)) in excluded:
                removed.append((str(path.relative_to(args.omni_root)), line.strip()))
                continue
            kept.append(line)
        if len(kept) != len(lines):
            path.write_text("".join(kept))

    for where, line in removed:
        print(f"excluded {line!r} from {where}")
    if not removed:
        # Not an error: the Omni ref may simply not require it any more, which
        # is a reason to delete the entry, not to fail the build.
        print(
            f"none of the configured exclusions appear in {requirements_dir}: {excluded}"
        )

    if args.record is not None:
        args.record.write_text(
            "".join(f"{where}: {line}\n" for where, line in removed)
            or f"configured={','.join(excluded)}; none present in this Omni ref\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
