#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Mirror vLLM-MUSA's Python 3.10 decision onto a vLLM-Omni source tree.

Upstream vLLM raised ``requires-python`` to ``>=3.11``; vLLM-Omni followed, and
vLLM-MUSA deliberately did not, carrying
``vllm_musa/patches/series/0196-MUSA-run-on-Python-3.10.patch`` instead. Editing
the Omni marker applies the same decision to the same stack.

The narrower-looking alternative, ``pip install --ignore-requires-python``, is
not used: it also admits unrelated 3.11-only dependencies, so it weakens a check
this build still wants. ``--policy strict`` refuses the build instead, for
someone who wants upstream's floor enforced.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PATTERN = re.compile(r'requires-python\s*=\s*"(?P<spec>[^"]*)"')


def admits(spec: str, version: str) -> bool:
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
    except ImportError:  # pragma: no cover - packaging ships with pip
        return True
    return Version(version) in SpecifierSet(spec)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyproject", required=True, type=Path)
    parser.add_argument("--policy", default="relax", choices=["relax", "strict"])
    parser.add_argument("--floor", default=">=3.10")
    parser.add_argument("--runtime", default=".".join(map(str, sys.version_info[:3])))
    parser.add_argument("--record", type=Path, default=None)
    args = parser.parse_args(argv)

    text = args.pyproject.read_text()
    match = PATTERN.search(text)
    original = match.group("spec") if match else "(absent)"

    if match is None:
        effective = original
        print("python floor: no requires-python to adjust")
    elif args.policy == "strict":
        effective = original
        if not admits(original, args.runtime):
            print(
                f"[error] {args.pyproject} requires {original}, which excludes the "
                f"Python {args.runtime} this MUSA line runs on. Use --policy relax to "
                "mirror vllm-musa's own 3.10 patch.",
                file=sys.stderr,
            )
            return 1
        print(f"python floor: strict, leaving requires-python = {original!r}")
    else:
        args.pyproject.write_text(
            PATTERN.sub(f'requires-python = "{args.floor}"', text, count=1)
        )
        effective = args.floor
        print(f"python floor: relaxed {original!r} -> {effective!r}")

    if args.record is not None:
        args.record.write_text(
            f"original={original}\neffective={effective}\npolicy={args.policy}\nruntime={args.runtime}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
