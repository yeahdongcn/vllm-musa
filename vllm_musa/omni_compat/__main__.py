# SPDX-License-Identifier: Apache-2.0
"""``python -m vllm_musa.omni_compat`` -- inspect or activate the Omni shim.

status        print the shim state as JSON (exit 0)
install-pth   install the interpreter-startup hook into site-packages
remove-pth    remove it
"""

from __future__ import annotations

import argparse
import json
import sys

from vllm_musa import omni_compat


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vllm_musa.omni_compat")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="print the shim state as JSON")
    install = sub.add_parser("install-pth", help="install the startup hook")
    install.add_argument(
        "--purelib",
        default=None,
        help="target site-packages (default: this interpreter's)",
    )
    remove = sub.add_parser("remove-pth", help="remove the startup hook")
    remove.add_argument("--purelib", default=None)
    args = parser.parse_args(argv)

    if args.command == "status":
        json.dump(omni_compat.status(), sys.stdout, indent=1, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    if args.command == "install-pth":
        path = omni_compat.write_pth(args.purelib)
        print(f"installed {path}")
        return 0
    path = omni_compat.pth_path(args.purelib)
    if path.exists():
        path.unlink()
        print(f"removed {path}")
    else:
        print(f"absent {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
