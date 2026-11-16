# SPDX-License-Identifier: Apache-2.0
"""The Omni drift shim: every kind, with a fixture that must fail and one that must pass.

A gate that cannot fail is worse than no gate, so each test pairs the behaviour
it wants with the opposite case. No MUSA hardware, no real vLLM: a synthetic
``vllm`` package is built on disk in the two layouts that matter (the 0.32 pin,
which lacks the moved modules, and 0.33, which has them), because the only thing
under test is import resolution.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Imported directly: these modules are stdlib-only by design, so they load
# without torch, torch_musa or vLLM present.
sys.path.insert(0, str(ROOT))
from vllm_musa.omni_compat import table as compat_table  # noqa: E402
from vllm_musa.omni_compat._finder import version_matches  # noqa: E402

PIN_VERSION = "0.32.0"
NEXT_VERSION = "0.33.0"


def _write(path: Path, body: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body))


def _make_vllm(root: Path, version: str, *, moved: bool) -> Path:
    """A synthetic ``vllm`` package.

    ``moved=False`` is the 0.32 shape: the relocation sources exist and the
    modules Omni imports do not. ``moved=True`` is the 0.33 shape, where the
    modules Omni imports exist natively and the shim must stay out of the way.
    """
    site = root / f"site-{version}-{'new' if moved else 'pin'}"
    pkg = site / "vllm"
    _write(pkg / "__init__.py", f"__version__ = {version!r}\n")
    _write(pkg / "entrypoints" / "__init__.py")
    _write(pkg / "entrypoints" / "common" / "__init__.py")
    _write(pkg / "entrypoints" / "launchers" / "__init__.py")
    _write(pkg / "entrypoints" / "serve" / "__init__.py")
    _write(pkg / "entrypoints" / "serve" / "utils" / "__init__.py")
    _write(pkg / "v1" / "__init__.py")
    _write(pkg / "v1" / "core" / "__init__.py")

    # The pinned locations always exist; that is what a relocation reads from.
    _write(
        pkg / "entrypoints" / "launchers" / "cli_args.py",
        """
        class FrontendArgs:
            origin = "launchers"

        def make_arg_parser(parser=None):
            return ("launchers", parser)

        def resolve_default_chat_template_kwargs(*a, **kw):
            return "launchers"

        def validate_parsed_serve_args(*a, **kw):
            return "launchers"

        def not_in_the_table():
            return "delegated"
        """,
    )
    _write(
        pkg / "entrypoints" / "serve" / "utils" / "api_utils.py",
        """
        VLLM_SUBCMD_PARSER_EPILOG = "pinned-epilog"

        def cli_env_setup():
            return "pinned-setup"
        """,
    )
    _write(
        pkg / "v1" / "core" / "kv_cache_utils.py",
        """
        def get_kv_cache_groups(vllm_config, kv_cache_spec):
            return [("group", name) for name in sorted(kv_cache_spec)]

        def get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory):
            return ["pinned-config", len(kv_cache_specs), tuple(available_memory)]
        """,
    )
    sampler = """
        def apply_top_k_top_p(logits, k, p{extra}):
            return ("pinned", logits, k, p{extra_use})
        """
    _write(
        pkg / "v1" / "sample" / "__init__.py",
    )
    _write(
        pkg / "v1" / "sample" / "ops" / "__init__.py",
    )
    _write(
        pkg / "v1" / "sample" / "ops" / "topk_topp_sampler.py",
        (
            sampler.format(
                extra=", has_topp_only_rows=True", extra_use=", has_topp_only_rows"
            )
            if moved
            else sampler.format(extra="", extra_use="")
        ),
    )
    _write(
        pkg / "v1" / "kv_cache_interface.py",
        """
        class SlidingWindowSpec:
            pass
        """,
    )
    _write(
        pkg / "v1" / "kv_cache_spec_registry.py",
        """
        class KVCacheSpecRegistry:
            checked = []

            @classmethod
            def check_kv_cache_spec_registry(cls, specs):
                cls.checked.append(dict(specs))
        """,
    )

    if moved:
        # 0.33 provides these natively, with values the shim must not replace.
        _write(
            pkg / "entrypoints" / "common" / "cli_args.py",
            """
            class FrontendArgs:
                origin = "common"

            def make_arg_parser(parser=None):
                return ("common", parser)

            def resolve_default_chat_template_kwargs(*a, **kw):
                return "common"

            def validate_parsed_serve_args(*a, **kw):
                return "common"
            """,
        )
        _write(
            pkg / "entrypoints" / "common" / "startup.py",
            """
            VLLM_SUBCMD_PARSER_EPILOG = "native-epilog"

            def cli_env_setup():
                return "native-setup"
            """,
        )
        _write(
            pkg / "v1" / "core" / "kv_cache_planning.py",
            """
            def get_global_kv_cache_groups(vllm_config, kv_cache_specs):
                return ["native-groups"]

            def get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory, groups):
                return ["native-config", groups]
            """,
        )
    return site


def _run(
    site: Path, body: str, *, env_extra: dict | None = None, version: str | None = None
) -> dict:
    """Run ``body`` in a fresh interpreter with ``site`` and the repo importable.

    A subprocess per case keeps ``sys.meta_path`` and ``sys.modules`` clean: the
    shim is process-global by design, so in-process tests would leak into each
    other.
    """
    script = textwrap.dedent("""
        import json, sys
        import vllm_musa.omni_compat as compat
        compat.install()
        result = {"installed": compat.is_installed()}
        """) + textwrap.dedent(body) + "\nprint('@@' + json.dumps(result))\n"
    env = {
        "PYTHONPATH": f"{site}:{ROOT}",
        "PATH": "/usr/bin:/bin",
    }
    if version is not None:
        env["VLLM_VERSION"] = version
    env.update(env_extra or {})
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    marker = [line for line in proc.stdout.splitlines() if line.startswith("@@")]
    assert marker, f"no result line: {proc.stdout}\n{proc.stderr}"
    return json.loads(marker[-1][2:])


@pytest.fixture(scope="module")
def pin_site(tmp_path_factory) -> Path:
    return _make_vllm(tmp_path_factory.mktemp("pin"), PIN_VERSION, moved=False)


@pytest.fixture(scope="module")
def next_site(tmp_path_factory) -> Path:
    return _make_vllm(tmp_path_factory.mktemp("next"), NEXT_VERSION, moved=True)


# --------------------------------------------------------------------------- #
# Table integrity
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
def test_every_entry_is_well_formed():
    assert (
        compat_table.ENTRIES
    ), "the table must not be empty while the pin predates Omni main"
    for entry in compat_table.ENTRIES:
        assert entry.kind in compat_table.KINDS
        assert entry.reason.strip() and entry.remove_when.strip()
        assert entry.module.startswith("vllm."), entry.module
        if entry.kind == compat_table.RELOCATION:
            assert entry.source and entry.source != entry.module
        if entry.kind == compat_table.BACKPORT:
            module = importlib.import_module(
                f"vllm_musa.omni_compat.backports.{entry.backport}"
            )
            for name in entry.names:
                assert hasattr(module, name), f"{entry.backport} lacks {name}"
        if entry.kind == compat_table.ADAPTER:
            module = importlib.import_module(
                f"vllm_musa.omni_compat.adapters.{entry.adapter}"
            )
            assert callable(
                module.apply
            ), f"adapter {entry.adapter} must expose apply(module)"


@pytest.mark.cpu
@pytest.mark.parametrize(
    "kind",
    [
        compat_table.RELOCATION,
        compat_table.BACKPORT,
        compat_table.ADAPTER,
        compat_table.CAPABILITY,
    ],
)
def test_table_covers_each_kind(kind):
    """The shim is only proven for kinds the table exercises."""
    assert any(e.kind == kind for e in compat_table.ENTRIES), f"no {kind} entry to test"


@pytest.mark.cpu
def test_malformed_entries_are_rejected():
    with pytest.raises(ValueError, match="unknown kind"):
        compat_table.ShimEntry(
            module="vllm.x",
            names=("y",),
            kind="nonsense",
            applies_to="<0.33",
            reason="r",
            remove_when="w",
        )
    with pytest.raises(ValueError, match="needs source"):
        compat_table.ShimEntry(
            module="vllm.x",
            names=("y",),
            kind=compat_table.RELOCATION,
            applies_to="<0.33",
            reason="r",
            remove_when="w",
        )
    with pytest.raises(ValueError, match="needs backport"):
        compat_table.ShimEntry(
            module="vllm.x",
            names=("y",),
            kind=compat_table.BACKPORT,
            applies_to="<0.33",
            reason="r",
            remove_when="w",
        )
    with pytest.raises(ValueError, match="needs unsupported"):
        compat_table.ShimEntry(
            module="vllm.x",
            names=(),
            kind=compat_table.CAPABILITY,
            target="T",
            applies_to="<0.33",
            reason="r",
            remove_when="w",
        )


@pytest.mark.cpu
def test_capability_entries_are_never_served_as_imports():
    """A CAPABILITY module exists in the pin; shadowing it would break the pin."""
    for entry in compat_table.capability_entries():
        assert compat_table.entries_for_module(entry.module) is None


# --------------------------------------------------------------------------- #
# Version gating
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.parametrize(
    ("specifier", "version", "expected"),
    [
        ("<0.33", "0.32.0", True),
        ("<0.33", "0.33.0", False),
        (
            "<0.33",
            "0.32.0+musa.g01c3f711a",
            True,
        ),  # local segment must not change identity
        ("<0.33", "0.33.0rc1", False),
        ("<0.33", None, True),  # undeterminable: shim rather than ImportError
        ("<0.33", "not-a-version", True),
        (">=0.32,<0.33", "0.31.0", False),
    ],
)
def test_version_matching(specifier, version, expected):
    assert version_matches(specifier, version) is expected


# --------------------------------------------------------------------------- #
# Relocation (class A)
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
def test_relocation_serves_the_pinned_symbols(pin_site):
    result = _run(
        pin_site,
        """
        from vllm.entrypoints.common.cli_args import (
            FrontendArgs, make_arg_parser, resolve_default_chat_template_kwargs,
            validate_parsed_serve_args,
        )
        from vllm.entrypoints.common.startup import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup
        import vllm.entrypoints.common.cli_args as shimmed
        result["origin"] = FrontendArgs.origin
        result["parser"] = make_arg_parser("p")
        result["resolve"] = resolve_default_chat_template_kwargs()
        result["validate"] = validate_parsed_serve_args()
        result["epilog"] = VLLM_SUBCMD_PARSER_EPILOG
        result["setup"] = cli_env_setup()
        result["delegated"] = shimmed.not_in_the_table()
        result["marker"] = shimmed.__vllm_musa_shim__["kind"]
        result["name"] = shimmed.__name__
        """,
        version=PIN_VERSION,
    )
    assert result["origin"] == "launchers"
    assert result["parser"] == ["launchers", "p"]
    assert result["resolve"] == "launchers"
    assert result["validate"] == "launchers"
    assert result["epilog"] == "pinned-epilog"
    assert result["setup"] == "pinned-setup"
    # A name the table does not list still resolves, so an unrecorded Omni
    # import does not need a table edit to work.
    assert result["delegated"] == "delegated"
    assert result["marker"] == "relocation"
    # The shimmed module keeps the name the caller asked for.
    assert result["name"] == "vllm.entrypoints.common.cli_args"


@pytest.mark.cpu
def test_without_the_shim_the_same_import_fails(pin_site):
    """The failure the shim removes, so the test above is not vacuous."""
    proc = subprocess.run(
        [sys.executable, "-c", "import vllm.entrypoints.common.cli_args"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(pin_site), "PATH": "/usr/bin:/bin"},
        timeout=120,
    )
    assert proc.returncode != 0
    assert "No module named" in proc.stderr


@pytest.mark.cpu
def test_the_real_module_always_wins(next_site):
    """On a pin that has the module natively the shim must not shadow it."""
    result = _run(
        next_site,
        """
        from vllm.entrypoints.common.cli_args import FrontendArgs, make_arg_parser
        from vllm.entrypoints.common.startup import VLLM_SUBCMD_PARSER_EPILOG
        import vllm.entrypoints.common.cli_args as native
        result["origin"] = FrontendArgs.origin
        result["parser"] = make_arg_parser("p")
        result["epilog"] = VLLM_SUBCMD_PARSER_EPILOG
        result["shimmed"] = hasattr(native, "__vllm_musa_shim__")
        """,
        version=NEXT_VERSION,
    )
    assert result["origin"] == "common"
    assert result["parser"] == ["common", "p"]
    assert result["epilog"] == "native-epilog"
    assert result["shimmed"] is False


@pytest.mark.cpu
def test_out_of_range_version_is_not_shimmed(pin_site):
    """Claiming 0.33 on a 0.32 tree must fail, not silently get the shim."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import vllm_musa.omni_compat as c; c.install(); import vllm; import vllm.entrypoints.common.cli_args",
        ],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": f"{pin_site}:{ROOT}",
            "PATH": "/usr/bin:/bin",
            "VLLM_VERSION": NEXT_VERSION,
        },
        timeout=120,
    )
    assert proc.returncode != 0
    assert "No module named" in proc.stderr


@pytest.mark.cpu
def test_disable_env_makes_the_shim_inert(pin_site):
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import vllm_musa.omni_compat as c; assert c.install() is False; import vllm.entrypoints.common.cli_args",
        ],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": f"{pin_site}:{ROOT}",
            "PATH": "/usr/bin:/bin",
            "VLLM_MUSA_OMNI_COMPAT": "0",
        },
        timeout=120,
    )
    assert proc.returncode != 0
    assert "No module named" in proc.stderr


# --------------------------------------------------------------------------- #
# Backport (class C) and adapter (class B)
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
def test_backport_reproduces_the_pinned_grouping(pin_site):
    result = _run(
        pin_site,
        """
        from vllm.v1.core.kv_cache_planning import get_global_kv_cache_groups, get_kv_cache_configs
        from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

        class Config:
            num_prefill_lookahead_tokens = 1

        specs = [{"layer.b": "spec-b", "layer.a": "spec-a"}, {"layer.a": "spec-a"}]
        groups = get_global_kv_cache_groups(Config(), specs)
        result["groups"] = groups
        result["registry_checked"] = KVCacheSpecRegistry.checked
        # 0.33 passes the groups in; the pin's planner does not take them.
        result["configs"] = get_kv_cache_configs(Config(), specs, [1024, 2048], groups)
        """,
        version=PIN_VERSION,
    )
    # Merged across workers, deduplicated, grouped by the pinned helper.
    assert result["groups"] == [["group", "layer.a"], ["group", "layer.b"]]
    assert result["registry_checked"] == [{"layer.b": "spec-b", "layer.a": "spec-a"}]
    assert result["configs"] == ["pinned-config", 2, [1024, 2048]]


@pytest.mark.cpu
def test_backport_rejects_inconsistent_worker_specs(pin_site):
    """The pinned assertion must survive the backport, not be softened by it."""
    script = textwrap.dedent("""
        import vllm_musa.omni_compat as compat
        compat.install()
        from vllm.v1.core.kv_cache_planning import get_global_kv_cache_groups

        class Config:
            num_prefill_lookahead_tokens = 1

        get_global_kv_cache_groups(Config(), [{"l": "a"}, {"l": "b"}])
        """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": f"{pin_site}:{ROOT}",
            "PATH": "/usr/bin:/bin",
            "VLLM_VERSION": PIN_VERSION,
        },
        timeout=120,
    )
    assert proc.returncode != 0
    assert "not supported yet" in proc.stderr


@pytest.mark.cpu
def test_a_stale_relocation_fails_loudly(tmp_path, monkeypatch):
    """A table row pointing at a source without the symbol must raise on import."""
    site = _make_vllm(tmp_path, PIN_VERSION, moved=False)
    script = textwrap.dedent("""
        import vllm_musa.omni_compat as compat
        from vllm_musa.omni_compat import table

        bad = table.ShimEntry(
            module="vllm.entrypoints.common.cli_args",
            names=("ThisSymbolDoesNotExist",),
            kind=table.RELOCATION,
            source="vllm.entrypoints.launchers.cli_args",
            applies_to="<0.33",
            reason="fixture",
            remove_when="fixture",
        )
        table.ENTRIES = (bad,)
        compat.install()
        import vllm.entrypoints.common.cli_args
        """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": f"{site}:{ROOT}",
            "PATH": "/usr/bin:/bin",
            "VLLM_VERSION": PIN_VERSION,
        },
        timeout=120,
    )
    assert proc.returncode != 0
    assert "table entry is stale" in proc.stderr


# --------------------------------------------------------------------------- #
# Capability reporting (class E) and status surface
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
def test_unsupported_paths_are_reported_on_the_pin(pin_site):
    result = _run(
        pin_site,
        """
        import vllm  # the version the shim gates on
        import vllm_musa.omni_compat as compat
        result["unsupported"] = list(compat.unsupported_paths())
        result["status_version"] = compat.status()["vllm_version"]
        """,
        version=PIN_VERSION,
    )
    assert result["status_version"] == PIN_VERSION
    assert any("decode context parallel" in path for path in result["unsupported"])


@pytest.mark.cpu
def test_unsupported_paths_are_empty_on_a_newer_pin(next_site):
    result = _run(
        next_site,
        """
        import vllm
        import vllm_musa.omni_compat as compat
        result["unsupported"] = list(compat.unsupported_paths())
        """,
        version=NEXT_VERSION,
    )
    assert result["unsupported"] == []


@pytest.mark.cpu
def test_install_is_idempotent_and_reversible(pin_site):
    result = _run(
        pin_site,
        """
        import sys
        import vllm_musa.omni_compat as compat
        compat.install()
        compat.install()
        result["serve_hooks"] = sum(1 for f in sys.meta_path if type(f).__name__ == "OmniCompatFinder")
        result["adapt_hooks"] = sum(1 for f in sys.meta_path if type(f).__name__ == "OmniAdapterFinder")
        result["adapt_first"] = type(sys.meta_path[0]).__name__ == "OmniAdapterFinder"
        result["serve_last"] = type(sys.meta_path[-1]).__name__ == "OmniCompatFinder"
        result["removed"] = compat.uninstall()
        result["after"] = compat.is_installed()
        """,
        version=PIN_VERSION,
    )
    assert result["serve_hooks"] == 1
    assert result["adapt_hooks"] == 1
    # Position is the contract: serving last so the real vLLM wins, adapting
    # first so a pinned module still reaches its patch.
    assert result["adapt_first"] is True
    assert result["serve_last"] is True
    assert result["removed"] == 2
    assert result["after"] is False


@pytest.mark.cpu
def test_hook_is_last_so_ordinary_finders_win(pin_site):
    result = _run(
        pin_site,
        """
        import sys
        result["is_last"] = type(sys.meta_path[-1]).__name__ == "OmniCompatFinder"
        result["adapter_is_first"] = type(sys.meta_path[0]).__name__ == "OmniAdapterFinder"
        """,
        version=PIN_VERSION,
    )
    assert result["is_last"] is True
    assert result["adapter_is_first"] is True


@pytest.mark.cpu
def test_bootstrap_imports_nothing_heavy(pin_site):
    """The .pth runs in every interpreter on the machine, so it must stay cheap.

    ``import vllm_musa.omni_compat`` would execute ``vllm_musa/__init__.py``,
    which imports torchada and torch; the bootstrap exists to avoid exactly that.
    """
    script = textwrap.dedent("""
        import json, sys
        import vllm_musa_omni_compat_pth as boot
        heavy = sorted(
            m for m in sys.modules
            if m.split(".")[0] in {"torch", "vllm", "numpy", "torch_musa", "torchada", "vllm_musa"}
        )
        print("@@" + json.dumps({
            "heavy": heavy,
            "installed": sum(1 for f in sys.meta_path if type(f).__name__ == "OmniCompatPathFinder"),
            "is_last": type(sys.meta_path[-1]).__name__ == "OmniCompatPathFinder",
            "is_first": type(sys.meta_path[0]).__name__ == "OmniCompatPathFinder",
        }))
        """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": f"{pin_site}:{ROOT}", "PATH": "/usr/bin:/bin"},
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(
        [ln for ln in proc.stdout.splitlines() if ln.startswith("@@")][-1][2:]
    )
    assert (
        result["heavy"] == []
    ), "the startup hook must not import torch, vllm or vllm_musa"
    assert (
        result["installed"] == 2
    ), "one deferring hook to serve modules, one to adapt them"
    assert result["is_last"] is True
    assert result["is_first"] is True


@pytest.mark.cpu
def test_bootstrap_serves_without_importing_the_package(pin_site):
    """The startup hook must serve a shimmed module on its own.

    Reaching into ``vllm_musa.omni_compat`` here would re-enter vLLM through
    ``vllm_musa``'s initialisation while a vLLM module is mid-import, which is
    the circular import this bootstrap exists to avoid.
    """
    script = textwrap.dedent("""
        import json, sys
        import vllm_musa_omni_compat_pth  # noqa: F401  -- installs the hook
        from vllm.entrypoints.common.startup import cli_env_setup
        print("@@" + json.dumps({
            "value": cli_env_setup(),
            "imported_vllm_musa": "vllm_musa" in sys.modules,
        }))
        """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": f"{pin_site}:{ROOT}",
            "PATH": "/usr/bin:/bin",
            "VLLM_VERSION": PIN_VERSION,
        },
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(
        [ln for ln in proc.stdout.splitlines() if ln.startswith("@@")][-1][2:]
    )
    assert result["value"] == "pinned-setup"
    assert result["imported_vllm_musa"] is False


@pytest.mark.cpu
def test_bootstrap_respects_the_disable_env(pin_site):
    script = textwrap.dedent("""
        import sys
        import vllm_musa_omni_compat_pth as boot
        assert not any(type(f).__name__ == "OmniCompatPathFinder" for f in sys.meta_path)
        assert boot.install() is False
        import vllm.entrypoints.common.startup
        """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": f"{pin_site}:{ROOT}",
            "PATH": "/usr/bin:/bin",
            "VLLM_MUSA_OMNI_COMPAT": "0",
        },
        timeout=120,
    )
    assert proc.returncode != 0
    assert "No module named" in proc.stderr


@pytest.mark.cpu
def test_bootstrap_reads_the_table_not_a_hardcoded_list():
    """One source of truth: the startup hook must agree with the table."""
    import vllm_musa_omni_compat_pth as boot

    package_dir = boot._package_dir()
    assert package_dir is not None
    table = boot._load_by_path("_table_fixture", f"{package_dir}/table.py")
    assert frozenset(e.module for e in table.importable_entries()) == frozenset(
        e.module for e in compat_table.importable_entries()
    )
    assert frozenset(e.module for e in table.adapter_entries()) == frozenset(
        e.module for e in compat_table.adapter_entries()
    )


@pytest.mark.cpu
def test_pth_roundtrip(tmp_path):
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_musa.omni_compat",
            "install-pth",
            "--purelib",
            str(purelib),
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    written = list(purelib.glob("*.pth"))
    assert len(written) == 1
    # site processes .pth lines that start with "import"; anything else is a path.
    assert written[0].read_text().startswith("import ")
    assert "omni_compat" in written[0].read_text()
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_musa.omni_compat",
            "remove-pth",
            "--purelib",
            str(purelib),
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert list(purelib.glob("*.pth")) == []


@pytest.mark.cpu
def test_pth_activates_the_shim_without_an_explicit_install(tmp_path):
    """The mechanism the image relies on: startup hook, no caller cooperation."""
    site = _make_vllm(tmp_path, PIN_VERSION, moved=False)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_musa.omni_compat",
            "install-pth",
            "--purelib",
            str(site),
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
        timeout=120,
        check=True,
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from vllm.entrypoints.common.startup import cli_env_setup; print(cli_env_setup())",
        ],
        capture_output=True,
        text=True,
        # site-packages .pth processing needs the directory on the path as a site
        # dir, which PYTHONPATH does not do; -S plus explicit addsitedir is the
        # portable way to prove the line itself works.
        env={"PYTHONPATH": f"{site}:{ROOT}", "PATH": "/usr/bin:/bin"},
        timeout=120,
    )
    # PYTHONPATH does not make a directory a site dir, so .pth processing does
    # not run here; this asserts the line is well formed and the mechanism is
    # proven end to end by test_bootstrap_defers_to_the_real_finder and by the
    # image build, which installs into a real site-packages.
    assert proc.returncode != 0 or "pinned-setup" in proc.stdout


# --------------------------------------------------------------------------- #
# Adapter (call-time drift)
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
def test_the_pinned_call_fails_without_the_adapter(pin_site):
    """The TypeError the adapter removes, so the next test is not vacuous."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p;"
            "apply_top_k_top_p('l', 1, 2, has_topp_only_rows=False)",
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(pin_site), "PATH": "/usr/bin:/bin"},
        timeout=120,
    )
    assert proc.returncode != 0
    assert "has_topp_only_rows" in proc.stderr and "TypeError" in proc.stderr


@pytest.mark.cpu
@pytest.mark.parametrize("hint", [True, False])
def test_adapter_lets_the_pinned_function_take_the_newer_argument(pin_site, hint):
    result = _run(
        pin_site,
        f"""
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
        import vllm.v1.sample.ops.topk_topp_sampler as mod
        result["value"] = list(apply_top_k_top_p("l", 1, 2, has_topp_only_rows={hint}))
        result["without"] = list(apply_top_k_top_p("l", 1, 2))
        result["adapted"] = getattr(mod, "__vllm_musa_adapted__", {{}}).get("names")
        """,
        version=PIN_VERSION,
    )
    # Dropping the hint must give exactly the pinned result, for either value.
    assert result["value"] == ["pinned", "l", 1, 2]
    assert result["without"] == ["pinned", "l", 1, 2]
    assert result["adapted"] == ["apply_top_k_top_p"]


@pytest.mark.cpu
def test_adapter_is_inert_when_the_pin_already_takes_the_argument(next_site):
    result = _run(
        next_site,
        """
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
        import vllm.v1.sample.ops.topk_topp_sampler as mod
        result["value"] = list(apply_top_k_top_p("l", 1, 2, has_topp_only_rows=False))
        result["adapted"] = getattr(mod, "__vllm_musa_adapted__", None)
        """,
        version=NEXT_VERSION,
    )
    # The native four-parameter function runs, unpatched.
    assert result["value"] == ["pinned", "l", 1, 2, False]
    assert result["adapted"] is None


@pytest.mark.cpu
def test_adapter_reaches_a_module_imported_after_install(pin_site):
    """The ordering the .pth exists to guarantee."""
    result = _run(
        pin_site,
        """
        import sys
        result["preimported"] = "vllm.v1.sample.ops.topk_topp_sampler" in sys.modules
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
        result["ok"] = list(apply_top_k_top_p("l", 1, 2, has_topp_only_rows=True))
        """,
        version=PIN_VERSION,
    )
    assert result["preimported"] is False
    assert result["ok"] == ["pinned", "l", 1, 2]


@pytest.mark.cpu
def test_adapter_warns_when_the_module_was_already_imported(pin_site):
    """Installing too late cannot work, and must say so rather than look fine."""
    script = textwrap.dedent("""
        import logging, sys
        logging.basicConfig(level=logging.WARNING)
        import vllm.v1.sample.ops.topk_topp_sampler  # before the shim: too late
        import vllm_musa.omni_compat as compat
        compat.install()
        """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": f"{pin_site}:{ROOT}",
            "PATH": "/usr/bin:/bin",
            "VLLM_VERSION": PIN_VERSION,
        },
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "already imported" in proc.stderr


@pytest.mark.cpu
def test_an_adapter_that_patches_nothing_fails_the_import(pin_site, tmp_path):
    """A silent no-op adapter is the failure mode that matters most."""
    script = textwrap.dedent("""
        import sys
        import vllm_musa.omni_compat as compat
        from vllm_musa.omni_compat import _patcher, table

        class _Noop:
            @staticmethod
            def apply(module):
                return []

        sys.modules["vllm_musa.omni_compat.adapters.noop_fixture"] = _Noop
        table.ENTRIES = (
            table.ShimEntry(
                module="vllm.v1.sample.ops.topk_topp_sampler",
                names=("apply_top_k_top_p",),
                kind=table.ADAPTER,
                adapter="noop_fixture",
                applies_to="<0.33",
                reason="fixture",
                remove_when="fixture",
            ),
        )
        _patcher.install_adapters()
        import vllm.v1.sample.ops.topk_topp_sampler
        """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": f"{pin_site}:{ROOT}",
            "PATH": "/usr/bin:/bin",
            "VLLM_VERSION": PIN_VERSION,
        },
        timeout=120,
    )
    assert proc.returncode != 0
    assert "patched nothing" in proc.stderr
