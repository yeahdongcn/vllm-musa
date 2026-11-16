# Running vLLM-Omni on a pinned vLLM-MUSA

vLLM-MUSA pins one upstream vLLM commit per release branch
(`third_party/PINS`) and validates that commit on MUSA. vLLM-Omni declares **no**
`vllm` requirement, tracks vLLM main, and imports vLLM internals directly. The
two projects move on different clocks, so an Omni tree is usually newer than the
pin and the mismatch is not a dependency-resolution problem: pip will happily
produce an image that imports today and dies in a model tomorrow.

This document records what comparable projects do, what the drift actually looks
like when measured, and the mechanism this repository uses to absorb it.

## What vllm-ascend does, and what transfers

Read at `vllm-project/vllm-ascend@main` and `vllm-project/vllm-omni@main`
(2026-11-16). Five mechanisms, which are worth separating because only three of
them apply to us.

1. **Version-matched releases.** Each vllm-ascend release is `vX.Y.Z[rcN]`
   matching the vLLM release of the same number, with a published compatibility
   matrix (`docs/source/community/versioning_policy.md`) covering vLLM, Python, the Ascend
   toolkit, torch-npu, Triton Ascend and Mooncake. Branches are `releases/vX.Y.Z`,
   maintained for 2-3 minor versions; `main` tracks vLLM main plus the latest 1-2
   releases.
2. **In-code version gating.** `vllm_ascend.utils.vllm_version_is(target)` with a
   `VLLM_VERSION` environment override, used in 57 files, so one branch supports
   several vLLM versions. It is `functools.cache`d, disabled for Dynamo, and
   strips PEP 440 local segments so a `0.30.0+empty` build keeps its release
   identity.
3. **An explicit patch ledger.** `vllm_ascend/patch/{platform,worker}/` with a
   docstring entry per patch: target, why, how, related PR, and a *future plan*
   saying when to delete it.
4. **CI that finds drift first.** `schedule_main2main.yaml` tests ascend main
   against vLLM main HEAD and opens a PR; `_schedule_nightly_pair_image_build.yaml`
   builds base+plugin image pairs; `schedule_e2e_upstream_test.yaml` runs e2e
   against upstream.
5. **How Omni consumes it.** `vllm-omni/docker/Dockerfile.npu` is simply
   `FROM quay.io/atlas-ci/vllm-ascend:${VLLM_ASCEND_VERSION}` with
   `VLLM_ASCEND_VERSION=v0.33.0`. Because the plugin's version *is* the vLLM
   version, Omni picks a matching base with one tag; its CUDA, ROCm and XPU
   Dockerfiles do the same against `vllm/vllm-openai:v0.33.0`.

(1) is a release-policy decision, and vllm-musa already has per-minor
`vX.Y.0-dev` branches; the gap is a *published, version-matched image* an Omni
Dockerfile could target, not the branch scheme. (3) and (4) transfer directly and
this document follows (3).

**(2) does not transfer**, and that is the crux. `vllm_version_is` adapts
*vllm-ascend's own* code. We cannot put version gates inside vLLM-Omni, which we
do not own and which is the tree that has to bend. The equivalent for a
third-party consumer is an **import-level compatibility shim** that makes the
pinned vLLM expose the surface a newer Omni expects. That is what
`vllm_musa/omni_compat/` is.

## The drift, measured

`tools/check_omni_compat.py` against extracted vLLM sources, no GPU:

| Omni tree | vLLM | vllm imports | unshimmed | module attributes | missing |
|---|---|---|---|---|---|
| `main` e937f1c9e | 0.33.0 | 4336 | 0 | 181 | 0 |
| `main` e937f1c9e | 0.32.0 (the pin) | 4336 | 8 | 181 | 1 |
| `01c3f711a` (0.32-era) | 0.32.0 | 4132 | 0 | 177 | 0 |

Omni main against 0.33 is clean and the 0.32-era Omni against 0.32 is clean, so
every finding below is exactly the version gap.

### Six classes, not one

The drift is not uniform, and the mechanism has to match the class. Measured on
Omni main against the 0.32 pin:

| Class | What drifted | Count | Mechanism | Found by |
|---|---|---|---|---|
| **A** | module relocated, symbols unchanged | 6 symbols | alias module (`RELOCATION`) | import audit |
| **B** | symbol absent, signature also changed | 2 symbols | reimplementation (`BACKPORT`) | import audit |
| **C** | symbol absent from an existing module, read as an attribute | 1 symbol | patch the pinned module (`ADAPTER`) | **module-attribute audit** |
| **D** | symbol present, Omni passes a newer argument | 1 symbol | patch the pinned function (`ADAPTER`) | **signature diff** |
| **E** | same name, same arity, changed meaning | 2 symbols | declare unsupported (`CAPABILITY`) | **signature diff** |
| **F** | newer Python feature | 1 | the existing 3.10 patch, reported | feature scan |

Classes C, D and E are the interesting ones, because an import-only gate reports
**zero problems** for all three and the run then fails at a model:

* **C** — `vllm_omni/utils/mm_prefix.py` does
  `from vllm.v1.attention.backends import utils` (resolves on any pin, the module
  exists) and then reads `utils.compute_mm_prefix_ranges`, which 0.33 added.
  `vllm_omni/patch.py` calls that at **import time**, so on the 0.32 pin
  `import vllm_omni` raises `AttributeError` and *no model runs at all*. This was
  found on hardware, after the import audit had passed, which is why the tool
  grew a module-attribute pass.
* **D** — `apply_top_k_top_p(..., has_topp_only_rows=...)` from the Qwen3-Omni
  and MiniCPM-o 4.5 talker samplers. 0.33 added the parameter; the pinned
  three-parameter function raises `TypeError` on the first talker sampling step.
  Qwen3-Omni is in the supported matrix.
* **E** — `VllmConfig.adjust_dcp_kv_cache_interleave_size` has one parameter in
  both versions, but the pin wants a `KVCacheConfig` and reads
  `.kv_cache_groups` off it while 0.33 wants the group list itself. Omni main
  passes the list. It is *latent*: both bodies return early unless
  `decode_context_parallel_size > 1` with a NixlConnector, so an ordinary
  single-card run never reaches the `AttributeError`.

**"The imports resolve" was never the right gate.** That is the single most
useful conclusion of this investigation, and the previous overlay
(`xd/omni-032-overlay`) is the evidence: its gate reported 0 missing symbols for
a tree with three runtime-breaking call sites.

## The mechanism

### One table

`vllm_musa/omni_compat/table.py` holds one `ShimEntry` per drift item: the module
Omni imports, the symbols, the kind, the vLLM versions it applies to, the reason,
and the condition that makes the entry dead. Supporting a new Omni commit is a
new row plus a test, never a change to the installer.

```python
ShimEntry(
    module="vllm.entrypoints.common.cli_args",
    names=("FrontendArgs", "make_arg_parser", ...),
    kind=RELOCATION,
    source="vllm.entrypoints.launchers.cli_args",
    applies_to="<0.33",
    reason="0.33 moved the serve CLI argument surface ...",
    remove_when="the pin is >= 0.33, where vllm.entrypoints.common.cli_args exists.",
)
```

`applies_to` is matched against the installed `vllm.__version__` (with
`VLLM_VERSION` as an override, as vllm-ascend does, and PEP 440 local segments
stripped so `0.32.0+musa` keeps its release identity). A pin bump that removes
the drift therefore disables the entry with no code change.

### Two hook positions, for opposite reasons

* **Serving** a module the pin lacks: the hook is **appended** to
  `sys.meta_path`, so Python consults it only after every ordinary finder has
  failed. The real vLLM always wins, and the shim needs no "does this exist?"
  probe that could recurse into itself.
* **Adapting** a module the pin has: the hook is **inserted first**, delegates to
  `PathFinder` for the real module, and wraps its loader to patch the module
  right after it executes. An ordinary finder would otherwise satisfy the import
  before the hook was consulted.

Timing is not negotiable. `vllm_omni/entrypoints/cli/main.py` imports moved vLLM
modules in its first statement, and `vllm_omni/patch.py` reads a drifted
attribute at import time, both long before vLLM loads any plugin entry point. So
the hook is installed by the interpreter itself, from a `.pth`:

```bash
python -m vllm_musa.omni_compat install-pth   # what docker/omni.Dockerfile runs
python -m vllm_musa.omni_compat status        # what is active, as JSON
VLLM_MUSA_OMNI_COMPAT=0 ...                   # make an installed hook inert
```

The `.pth` is **not** installed by `pip install vllm-musa`: a shim that changes
the meaning of `import vllm.*` belongs to images that run Omni, so activation is
explicit and scoped to `docker/omni.Dockerfile`.

### The bootstrap imports nothing

`vllm_musa_omni_compat_pth.py` is a top-level module, standard library only, and
it never imports `vllm_musa`. Two reasons, the second discovered the hard way:

* **Cost** — a `.pth` runs in *every* interpreter on the machine, and
  `vllm_musa/__init__.py` pulls in `torchada` and `torch`. A health probe or a
  `pip` subprocess must not pay seconds of torch import for a hook it never uses.
* **Correctness** — the hook fires *during* a vLLM import. Re-entering vLLM
  through `vllm_musa`'s initialisation at that moment raises
  `cannot import name 'rocm_aiter_ops' from partially initialized module
  'vllm._aiter_ops'`. The first version of the bootstrap did exactly that and
  broke every import of the adapted modules.

So the table, the backports and the adapters are all loaded **by file path**,
from the installed package directory located with `find_spec` without executing
it. Each is dependency-free at module level, and the adapters import vLLM only
inside the functions they install. A test asserts the startup hook imports no
`torch`, `vllm` or `vllm_musa`.

### A capability entry is an honest answer

Class E cannot be fixed by aliasing: the pinned API means something different
under the same name, so any bridge would return quietly wrong values. Those rows
are `CAPABILITY`: they name the unsupported feature path, the audit reports it,
and `omni_compat.unsupported_paths()` lists it at runtime. Current entries on the
0.32 pin:

* the worker_v2 autoregressive model runner under data parallelism
  (`vllm_omni/worker_v2/omni_model_runner.py`; diffusion pipelines do not use it);
* decode context parallelism with a NixlConnector KV transfer.

Writing a wrapper that guesses at either would be worse than saying so.

## Finding the next drift

`tools/check_omni_compat.py` runs four passes on source alone, in seconds, with
no GPU and no install:

```bash
# in an image, against the installed vLLM
python3 tools/check_omni_compat.py --omni-root /vllm-workspace/vllm-omni

# in CI, against extracted sources, with the signature diff
python3 tools/check_omni_compat.py \
    --omni-root ./vllm-omni \
    --vllm-root ./vllm-0.32/vllm --vllm-version 0.32.0 \
    --target-vllm ./vllm-0.33/vllm --target-version 0.33.0
```

| Pass | Question | Fails the build |
|---|---|---|
| `imports` | does every `from vllm.x import y` resolve on the pin? | an unshimmed miss outside `TYPE_CHECKING` |
| `attrs` | does every attribute Omni reads off a vLLM module exist? | a missing attribute with no shim |
| `staleness` | is every active entry still needed on this pin? | an entry the pin has made dead |
| `semantics` | did a symbol Omni calls change signature between the pin and the vLLM Omni targets? | a call that passes what the pin cannot take |

Two details decide whether the output is usable rather than noise, and both were
false-positive bugs first:

* `vllm/distributed/__init__.py` is nothing but `from .x import *`, and
  `vllm/platforms/__init__.py` resolves `current_platform` through a module-level
  `__getattr__`. Without resolving star re-exports and trusting dynamic modules
  the tool reported **156** missing symbols instead of 8.
* a plain-name call only means the vLLM function in files that imported it from
  vLLM. `vllm_omni/platforms/npu/_310p/patch/qwen3_tts.py` has its own
  `apply_top_k_top_p`; attributing its arguments to vLLM's was wrong.

Attribute calls are reported but never failed: the receiver cannot be resolved
statically, and vLLM-Omni has its own `post_forward` hook protocol that name
matching alone cannot distinguish from `ActiveKVConnector.post_forward`.

**What no static pass can see** is a function that keeps its name, arity and
annotations and changes behaviour. That is what the hardware smoke is for, and
why `docker/omni.Dockerfile` is not the last gate.

## The image

```bash
bash docker/build_omni_image.sh --base <pinned-vllm-musa-image> --omni-ref origin/main
bash docker/build_omni_image.sh --base <image> --omni-ref origin/main --dry-run
```

Stages, each of which fails the build rather than deferring a surprise:

1. **Source** — `git archive` export from a local clone (immutable: no dirty
   tree, no moving branch) or a clone in-build.
2. **Base contract** — read the base's real `vllm`, `torch` and Python versions
   and record them. A base bump fails here, before any Omni code is installed.
3. **Shim staging** — `docker/omni/stage_shim.py` copies this checkout's
   pure-Python shim into the base's installed `vllm_musa` when the base predates
   it. A no-op once the shim ships in a release, and the reason an Omni ref can
   need a *newer* shim than its base was built with. The top-level bootstrap goes
   to `purelib`, not next to the package: vllm-musa is installed through a PEP
   660 strict finder that maps only the `vllm_musa` package, so a module dropped
   beside it is never on `sys.path`.
4. **Gate 1** — the four audit passes, on source, before installing anything.
5. **Python floor** — mirror vllm-musa's own 3.10 decision onto Omni's
   `requires-python` (`--policy strict` refuses instead). Narrower than
   `--ignore-requires-python`, which would also admit unrelated 3.11-only deps.
6. **Requirement exclusions** — `docker/omni/requirement-exclusions.txt`, one
   distribution per line with its reason. Currently `cosmos-guardrail`, which
   wants `numpy>=1.26.4` against the pinned `numpy==1.26.0`; only Cosmos3 uses
   it, through a `try/except ImportError`.
7. **Install** — `pip install -e` with every package the base installed frozen as
   a constraint, so Omni may *add* packages and any attempt to *move* one is a
   `ResolutionImpossible` at build time. Verified afterwards by comparing the
   two freezes: "foundation unchanged: 177 pins held".
8. **Gate 2 + 3** — the audit against the installed trees, then
   `docker/omni/verify_shim.py`, which imports every served module and checks
   every adapter actually patched. `--driverless` reports a module that needs a
   GPU as skipped; the build uses it, hardware does not.

`import vllm_omni` is deliberately **not** a build step: it activates the MUSA
platform plugin, which queries device capability through `torch_musa` and raises
without a driver. It belongs to the hardware run:

```bash
python /opt/omni-compat/verify_shim.py --require-omni
```

## Adding support for a newer Omni ref

1. Run the audit against the candidate ref with `--target-vllm` pointed at the
   vLLM that ref targets. The four passes name every new drift item and its class.
2. For each one, add a `ShimEntry`:
   * moved module, same symbols -> `RELOCATION` with `source=`;
   * module absent -> `BACKPORT` plus a module under `backports/` built from
     pinned APIs, with its docstring arguing the equivalence;
   * symbol absent from, or newer than, an existing module -> `ADAPTER` plus a
     module under `adapters/` exposing `apply(module) -> list[str]`;
   * meaning changed under the same shape -> `CAPABILITY` naming the unsupported
     path. Do not write a wrapper that guesses.
3. Add the test. `tests/test_omni_compat.py` pairs every behaviour with the
   opposite case, because a gate that cannot fail is worse than no gate.
4. Build the image and run the hardware smoke. The static passes cannot see
   behaviour drift.

When the pin moves forward, the entries whose `applies_to` no longer matches go
inert on their own, and the staleness pass tells you to delete them.

## Verified

Built and run on one S5000 (driver 5.2.0) from the pinned base
`vllm:v0.32.0-ph1-5.2.0-torch2.11.0.post2-20261112`, with vLLM-Omni
`origin/main` e937f1c9e -- an Omni tree that targets vLLM 0.33 on a vLLM 0.32
pin:

```text
[imports]   pass: 4336 vllm imports, 11 shimmed, 0 unshimmed
[attrs]     pass: 181 module attributes used, 2 shimmed, 0 missing
[staleness] pass: 7 entries still needed, 0 stale
[semantics] review: 2578 symbols compared, 0 breaking, 6 changed meaning, 25 latent
foundation unchanged: 177 pins held
verify_shim --require-omni: verdict pass
  vllm 0.32.0, vllm-omni 0.32.0+musa.ge937f1c9e
  hooks: serve, adapt, startup all live
  adapted: compute_mm_prefix_ranges, apply_top_k_top_p
Qwen-Image-2.1 512x512, 8 steps, seed 42, eager: 0.997 s, 388688-byte PNG
```

Adapter behaviour against the real pinned vLLM:
`apply_top_k_top_p` accepts `has_topp_only_rows` and returns a result identical
to the pin's for **both** values of the hint; `compute_mm_prefix_ranges` bridges
to the pinned builder and raises `NotImplementedError` naming the shim for the
two arguments the pin cannot express.

Not run: any CUDA, ROCm, XPU or NPU build; the Qwen3-Omni and MiniCPM-o talker
paths the top-k adapter serves (no staged checkpoint that fits one card); a
multi-card or served-endpoint smoke; the two declared-unsupported paths, which
are declared precisely because they are not usable on this pin.
