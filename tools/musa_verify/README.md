# `tools/musa_verify/` — parallel device-pinned verification harness

A small update→verify harness for the patch-maintenance branch (and any vllm-musa
change). It runs the **unit tests** plus **one functional server smoke per model**,
all **concurrently**, with each smoke pinned to its own MUSA device via
`MUSA_VISIBLE_DEVICES` (e.g. DeepSeek-V2-Lite on device 0, Qwen3-8B on device 1).

## Test environment (pinned)

All MUSA-0300..0309 verification runs in **one** environment, for reproducibility:

- Host `mccxadmin@10.18.32.18`, container **`yeahdongcn70`**.
- Python venv **`/root/.virtualenvs/sglang-0.5.6`** (a `uv` virtualenv; carries
  `mate 0.2.0+mu437torch2.9`, `torchada 0.1.56`, `torch_musa 2.9.0`, `triton 3.2.0`,
  `sglang 0.5.6.post2`). Activate with
  `source /root/.virtualenvs/sglang-0.5.6/bin/activate` (a login shell also prepends it
  to `PATH` via `.bashrc`). Export `MUSA_VENV=/root/.virtualenvs/sglang-0.5.6` so the
  scripts/orchestrators activate it automatically.
- **Do not switch envs between tickets** — mixing envs invalidates cross-ticket comparison.
  The base/system env (vllm 0.20.1, mate 0.1.3) is the M3/dflash line; do not test here.

Two checkout layouts are supported:

- **Normal flow** — `CONTAINER_WS=/ws` is a git checkout you can `git reset --hard` to the
  SHA under test (use when `/ws` is free).
- **Isolated flow** — when `/ws` holds other WIP, ship the branch with
  `git archive <branch> | ssh '… tar -x -C /ws-verify'` and build there
  (`CONTAINER_WS=/ws-verify`). Building re-points the venv's editable `vllm-musa`; restore
  the prior line with `cd /ws && pip install -e . --no-build-isolation --no-deps`.
- Build with `--no-deps` (the venv already satisfies the pins; avoids the broken
  `mate 0.2.0` mirror metadata).

## Files

| File | Runs where | Purpose |
|---|---|---|
| `models.conf` | — | the verify matrix: `family \| model_path \| device \| port \| tp \| expect` |
| `smoke_one_model.sh` | in container | one `vllm serve` smoke pinned to one device; health-wait → semantic request → PASS/FAIL/SKIP |
| `unit_tests.sh` | in container | `pytest tests/test_patches.py` (the patch-mechanism guard), pinned to a spare device |
| `verify.sh` | local | push → sync container → run unit + all smokes in parallel → summary (pure bash, no Claude) |
| `verify_workflow.js` | local (Workflow tool) | same parallel verify phase, agent-orchestrated |

## Shared-box safety (important)

This box also hosts the private M3 epic. The harness is built to be safe alongside other jobs:

- `smoke_one_model.sh` checks `mthreads-gmi` for its target device and **SKIPs** (does
  not fail, does not kill anything) if free memory is below `MIN_FREE_MIB` (default 40 GiB).
  It never steals a GPU from another job.
- On exit it kills **only its own process group** (`setsid` + `kill -- -PGID`), never a
  blanket `pkill -f vllm`.
- If you intend to reclaim the whole box, kill the prior job explicitly first (operator
  decision), then run the harness.

## Secrets

Never hardcode the SSH password. `verify.sh` reads `$REMOTE_PASS`; the Workflow takes it
via `args.pass` at invoke time. The value lives in `curated/musa-remote-config.md`.

> **Runner recommendation:** prefer **`verify.sh`** (bash, reads `$REMOTE_PASS` directly).
> The Workflow-tool path (`verify_workflow.js`) routes the password through subagents, which
> proved unreliable in practice (agents fell back to an unset `$REMOTE_PASS` → `sshpass -p ''`
> → auth failure). Use the Workflow only when you can confirm the credential reaches the agents.

```bash
export REMOTE_PASS=...        # from curated/musa-remote-config.md (or the dev shell)
```

## The update → verify loop

1. Make a change on the branch, commit.
2. **Sync + (re)build the container** to the new SHA:
   - Python-only diff → `git fetch && git reset --hard <sha>` (editable install picks it up).
   - csrc / setup.py / `.cu` diff, **or first switch onto this branch** → also
     `pip install -e . --no-build-isolation -v` (set `REINSTALL=1` for `verify.sh`).
3. **Verify** (unit + smokes, parallel, device-pinned):

   ```bash
   # one-shot bash (does push + sync + parallel verify):
   REMOTE_PASS=... REINSTALL=1 tools/musa_verify/verify.sh         # first run on a fresh container
   REMOTE_PASS=...             tools/musa_verify/verify.sh         # subsequent Python-only iterations
   ```

   or via the Workflow tool (after the container is already synced/built):

   ```
   Workflow({ scriptPath: "vllm-musa/tools/musa_verify/verify_workflow.js",
              args: { pass: "<REMOTE_PASS>", container: "yeahdongcn70",
                      models: [ {family:"deepseek_v2_lite", path:"/home/dist/DeepSeek-V2-Lite", device:0, port:8000, tp:1, expect:"beijing"},
                                {family:"qwen3_8b",         path:"/home/dist/Qwen3-8B",          device:1, port:8001, tp:1, expect:"beijing"} ] } })
   ```

4. Read the summary. A run is GREEN only if no line is `FAIL` (`SKIP` is allowed — it means a
   device was busy). Re-run skipped models once their device frees up.

## Defaults

- Container `yeahdongcn70` on `mccxadmin@10.18.32.18`, repo at `/ws`, fork remote `yeahdongcn`.
- Functional-smoke env: `TORCHDYNAMO_DISABLE=1`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`,
  `--enforce-eager`. These are **functional** smokes — do not read perf numbers off them.
- Override any default via env (`MUSA_CONTAINER`, `MUSA_HOST`, `UNIT_TEST_DEVICE`, …) or `models.conf`.

## Add a model

Append a row to `models.conf` on a free device/port, or pass a longer `models` array to the
Workflow. Keep one device per concurrent smoke.
