#!/usr/bin/env bash
# verify.sh — update->verify driver for the patch-maintenance branch.
#
# Runs from your LOCAL machine. It (1) pushes the current branch to the fork,
# (2) syncs the MUSA container to that exact SHA (reinstalling only when csrc
# changed), then (3) runs unit tests + every model smoke in models.conf
# CONCURRENTLY, each pinned to its own MUSA device via MUSA_VISIBLE_DEVICES.
#
# Secrets: reads $REMOTE_PASS from the environment. NEVER hardcode the password.
#   export REMOTE_PASS=...   # value in curated/musa-remote-config.md
#
# Config (env overrides):
#   MUSA_HOST=mccxadmin@10.18.32.18   MUSA_CONTAINER=yeahdongcn70
#   FORK_REMOTE=yeahdongcn            CONTAINER_WS=/ws
#   REINSTALL=1                       # force `pip install -e .` (set when you touched csrc)
#   UNIT_TEST_DEVICE=7                # spare device for the unit-test import
#
# Exit code: 0 only if no smoke/unit line is FAIL (SKIP is allowed).
set -uo pipefail

: "${REMOTE_PASS:?export REMOTE_PASS (see curated/musa-remote-config.md)}"
HOST="${MUSA_HOST:-mccxadmin@10.18.32.18}"
CONTAINER="${MUSA_CONTAINER:-yeahdongcn70}"
FORK="${FORK_REMOTE:-yeahdongcn}"
WS="${CONTAINER_WS:-/ws}"
UNIT_TEST_DEVICE="${UNIT_TEST_DEVICE:-7}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
SHA="$(git rev-parse HEAD)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

rexec(){ sshpass -p "$REMOTE_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$HOST" "docker exec -i $CONTAINER bash -s"; }

echo "==== verify branch=$BRANCH sha=${SHA:0:9} container=$CONTAINER ===="

echo "---- 1) push to fork ----"
git push "$FORK" "$BRANCH"

echo "---- 2) sync container to $SHA ----"
rexec >"$TMP/sync.out" 2>&1 <<EOF
set -uo pipefail
cd $WS
git fetch $FORK $BRANCH
git reset --hard $SHA
git log -1 --oneline
EOF
cat "$TMP/sync.out"
grep -q "${SHA:0:9}" "$TMP/sync.out" || { echo "ABORT: container did not sync to $SHA"; exit 1; }

if [ "${REINSTALL:-0}" = "1" ]; then
  echo "---- 2b) reinstall (csrc touched) ----"
  rexec <<EOF
set -uo pipefail
cd $WS && python -m pip install -e . --no-build-isolation -v
EOF
fi

echo "---- 3) parallel verify (unit + smokes) ----"
pids=(); names=()

( rexec <<EOF
cd $WS && UNIT_TEST_DEVICE=$UNIT_TEST_DEVICE bash tools/musa_verify/unit_tests.sh
EOF
) >"$TMP/unit.out" 2>&1 & pids+=($!); names+=("unit")

while IFS='|' read -r family model dev port tp expect; do
  family="$(echo "$family" | xargs)"; [ -z "$family" ] && continue
  case "$family" in \#*) continue;; esac
  model="$(echo "$model" | xargs)"; dev="$(echo "$dev" | xargs)"
  port="$(echo "$port" | xargs)"; tp="$(echo "$tp" | xargs)"; expect="$(echo "$expect" | xargs)"
  ( rexec <<EOF
cd $WS && bash tools/musa_verify/smoke_one_model.sh $family $model $dev $port $tp $expect
EOF
  ) >"$TMP/${family}.out" 2>&1 & pids+=($!); names+=("$family")
done < "$HERE/models.conf"

for p in "${pids[@]}"; do wait "$p" || true; done

echo ""
echo "==== VERIFY SUMMARY (branch=$BRANCH sha=${SHA:0:9} container=$CONTAINER) ===="
rc=0
for n in "${names[@]}"; do
  line="$(grep -hE '^(PASS|FAIL|SKIP) ' "$TMP/${n}.out" | tail -1)"
  [ -z "$line" ] && line="FAIL $n reason=no_result_line (see output)"
  echo "  $line"
  case "$line" in FAIL*) rc=1;; esac
done
exit "$rc"
