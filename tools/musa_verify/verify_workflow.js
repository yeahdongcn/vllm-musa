// verify_workflow.js — Claude Code Workflow for the patch-maintenance branch.
//
// Runs the parallel verify phase: unit tests + every model smoke CONCURRENTLY,
// each smoke pinned to its own MUSA device via MUSA_VISIBLE_DEVICES. Assumes the
// container has ALREADY been synced to the SHA under test and rebuilt if csrc
// changed (do the kill/reset/reinstall step first — see README "update->verify loop").
//
// Launch with the Workflow tool:
//   Workflow({ scriptPath: "vllm-musa/tools/musa_verify/verify_workflow.js", args: {
//     pass: "<REMOTE_PASS>",                 // do NOT commit a real value
//     host: "mccxadmin@10.18.32.18",
//     container: "yeahdongcn70",
//     ws: "/ws",
//     unitDevice: 7,
//     models: [
//       { family: "deepseek_v2_lite", path: "/home/dist/DeepSeek-V2-Lite", device: 0, port: 8000, tp: 1, expect: "beijing" },
//       { family: "qwen3_8b",         path: "/home/dist/Qwen3-8B",          device: 1, port: 8001, tp: 1, expect: "beijing" }
//     ]
//   }})
//
// Each smoke runs the committed tools/musa_verify/smoke_one_model.sh, which SKIPs a
// busy device rather than stealing it, and tears down only its own process group.

export const meta = {
  name: 'musa-verify',
  description: 'Parallel vllm-musa verify: unit tests + per-model smokes, one MUSA device each',
  phases: [{ title: 'Verify', detail: 'unit tests + model smokes in parallel, device-pinned' }],
}

const a = args || {}
const host = a.host || 'mccxadmin@10.18.32.18'
const container = a.container || 'yeahdongcn70'
const ws = a.ws || '/ws'             // use '/ws-verify' for the isolated-build flow
const venv = a.venv || '/root/.virtualenvs/sglang-0.5.6'   // pinned test env (see README)
const pass = a.pass || ''            // required; passed at invoke time, never committed
const unitDevice = a.unitDevice ?? 7
const models = a.models || [
  { family: 'deepseek_v2_lite', path: '/home/dist/DeepSeek-V2-Lite', device: 0, port: 8000, tp: 1, expect: 'beijing' },
  { family: 'qwen3_8b', path: '/home/dist/Qwen3-8B', device: 1, port: 8001, tp: 1, expect: 'beijing' },
]

const RESULT = {
  type: 'object',
  properties: {
    name: { type: 'string' },
    status: { type: 'string', enum: ['PASS', 'FAIL', 'SKIP'] },
    line: { type: 'string', description: 'the verbatim PASS/FAIL/SKIP line from the script' },
  },
  required: ['name', 'status', 'line'],
}

// sshpass wrapper: run one in-container command, non-interactively.
const rexec = (inner) =>
  `sshpass -p '${pass}' ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 ${host} ` +
  `"docker exec -i ${container} bash -lc 'source ${venv}/bin/activate 2>/dev/null || true; cd ${ws} && ${inner}'"`

const runAgent = (name, label, cmd) =>
  agent(
    `Run EXACTLY this command with the Bash tool (do not modify it, do not add flags):\n\n` +
    `${cmd}\n\n` +
    `The script prints exactly one line beginning with PASS, FAIL, or SKIP as its final stdout line. ` +
    `Return that line verbatim as "line", set "status" to its first word, and "name" to "${name}". ` +
    `If the command errors or prints no such line, set status=FAIL and line to the last ~200 chars of output.`,
    { label, phase: 'Verify', schema: RESULT },
  )

phase('Verify')

const thunks = []
thunks.push(() =>
  runAgent('unit', 'verify:unit',
    rexec(`UNIT_TEST_DEVICE=${unitDevice} bash tools/musa_verify/unit_tests.sh`)))

for (const m of models) {
  thunks.push(() =>
    runAgent(m.family, `verify:${m.family}@dev${m.device}`,
      rexec(`bash tools/musa_verify/smoke_one_model.sh ${m.family} ${m.path} ${m.device} ${m.port} ${m.tp} ${m.expect}`)))
}

const results = (await parallel(thunks)).filter(Boolean)

log('verify results:\n' + results.map(r => `  ${r.line}`).join('\n'))

const failed = results.filter(r => r.status === 'FAIL')
return {
  green: failed.length === 0,
  results,
  summary: results.map(r => r.line),
}
