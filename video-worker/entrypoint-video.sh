#!/bin/bash
# entrypoint-video.sh — EP2 video worker entrypoint (Phase B).
#
# 1. Run check_imports.py as a dep-smoke-test.
#    Exit 0  → proceed to /start.sh (ComfyUI + RunPod serverless handler).
#    Exit 1  → write /tmp/worker_error.json and exit 1 so the RunPod job
#              returns FAILED with a readable message instead of timing out.
#
# This fixes the EP1 Critical: silent 10-min timeout on broken dependencies.
set -euo pipefail

# CI smoke-only mode: run imports check and exit without starting the RunPod worker.
if [ "${SMOKE_TEST_ONLY:-0}" = "1" ]; then
    python /check_imports.py
    exit $?
fi

CHECK_STDERR=$(mktemp)

if ! python /check_imports.py 2>"${CHECK_STDERR}"; then
    # Pass trace via env var — avoids shell quoting issues with arbitrary text.
    IMPORT_TRACE=$(cat "${CHECK_STDERR}") python3 -c "
import json, os
payload = {
    'error': 'check_imports failed — one or more deps could not be imported',
    'trace': os.environ.get('IMPORT_TRACE', ''),
}
with open('/tmp/worker_error.json', 'w') as f:
    json.dump(payload, f)
"
    rm -f "${CHECK_STDERR}"
    exit 1
fi

rm -f "${CHECK_STDERR}"

exec /start.sh
