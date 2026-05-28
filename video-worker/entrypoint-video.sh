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

# ---------------------------------------------------------------------------
# iter13: write extra_model_paths.yaml so ComfyUI discovers Wan models on the
# network volume.
#
# worker-comfyui:5.8.5-base ships its own extra_model_paths.yaml that covers
# checkpoints, loras, vae, etc., but NOT diffusion_models, text_encoders, or
# clip — all three are required by WanVideoModelLoader and its T5 text encoder.
#
# extra_model_paths.yaml is read by ComfyUI at startup, after RunPod mounts the
# network volume. This is more reliable than symlinks created before the volume
# is mounted (the previous approach: symlinks were silently skipped when
# VOLUME_DIR did not yet exist at entrypoint startup time).
#
# Two base paths are probed to cover both populate layouts:
#   - /runpod-volume/models/<type>/  (default: MODELS_ROOT=/runpod-volume/models)
#   - /runpod-volume/<type>/         (fallback: MODELS_ROOT=/runpod-volume)
# ComfyUI merges paths from all entries, so listing both is harmless even when
# one of them doesn't exist.
# ---------------------------------------------------------------------------
cat > /comfyui/extra_model_paths.yaml << 'YAML'
comfyui:
    diffusion_models: |
        /runpod-volume/models/diffusion_models/
        /runpod-volume/diffusion_models/
    text_encoders: |
        /runpod-volume/models/text_encoders/
        /runpod-volume/text_encoders/
    clip: |
        /runpod-volume/models/clip/
        /runpod-volume/clip/
    clip_vision: |
        /runpod-volume/models/clip_vision/
        /runpod-volume/clip_vision/
    vae: |
        /runpod-volume/models/vae/
        /runpod-volume/vae/
    loras: |
        /runpod-volume/models/loras/
        /runpod-volume/loras/
YAML
echo "entrypoint: wrote /comfyui/extra_model_paths.yaml (iter14)"

# ---------------------------------------------------------------------------
# Symlink approach as belt-and-suspenders: also create symlinks if the volume
# directory exists and the ComfyUI target does not yet exist as a real dir.
#
# If COMFYUI_DIR already exists as an empty directory (created by base image),
# remove it first so the symlink can be placed.
# ---------------------------------------------------------------------------
for MODEL_TYPE in diffusion_models text_encoders clip loras; do
    COMFYUI_DIR="/comfyui/models/${MODEL_TYPE}"

    # Try both populate layouts.
    for CANDIDATE in \
        "/runpod-volume/models/${MODEL_TYPE}" \
        "/runpod-volume/${MODEL_TYPE}"; do

        if [ -d "${CANDIDATE}" ]; then
            VOLUME_DIR="${CANDIDATE}"
            # Remove empty placeholder dir if present (base image creates these).
            if [ -d "${COMFYUI_DIR}" ] && [ ! -L "${COMFYUI_DIR}" ]; then
                # Only remove if empty — don't silently nuke real content.
                if [ -z "$(ls -A "${COMFYUI_DIR}" 2>/dev/null)" ]; then
                    rmdir "${COMFYUI_DIR}"
                fi
            fi
            # Create symlink if target slot is free.
            if [ ! -e "${COMFYUI_DIR}" ]; then
                ln -s "${VOLUME_DIR}" "${COMFYUI_DIR}"
                echo "entrypoint: linked ${COMFYUI_DIR} -> ${VOLUME_DIR}"
            fi
            break
        fi
    done
done

exec /start.sh
