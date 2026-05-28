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
    detection: |
        /runpod-volume/models/detection/
        /runpod-volume/detection/
YAML
echo "entrypoint: wrote /comfyui/extra_model_paths.yaml (iter14)"

# ---------------------------------------------------------------------------
# Symlink approach as belt-and-suspenders: also create symlinks if the volume
# directory exists and the ComfyUI target does not yet exist as a real dir.
#
# If COMFYUI_DIR already exists as an empty directory (created by base image),
# remove it first so the symlink can be placed.
# ---------------------------------------------------------------------------
for MODEL_TYPE in diffusion_models text_encoders clip loras vae upscale_models clip_vision detection; do
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

echo "entrypoint: loras dir contents -> $(ls /comfyui/models/loras/ 2>/dev/null | head -5 || echo MISSING)"

# iter18: symlink RIFE checkpoint for Frame-Interpolation plugin.
# Plugin reads ckpts from /comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/
# Volume has it at /runpod-volume/models/upscale_models/rife/rife49.pth
FI_CKPTS=/comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts
VOL_RIFE=/runpod-volume/models/upscale_models/rife
if [ -d "${VOL_RIFE}" ]; then
    mkdir -p "${FI_CKPTS}"
    if [ ! -e "${FI_CKPTS}/rife" ]; then
        ln -s "${VOL_RIFE}" "${FI_CKPTS}/rife"
        echo "entrypoint: linked Frame-Interpolation ckpts/rife -> ${VOL_RIFE}"
    fi
fi

# ---------------------------------------------------------------------------
# iter16: patch rp_upload.py in-place to honor BUCKET_NAME env var.
#
# Stock rp_upload uses time.strftime("%m-%y") (e.g. "05-26") if no
# bucket_name is passed — and worker-comfyui's handler doesn't pass one.
# B2 bucket names must be >=6 chars + globally unique, so we can't use "05-26".
#
# iter15 attempted sitecustomize.py monkey-patch but handler didn't pick it up.
# This direct sed-patch on rp_upload.py is the simplest reliable fix.
# ---------------------------------------------------------------------------
RP_UPLOAD=$(python -c "from runpod.serverless.utils import rp_upload; print(rp_upload.__file__)" 2>/dev/null)
if [ -n "${RP_UPLOAD}" ] && [ -f "${RP_UPLOAD}" ]; then
    # Replace the two default-bucket-name lines with BUCKET_NAME env fallback.
    # Both occurrences use the same time.strftime pattern.
    sed -i 's|bucket_name = time\.strftime("%m-%y")|bucket_name = os.environ.get("BUCKET_NAME") or time.strftime("%m-%y")|g' "${RP_UPLOAD}"
    sed -i 's|bucket = bucket_name if bucket_name else time\.strftime("%m-%y")|bucket = bucket_name if bucket_name else (os.environ.get("BUCKET_NAME") or time.strftime("%m-%y"))|g' "${RP_UPLOAD}"
    echo "entrypoint: patched ${RP_UPLOAD} (iter16, BUCKET_NAME fallback)"
    grep -n "BUCKET_NAME" "${RP_UPLOAD}" | head -5
else
    echo "entrypoint: WARN — could not locate rp_upload.py" >&2
fi

exec /start.sh
