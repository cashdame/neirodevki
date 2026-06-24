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
# iter23: idempotent faceswap weight download onto the network volume.
#
# ReActor swap_model lists only files physically present in insightface/ at
# ComfyUI startup — symlinks from iter13+ require the real files to exist on
# the volume first.  This block downloads them once; subsequent cold starts
# skip the download (file present + size > 1 MB).
#
# Placed BEFORE extra_model_paths.yaml + symlink loop so the directories
# exist and are populated before ComfyUI discovers them.
#
# Models sourced from the canonical ReActor HuggingFace dataset repo
# (same source referenced in the ReActor README).
# ---------------------------------------------------------------------------
FACESWAP_INSWAPPER_DST="/runpod-volume/models/insightface/inswapper_128.onnx"
FACESWAP_INSWAPPER_URL="https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx"
FACESWAP_GFPGAN_DST="/runpod-volume/models/facerestore_models/GFPGANv1.4.pth"
FACESWAP_GFPGAN_URL="https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.pth"
FACESWAP_MIN_SIZE=1048576  # 1 MB — anything smaller is a broken download

_download_if_missing() {
    local dst="$1"
    local url="$2"
    local label="$3"
    local dir
    dir="$(dirname "${dst}")"

    mkdir -p "${dir}"

    local actual_size=0
    if [ -f "${dst}" ]; then
        actual_size=$(stat -c%s "${dst}" 2>/dev/null || echo 0)
    fi

    if [ "${actual_size}" -ge "${FACESWAP_MIN_SIZE}" ]; then
        echo "[faceswap-dl] ${label} already on volume (${actual_size} bytes), skipping."
        return 0
    fi

    if [ -f "${dst}" ]; then
        echo "[faceswap-dl] ${label} exists but too small (${actual_size} bytes) — removing and re-downloading."
        rm -f "${dst}"
    fi

    echo "[faceswap-dl] Downloading ${label} -> ${dst} ..."
    if wget --tries=3 --timeout=120 --no-verbose -O "${dst}" "${url}"; then
        local final_size
        final_size=$(stat -c%s "${dst}" 2>/dev/null || echo 0)
        echo "[faceswap-dl] ${label} downloaded: ${final_size} bytes."
    else
        echo "[faceswap-dl] ERROR: failed to download ${label} from ${url}" >&2
        rm -f "${dst}"  # remove partial file so next restart retries
        exit 1
    fi
}

_download_if_missing "${FACESWAP_INSWAPPER_DST}" "${FACESWAP_INSWAPPER_URL}" "inswapper_128.onnx"
_download_if_missing "${FACESWAP_GFPGAN_DST}"    "${FACESWAP_GFPGAN_URL}"    "GFPGANv1.4.pth"

# ---------------------------------------------------------------------------
# iter25: idempotent download of Wan 2.2 Animate models onto the network volume.
#
# These models are required by the wan_animate.json workflow but are NOT present
# on the volume after the standard Wan 2.1 i2v setup:
#   - Animate backbone (14B fp8, ~16 GB)         → diffusion_models/
#   - CLIP Vision H encoder                      → clip_vision/
#   - YOLOv10m detection model (ONNX)            → detection/
#   - ViTPose wholebody model (ONNX)             → detection/
#   - ViTPose wholebody external data (.bin)     → detection/  (required by ONNX runtime)
#
# Uses the same _download_if_missing helper as iter23 (idempotent, exits 1 on
# failure so broken cold-starts are caught immediately).
#
# Placed BEFORE iter13 (extra_model_paths.yaml + symlink loop) so the files
# are present when ComfyUI discovers them.
# ---------------------------------------------------------------------------
echo "entrypoint: iter25 — downloading Wan 2.2 Animate models if missing"

_download_if_missing \
    "/runpod-volume/models/diffusion_models/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors" \
    "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors" \
    "Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors"

_download_if_missing \
    "/runpod-volume/models/clip_vision/clip_vision_h.safetensors" \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors" \
    "clip_vision_h.safetensors"

_download_if_missing \
    "/runpod-volume/models/detection/yolov10m.onnx" \
    "https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/resolve/main/process_checkpoint/det/yolov10m.onnx" \
    "yolov10m.onnx"

_download_if_missing \
    "/runpod-volume/models/detection/vitpose_h_wholebody_model.onnx" \
    "https://huggingface.co/Kijai/vitpose_comfy/resolve/main/onnx/vitpose_h_wholebody_model.onnx" \
    "vitpose_h_wholebody_model.onnx"

_download_if_missing \
    "/runpod-volume/models/detection/vitpose_h_wholebody_data.bin" \
    "https://huggingface.co/Kijai/vitpose_comfy/resolve/main/onnx/vitpose_h_wholebody_data.bin" \
    "vitpose_h_wholebody_data.bin"

echo "entrypoint: iter25 done"

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
    facerestore_models: |
        /runpod-volume/models/facerestore_models/
        /runpod-volume/facerestore_models/
    insightface: |
        /runpod-volume/models/insightface/
        /runpod-volume/insightface/
YAML
echo "entrypoint: wrote /comfyui/extra_model_paths.yaml (iter14)"

# ---------------------------------------------------------------------------
# Symlink approach as belt-and-suspenders: also create symlinks if the volume
# directory exists and the ComfyUI target does not yet exist as a real dir.
#
# If COMFYUI_DIR already exists as an empty directory (created by base image),
# remove it first so the symlink can be placed.
# ---------------------------------------------------------------------------
for MODEL_TYPE in diffusion_models text_encoders clip loras vae upscale_models clip_vision detection facerestore_models insightface; do
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

# Faceswap model diagnostics — visible on every cold start so path issues are
# caught immediately without a separate debug run.
echo "[faceswap-diag] inswapper: $(ls /comfyui/models/insightface/ 2>/dev/null | grep -i inswapper || echo MISSING)"
echo "[faceswap-diag] gfpgan: $(ls /comfyui/models/facerestore_models/ 2>/dev/null | grep -i gfpgan || echo MISSING)"

# ---------------------------------------------------------------------------
# iter22: RIFE import diagnostic — runs on REAL GPU, no CUDA mock.
#
# Imports ComfyUI-Frame-Interpolation __init__.py directly and prints whether
# RIFE VFI appears in NODE_CLASS_MAPPINGS. If the iter22 lazy-patch worked,
# you will see:
#   [rife-diag] RIFE VFI present: True | keys: [...]
# If something is still wrong (exception swallowed by ComfyUI node-loader),
# this block surfaces the actual traceback BEFORE ComfyUI starts.
#
# `|| true` ensures a diagnostic failure does not abort worker startup.
# ---------------------------------------------------------------------------
echo "=== RIFE import diagnostic (real GPU, no CUDA mock) ==="
python -c "
import sys, traceback, importlib.util
sys.path.insert(0, '/comfyui/custom_nodes/ComfyUI-Frame-Interpolation')
try:
    spec = importlib.util.spec_from_file_location(
        '_rife_diag',
        '/comfyui/custom_nodes/ComfyUI-Frame-Interpolation/__init__.py',
        submodule_search_locations=['/comfyui/custom_nodes/ComfyUI-Frame-Interpolation'])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = getattr(mod, 'NODE_CLASS_MAPPINGS', {})
    print('[rife-diag] RIFE VFI present:', 'RIFE VFI' in m, '| keys:', list(m.keys()))
except Exception:
    print('[rife-diag] IMPORT FAILED:')
    traceback.print_exc()
" 2>&1 || true
echo "=== END RIFE diagnostic ==="

exec /start.sh
