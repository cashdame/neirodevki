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
# iter26: idempotent model download via huggingface_hub (replaces iter23+iter25 wget).
#
# huggingface_hub.hf_hub_download:
#   - Handles XetHub CAS (plain wget/curl only gets a manifest for Xet-backed files).
#   - Resumes partial downloads automatically.
#   - Does NOT re-verify a file already at local_dir/{filename} by default, so we
#     add a pre-check: if the file exists but its size differs from the HF metadata
#     size, delete it first so hf_hub_download re-fetches cleanly.  This is the
#     critical guard for the Animate-14B 18.4 GB file that was left partial by wget.
#
# Covers (previously iter23): inswapper_128.onnx, GFPGANv1.4.pth
# Covers (previously iter25): Wan2_2 Animate backbone, clip_vision_h, yolov10m,
#                              vitpose model+data
#
# Placed BEFORE iter13 (extra_model_paths.yaml + symlink loop).
# ---------------------------------------------------------------------------
echo "entrypoint: iter26 — downloading models via huggingface_hub"

python3 - <<'PYEOF'
import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")  # public repos — token optional, avoids rate-limit

# Each entry: (repo_id, repo_type, filename_in_repo, dst_path_on_volume)
# filename_in_repo is the path as stored in the HF repo (may contain subdirs).
# dst_path_on_volume is the exact path ComfyUI expects.
MODELS = [
    # --- faceswap (ex-iter23) ---
    (
        "Gourieff/ReActor", "dataset",
        "models/inswapper_128.onnx",
        "/runpod-volume/models/insightface/inswapper_128.onnx",
    ),
    (
        "Gourieff/ReActor", "dataset",
        "models/facerestore_models/GFPGANv1.4.pth",
        "/runpod-volume/models/facerestore_models/GFPGANv1.4.pth",
    ),
    # --- Wan 2.2 Animate (ex-iter25) ---
    # Wan2_2-Animate-14B baked into image at /comfyui/baked_models/diffusion_models/
    # — no runtime download needed. See Dockerfile-video Layer 6.
    (
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "model",
        "split_files/clip_vision/clip_vision_h.safetensors",
        "/runpod-volume/models/clip_vision/clip_vision_h.safetensors",
    ),
    (
        "Wan-AI/Wan2.2-Animate-14B", "model",
        "process_checkpoint/det/yolov10m.onnx",
        "/runpod-volume/models/detection/yolov10m.onnx",
    ),
    (
        "Kijai/vitpose_comfy", "model",
        "onnx/vitpose_h_wholebody_model.onnx",
        "/runpod-volume/models/detection/vitpose_h_wholebody_model.onnx",
    ),
    (
        "Kijai/vitpose_comfy", "model",
        "onnx/vitpose_h_wholebody_data.bin",
        "/runpod-volume/models/detection/vitpose_h_wholebody_data.bin",
    ),
]

api = HfApi(token=HF_TOKEN)

def get_expected_size(repo_id, repo_type, filename):
    """Return expected file size in bytes from HF metadata, or None on error."""
    try:
        # get_paths_info returns a list of RepoSibling / RepoFile objects.
        infos = api.get_paths_info(
            repo_id=repo_id,
            paths=filename,
            repo_type=repo_type,
            expand=True,
        )
        for item in infos:
            if hasattr(item, "lfs") and item.lfs is not None:
                return item.lfs.size
            if hasattr(item, "size") and item.size is not None:
                return item.size
    except Exception as exc:
        print(f"[hf-dl] WARN: could not fetch metadata for {repo_id}/{filename}: {exc}", flush=True)
    return None

def ensure_model(repo_id, repo_type, filename, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    label = dst.name

    expected = get_expected_size(repo_id, repo_type, filename)

    if dst.exists():
        actual = dst.stat().st_size
        if expected is not None and actual != expected:
            print(
                f"[hf-dl] {label}: stale/partial file on volume "
                f"({actual:,} bytes, expected {expected:,}) — removing.",
                flush=True,
            )
            dst.unlink()
        elif expected is None and actual > 0:
            # Metadata unavailable; trust the existing file to avoid re-downloading.
            print(f"[hf-dl] {label}: already on volume ({actual:,} bytes, size unverified), skipping.", flush=True)
            return
        else:
            print(f"[hf-dl] {label}: already on volume ({actual:,} bytes), skipping.", flush=True)
            return

    print(f"[hf-dl] {label}: downloading from {repo_id} ...", flush=True)

    # hf_hub_download with local_dir places the file at:
    #   {local_dir}/{filename}   (preserving subdirectories from the repo).
    # We use a temp staging dir so we can move the result to the exact dst path.
    stage_dir = dst.parent / ".hf_stage"
    stage_dir.mkdir(parents=True, exist_ok=True)

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=str(stage_dir),
            local_dir_use_symlinks=False,
            token=HF_TOKEN,
        )
        downloaded = Path(downloaded)
        # Move to the exact target path expected by ComfyUI.
        shutil.move(str(downloaded), str(dst))
    finally:
        # Clean up the staging dir (including any empty subdirs left by hf_hub).
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)

    final_size = dst.stat().st_size
    print(f"[hf-dl] {label}: done ({final_size:,} bytes -> {dst})", flush=True)

    if expected is not None and final_size != expected:
        print(
            f"[hf-dl] ERROR: {label} size mismatch after download "
            f"(got {final_size:,}, expected {expected:,})",
            file=sys.stderr,
        )
        sys.exit(1)

errors = []
for repo_id, repo_type, filename, dst in MODELS:
    try:
        ensure_model(repo_id, repo_type, filename, dst)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[hf-dl] ERROR: {dst} — {exc}", file=sys.stderr, flush=True)
        errors.append(dst)

if errors:
    print(f"[hf-dl] {len(errors)} model(s) failed to download — aborting.", file=sys.stderr)
    sys.exit(1)

print("[hf-dl] All models ready.", flush=True)
PYEOF

echo "entrypoint: iter26 done"

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
        /comfyui/baked_models/diffusion_models/
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

# Verify baked Animate-14B is present in the image layer.
echo "[bake-diag] baked_models/diffusion_models:"
ls -la /comfyui/baked_models/diffusion_models/ 2>/dev/null || echo "[bake-diag] WARN: /comfyui/baked_models/diffusion_models/ not found"

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
