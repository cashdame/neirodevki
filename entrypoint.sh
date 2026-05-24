#!/bin/bash
set -e

VOLUME="/runpod-volume"
FLUX_SRC="$VOLUME/models/checkpoints/flux1-dev-fp8.safetensors"
FLUX_DST="/comfyui/models/checkpoints/flux1-dev-fp8.safetensors"
MIN_SIZE=15000000000  # 15 GB minimum

mkdir -p "$VOLUME/models/checkpoints"
mkdir -p "$(dirname "$FLUX_DST")"

# huggingface_hub handles XetHub CAS (plain curl gets only a manifest). Install
# once up front -- both Flux and ControlNet downloads below rely on it.
pip install -q --upgrade huggingface_hub 2>&1 | tail -1

# Check if file exists and is large enough
NEED_DOWNLOAD=1
if [ -f "$FLUX_SRC" ]; then
    ACTUAL_SIZE=$(stat -c%s "$FLUX_SRC" 2>/dev/null || echo 0)
    if [ "$ACTUAL_SIZE" -ge "$MIN_SIZE" ]; then
        echo "[entrypoint] Flux already on volume ($(numfmt --to=iec $ACTUAL_SIZE)), skipping download."
        NEED_DOWNLOAD=0
    else
        echo "[entrypoint] Flux on volume is too small ($ACTUAL_SIZE bytes) — deleting and re-downloading."
        rm -f "$FLUX_SRC"
    fi
fi

if [ "$NEED_DOWNLOAD" -eq 1 ]; then
    echo "[entrypoint] Downloading Flux fp8 (~17GB) via huggingface_hub (handles XetHub CAS)..."
    python3 - <<'EOF'
import sys, os
from huggingface_hub import hf_hub_download

dst_dir = "/runpod-volume/models/checkpoints"
filename = "flux1-dev-fp8.safetensors"
print(f"[entrypoint] Starting hf_hub_download ...", flush=True)
path = hf_hub_download(
    repo_id="Comfy-Org/flux1-dev",
    filename=filename,
    local_dir=dst_dir,
    local_dir_use_symlinks=False,
)
size = os.path.getsize(path)
print(f"[entrypoint] Downloaded {size:,} bytes -> {path}", flush=True)
if size < 15_000_000_000:
    print(f"[entrypoint] ERROR: file too small ({size} bytes)", file=sys.stderr)
    sys.exit(1)
EOF
    DOWNLOADED=$(stat -c%s "$FLUX_SRC" 2>/dev/null || echo 0)
    echo "[entrypoint] Download complete: $(numfmt --to=iec $DOWNLOADED)"
fi

ln -sf "$FLUX_SRC" "$FLUX_DST"
echo "[entrypoint] Flux ready at $FLUX_DST"

# --- ControlNet Union Pro 2.0 (~4.3 GB) on the volume, same scheme as Flux ----
CN_SRC="$VOLUME/models/controlnet/flux_union_pro2.safetensors"
CN_DST="/comfyui/models/controlnet/flux_union_pro2.safetensors"
CN_MIN_SIZE=4000000000  # 4 GB minimum
mkdir -p "$VOLUME/models/controlnet"
mkdir -p "$(dirname "$CN_DST")"

CN_NEED=1
if [ -f "$CN_SRC" ]; then
    CN_ACTUAL=$(stat -c%s "$CN_SRC" 2>/dev/null || echo 0)
    if [ "$CN_ACTUAL" -ge "$CN_MIN_SIZE" ]; then
        echo "[entrypoint] ControlNet already on volume ($(numfmt --to=iec $CN_ACTUAL)), skipping."
        CN_NEED=0
    else
        echo "[entrypoint] ControlNet on volume too small ($CN_ACTUAL bytes) — re-downloading."
        rm -f "$CN_SRC"
    fi
fi

if [ "$CN_NEED" -eq 1 ]; then
    echo "[entrypoint] Downloading Flux ControlNet Union Pro 2.0 (~4.3GB) via huggingface_hub..."
    python3 - <<'EOF'
import sys, os, shutil
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
    filename="diffusion_pytorch_model.safetensors",
    local_dir="/runpod-volume/models/controlnet",
    local_dir_use_symlinks=False,
)
dst = "/runpod-volume/models/controlnet/flux_union_pro2.safetensors"
if path != dst:
    shutil.move(path, dst)
size = os.path.getsize(dst)
print(f"[entrypoint] ControlNet downloaded {size:,} bytes -> {dst}", flush=True)
if size < 4_000_000_000:
    print(f"[entrypoint] ERROR: ControlNet too small ({size} bytes)", file=sys.stderr)
    sys.exit(1)
EOF
fi

ln -sf "$CN_SRC" "$CN_DST"
echo "[entrypoint] ControlNet ready at $CN_DST"

exec /start.sh
