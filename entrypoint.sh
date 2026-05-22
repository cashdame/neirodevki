#!/bin/bash
set -e

VOLUME="/runpod-volume"
FLUX_SRC="$VOLUME/models/checkpoints/flux1-dev-fp8.safetensors"
FLUX_DST="/comfyui/models/checkpoints/flux1-dev-fp8.safetensors"
MIN_SIZE=15000000000  # 15 GB minimum

mkdir -p "$VOLUME/models/checkpoints"
mkdir -p "$(dirname "$FLUX_DST")"

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
    # HuggingFace now uses XetHub CAS protocol — plain curl/wget only downloads
    # a 3.7 MB reconstruction manifest, not the actual file. huggingface_hub
    # handles the Xet protocol and reassembles chunks correctly.
    pip install -q --upgrade huggingface_hub 2>&1 | tail -1
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

exec /start.sh
