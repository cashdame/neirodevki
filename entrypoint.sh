#!/bin/bash
set -e

VOLUME="/runpod-volume"
FLUX_SRC="$VOLUME/models/checkpoints/flux1-dev-fp8.safetensors"
FLUX_DST="/comfyui/models/checkpoints/flux1-dev-fp8.safetensors"
FLUX_URL="https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors"
MIN_SIZE=15000000000  # 15 GB minimum — anything smaller is a corrupt/partial download

mkdir -p "$VOLUME/models/checkpoints"
mkdir -p "$(dirname "$FLUX_DST")"

# Check if file exists and is large enough (not an HTML error page)
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
    echo "[entrypoint] Downloading Flux fp8 (~17GB) to network volume..."
    # curl -fSL: -f fail on HTTP errors, -S show errors, -L follow redirects
    curl -fSL --retry 3 --retry-delay 5 \
        -o "$FLUX_SRC" \
        "$FLUX_URL"
    DOWNLOADED=$(stat -c%s "$FLUX_SRC" 2>/dev/null || echo 0)
    if [ "$DOWNLOADED" -lt "$MIN_SIZE" ]; then
        echo "[entrypoint] ERROR: Download too small ($DOWNLOADED bytes). Aborting."
        rm -f "$FLUX_SRC"
        exit 1
    fi
    echo "[entrypoint] Download complete: $(numfmt --to=iec $DOWNLOADED)"
fi

ln -sf "$FLUX_SRC" "$FLUX_DST"
echo "[entrypoint] Flux ready at $FLUX_DST"

exec /start.sh
