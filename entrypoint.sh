#!/bin/bash
set -e

VOLUME="/runpod-volume"
FLUX_SRC="$VOLUME/models/checkpoints/flux1-dev-fp8.safetensors"
FLUX_DST="/comfyui/models/checkpoints/flux1-dev-fp8.safetensors"

mkdir -p "$VOLUME/models/checkpoints"
mkdir -p "$(dirname "$FLUX_DST")"

if [ ! -f "$FLUX_SRC" ]; then
    echo "[entrypoint] Flux not found on volume, downloading..."
    wget -q --tries=3 --continue \
        -O "$FLUX_SRC" \
        https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors
fi

ln -sf "$FLUX_SRC" "$FLUX_DST"
echo "[entrypoint] Flux ready: $(ls -lh "$FLUX_DST")"

exec /start.sh
