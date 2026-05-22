# Flux + Leila/unlock LoRA worker for RunPod Serverless.
#
# Models are baked into the image (not on a network volume) so the serverless
# endpoint can pull workers from ANY datacenter. A network volume would lock the
# endpoint to a single DC -- the exact capacity trap that pushed us off on-demand.
#
# Layer order matters for cheap updates:
#   1. Flux fp8 (~17 GB) -- heavy + stable, downloaded once, reused from cache.
#   2. LoRAs (~183 MB)   -- light + volatile; changing a LoRA rebuilds only this
#                            layer, so Maya / new Leila versions cost minutes.
FROM runpod/worker-comfyui:5.8.5-base

# --- Heavy stable layer: Flux fp8 checkpoint ---------------------------------
# Comfy-Org mirror is public (no HF token needed). Cached via registry buildcache;
# only re-downloads if this RUN instruction text changes.
RUN mkdir -p /comfyui/models/checkpoints /comfyui/models/loras && \
    wget -q --tries=3 --continue \
      -O /comfyui/models/checkpoints/flux1-dev-fp8.safetensors \
      https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors

# --- Light volatile layer: LoRAs ---------------------------------------------
# Keep these COPYs LAST so a LoRA swap doesn't invalidate the Flux layer above.
COPY models/loras/leila_lora_v2.safetensors             /comfyui/models/loras/
COPY models/loras/aidmaNSFWunlock-FLUX-V0.2.safetensors /comfyui/models/loras/
