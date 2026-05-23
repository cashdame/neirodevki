# Flux + Leila/unlock LoRA worker for RunPod Serverless.
#
# Lightweight image: no model baked in. Flux (17 GB) lives on a RunPod
# network volume (ID: 41sfbn01s9, EU-RO-1). entrypoint.sh symlinks it
# into ComfyUI's checkpoints dir at container startup.
#
# LoRAs (183 MB) are baked in -- they're small and change more often.
FROM runpod/worker-comfyui:5.8.5-base

# LoRAs baked in
RUN mkdir -p /comfyui/models/loras
COPY models/loras/leila_lora_v3.safetensors             /comfyui/models/loras/
COPY models/loras/aidmaNSFWunlock-FLUX-V0.2.safetensors /comfyui/models/loras/

# Entrypoint: symlinks Flux from volume, then starts the worker
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
