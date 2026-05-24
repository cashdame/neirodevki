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

# ControlNet preprocessor nodes (depth/pose -> transfer pose + clothing position
# from a reference frame). The node is small; preprocessor weights (DepthAnything,
# ~400 MB) auto-download to its ckpts/ on first use. The ControlNet MODEL itself
# (~4.3 GB) lives on the network volume -- fetched by entrypoint.sh, like Flux.
RUN cd /comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux && \
    pip install --no-cache-dir -r comfyui_controlnet_aux/requirements.txt && \
    pip install --no-cache-dir "opencv-python>=4.8.0" && \
    rm -rf comfyui_controlnet_aux/.git

# Entrypoint: symlinks Flux from volume, then starts the worker
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
