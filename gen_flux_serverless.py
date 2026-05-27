"""
Generate Leila images via the RunPod Serverless ComfyUI endpoint.

Unlike gen_flux_pod.py (which talks to a ComfyUI running ON a pod), this builds
the same Flux + LoRA workflow and submits it to a serverless endpoint over HTTPS.
The endpoint returns images as base64 (or an S3 URL if S3 is configured), so the
old PTY-SSH base64 corruption is gone -- it's a clean HTTP response.

Env:
  RUNPOD_API_KEY      from env or ~/.claude/.credentials.master.env
  RUNPOD_ENDPOINT_ID  the serverless endpoint id (--endpoint overrides)

Usage:
  python gen_flux_serverless.py "PROMPT" [seed] [width] [height] [--endpoint ID]

Stdlib only.
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

# --- args --------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Generate via RunPod Serverless")
parser.add_argument("prompt", nargs="?", default="leila_woman, portrait")
parser.add_argument("seed", nargs="?", type=int, default=1234)
parser.add_argument("width", nargs="?", type=int, default=768)
parser.add_argument("height", nargs="?", type=int, default=1024)
parser.add_argument("--endpoint", default=None, help="RunPod serverless endpoint ID")
parser.add_argument("--ref-image", default=None,
                    help="reference image -> ControlNet (transfer pose + clothing position)")
parser.add_argument("--cn-strength", type=float, default=0.65, help="ControlNet strength (0.6-0.7)")
ns = parser.parse_args()
prompt, seed, W, H = ns.prompt, ns.seed, ns.width, ns.height
endpoint = ns.endpoint
ref_image, cn_strength = ns.ref_image, ns.cn_strength

# Control image (canny edges) is computed LOCALLY (the comfyui_controlnet_aux
# preprocessor nodes don't load in the worker image), then sent as base64. The
# worker workflow uses only core ComfyUI ControlNet nodes. SetUnionControlNetType
# valid types (from the live API): auto/openpose/depth/hed-pidi-scribble-ted/
# canny-lineart-anime_lineart-mlsd/normal/segment/tile/repaint. Our edge map = canny.
CN_MODE = "canny/lineart/anime_lineart/mlsd"
REF_NAME = "ref_control.png"


def make_canny(path):
    """Canny-style edge map via Pillow (no cv2/torch): grayscale -> FIND_EDGES ->
    autocontrast -> threshold to crisp white lines on black. Returns a temp PNG path."""
    from PIL import Image, ImageFilter, ImageOps
    im = Image.open(path).convert("L")
    edges = ImageOps.autocontrast(im.filter(ImageFilter.FIND_EDGES))
    edges = edges.point(lambda p: 255 if p > 40 else 0).convert("RGB")
    import tempfile
    fd, out = tempfile.mkstemp(suffix=".png", prefix="canny_")
    os.close(fd)
    edges.save(out)
    return out

# --- LoRA stack (mirrors gen_flux_pod.py) ------------------------------------
LEILA_LORA = os.environ.get("LEILA_LORA", "leila_lora_v3.safetensors")
UNLOCK_LORA = "aidmaNSFWunlock-FLUX-V0.2.safetensors"
UNLOCK_STRENGTH = float(os.environ.get("UNLOCK", "0.7"))  # 0 disables the adapter
UNLOCK_TRIGGER = "aidmaNSFWunlock"
# Areola/nipple detail tail. base flux1-dev-fp8 (not the LoRA) renders nipples, so
# explicit positive guidance is what fixes texture/symmetry (cfg=1.0 ignores negatives).
# Only appended on the nude path (unlock active); pointless for SFW/clothed prompts.
NIPPLE_DETAIL = "detailed natural areolas, symmetric nipples, soft realistic areola texture"

use_unlock = UNLOCK_STRENGTH > 0
if use_unlock:
    if UNLOCK_TRIGGER.lower() not in prompt.lower():
        prompt += ", " + UNLOCK_TRIGGER
    if "areola" not in prompt.lower():
        prompt += ", " + NIPPLE_DETAIL

# --- sampler overrides (env; defaults = the A/B-winning "B" recipe) -----------
STEPS = int(os.environ.get("STEPS", "32"))
SAMPLER = os.environ.get("SAMPLER", "dpmpp_2m")
SCHEDULER = os.environ.get("SCHEDULER", "beta")
OUT_TAG = os.environ.get("OUT_TAG", "")  # appended to filename to keep A/B apart
# ControlNet end_percent: how late into denoising the control image keeps steering.
# Lower (~0.5) = pose/composition is locked early but the LAST steps (where the face
# forms) are driven by the LoRA -> identity stays Leila instead of drifting toward the
# reference person's facial geometry. Only used on the --ref-image path.
CN_END = float(os.environ.get("CN_END", "0.5"))


def build_workflow():
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "flux1-dev-fp8.safetensors"}},
        "2": {"class_type": "LoraLoader",
              "inputs": {"model": ["1", 0], "clip": ["1", 1],
                         "lora_name": LEILA_LORA,
                         "strength_model": 1.0, "strength_clip": 1.0}},
        "6": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": W, "height": H, "batch_size": 1}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "leila"}},
    }
    if use_unlock:
        wf["2b"] = {"class_type": "LoraLoader",
                    "inputs": {"model": ["2", 0], "clip": ["2", 1],
                               "lora_name": UNLOCK_LORA,
                               "strength_model": UNLOCK_STRENGTH,
                               "strength_clip": UNLOCK_STRENGTH}}
        last = "2b"
    else:
        last = "2"
    wf["3"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": [last, 1], "text": prompt}}
    wf["4"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": [last, 1], "text": ""}}
    wf["5"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["3", 0], "guidance": 3.5}}

    # ControlNet branch: depth/pose from a reference -> transfer pose + clothing
    # position onto Leila. Only added when --ref-image is given; otherwise plain txt2img.
    pos, neg = ["5", 0], ["4", 0]
    if ref_image:
        # Control image (canny) is already preprocessed locally -> feed straight in.
        wf["10"] = {"class_type": "LoadImage", "inputs": {"image": REF_NAME}}
        wf["12"] = {"class_type": "ControlNetLoader",
                    "inputs": {"control_net_name": "flux_union_pro2.safetensors"}}
        wf["13"] = {"class_type": "SetUnionControlNetType",
                    "inputs": {"control_net": ["12", 0], "type": CN_MODE}}
        wf["14"] = {"class_type": "ControlNetApplyAdvanced",
                    "inputs": {"positive": ["5", 0], "negative": ["4", 0],
                               "control_net": ["13", 0], "image": ["10", 0],
                               "strength": cn_strength, "start_percent": 0.0,
                               "end_percent": CN_END, "vae": ["1", 2]}}
        pos, neg = ["14", 0], ["14", 1]

    wf["7"] = {"class_type": "KSampler",
               "inputs": {"model": [last, 0], "positive": pos, "negative": neg,
                          "latent_image": ["6", 0], "seed": seed, "steps": STEPS, "cfg": 1.0,
                          "sampler_name": SAMPLER, "scheduler": SCHEDULER, "denoise": 1.0}}
    return wf


def load_key():
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key
    cred = os.path.expanduser("~/.claude/.credentials.master.env")
    with open(cred, encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("RUNPOD_API_KEY"):
                return line.split("=", 1)[1].strip()
    sys.exit("RUNPOD_API_KEY not found in env or credentials file")


def _http(url, key, payload=None, retries=10):
    # retries high + capped backoff so a flaky VPN tunnel (ConnectionReset on the
    # way to RunPod) doesn't kill a run -- the job lives on the endpoint regardless.
    for attempt in range(retries):
        try:
            body = json.dumps(payload).encode() if payload is not None else None
            method = "POST" if payload is not None else "GET"
            r = urllib.request.Request(url, data=body, method=method)
            r.add_header("Authorization", "Bearer " + key)
            if body:
                r.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(r, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(min(5 * (attempt + 1), 20))
            else:
                raise


def post(url, key, payload):
    return _http(url, key, payload=payload)


def get(url, key):
    return _http(url, key)


def save_image(img, seed, idx):
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "leila", "serverless_test")
    os.makedirs(out_dir, exist_ok=True)
    kind = img.get("type")
    data = img.get("data", "")
    suffix = f"_{idx}" if idx > 0 else ""
    path = os.path.join(out_dir, f"leila_{seed}{OUT_TAG}{suffix}.png")
    if kind == "base64":
        with open(path, "wb") as f:
            f.write(base64.b64decode(data))
    elif kind == "s3_url":
        with urllib.request.urlopen(data, timeout=120) as resp, open(path, "wb") as f:
            f.write(resp.read())
    else:
        sys.exit(f"Unknown image type: {kind}")
    return path


def main():
    eid = endpoint or os.environ.get("RUNPOD_ENDPOINT_ID")
    if not eid:
        sys.exit("Set RUNPOD_ENDPOINT_ID or pass --endpoint ID")
    key = load_key()
    base = f"https://api.runpod.ai/v2/{eid}"

    job_input = {"workflow": build_workflow()}
    canny_tmp = None
    if ref_image:
        canny_tmp = make_canny(ref_image)  # local edge map -> control image
        with open(canny_tmp, "rb") as f:
            job_input["images"] = [{"name": REF_NAME, "image": base64.b64encode(f.read()).decode()}]
        print(f"Submitting to {eid} (seed={seed}, {W}x{H}, ControlNet canny "
              f"@{cn_strength} from {os.path.basename(ref_image)})...")
    else:
        print(f"Submitting to endpoint {eid} (seed={seed}, {W}x{H})...")
    job = post(f"{base}/run", key, {"input": job_input})
    if canny_tmp:
        os.remove(canny_tmp)
    job_id = job.get("id")
    if not job_id:
        sys.exit(f"No job id in response: {json.dumps(job)[:400]}")

    # Poll. Cold start (worker scaled to zero -> image pull + 17 GB into VRAM) can
    # sit IN_QUEUE for several minutes; warm runs finish in seconds. Default ~15 min
    # cap so a cold first request doesn't get dropped mid-spinup. POLL_ITERS raises it
    # (e.g. for riding out GPU-throttle peaks where the job must stay queued for hours
    # until a worker survives a full cold start and FlashBoot seeds its snapshot).
    max_iters = int(os.environ.get("POLL_ITERS", "450"))
    for i in range(max_iters):  # each iter ~2s
        try:
            st = get(f"{base}/status/{job_id}", key)
        except Exception as e:
            # VPN tunnel dropped mid-poll: the job keeps running on RunPod, so
            # don't abort -- wait and retry the same job_id until it's ready.
            print(f"  (poll network error, retrying: {type(e).__name__})")
            time.sleep(5)
            continue
        status = st.get("status")
        if status == "COMPLETED":
            images = st.get("output", {}).get("images", [])
            if not images:
                sys.exit(f"COMPLETED but no images: {json.dumps(st)[:400]}")
            paths = [save_image(img, seed, idx) for idx, img in enumerate(images)]
            print(f"OK -> {paths[0]}" if len(paths) == 1 else f"OK -> {paths}")
            return
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            sys.exit(f"Job {status}: {json.dumps(st)[:600]}")
        if i == 0 or i % 10 == 0:
            print(f"  {status}... ({i*2}s)")
        time.sleep(2)
    sys.exit("Timed out waiting for job")


if __name__ == "__main__":
    main()
