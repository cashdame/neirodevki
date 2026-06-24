"""
Submit a Wan 2.2 Animate (character replace) job to RunPod Serverless EP2.

Takes a reference image (Leila's photo) and a driving video (source reel),
uploads both via the images[] transport (worker-comfyui writes each to
ComfyUI input/<name>), patches wan_animate.json, polls until COMPLETED, then
assembles the returned PNG frames into an mp4 locally using ffmpeg.

Workflow: video-worker/workflows/wan_animate.json
  57  LoadImage              <- ref (Leila identity: face + hair + body)
  63  VHS_LoadVideo          <- driving video (motion/scene source)
  16  WanVideoTextEncode     <- positive / negative prompt
  89  WanVideoAnimateEmbeds  <- num_frames, width, height
  27  WanVideoSampler        <- seed
  60  SaveImage              <- output frames (surfaced as output.images[])

Worker returns output.images[] as a list of per-frame items with type s3_url
or base64. VHS_VideoCombine is NOT surfaced (confirmed on faceswap run), so we
assemble locally with ffmpeg: ffprobe driving video duration → compute fps →
encode silent mp4 → mux original audio.

Env:
  RUNPOD_API_KEY            from env or ~/.claude/.credentials.master.env
  RUNPOD_VIDEO_ENDPOINT_ID  serverless video endpoint id (--endpoint overrides)
  POLL_ITERS                max poll iterations (default 720, ~36 min @ 3s/iter)

Usage:
  python gen_animate_serverless.py REF_IMAGE DRIVING_VIDEO \\
         [--prompt "..."] [--negative "..."] [--frames N] [--seed N] \\
         [--endpoint ID] [--out PATH]

Stdlib + subprocess(ffmpeg) only.
"""

import argparse
import base64
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

# ---------------------------------------------------------------------------
# Workflow node ids — keep in sync with wan_animate.json.
# A KeyError here means the JSON was edited and these ids drifted; fail loud.
# ---------------------------------------------------------------------------
NODE_REF_IMAGE      = "57"   # LoadImage: Leila reference
NODE_DRIVING_VIDEO  = "63"   # VHS_LoadVideo: driving reel
NODE_TEXT_ENCODE    = "16"   # WanVideoTextEncode: positive/negative prompt
NODE_ANIMATE_EMBEDS = "89"   # WanVideoAnimateEmbeds: num_frames, width, height
NODE_SAMPLER        = "27"   # WanVideoSampler: seed

# Names the worker writes into ComfyUI input/. Must match what we set in
# the node inputs above; if they diverge the worker will not find the file.
REF_IMAGE_NAME   = "input_ref_image.png"
DRIVING_VID_NAME = "driving.mp4"

DEFAULT_ENDPOINT = "0gtqrw3xo6w4uz"

# Canonical Wan 2.2 negative prompt (from kijai wan_i2v reference).
DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，"
    "细节模糊不清，字幕，风格，"
    "作品，画作，画面，静止，"
    "最差质量，低质量，畸形的，"
    "毁容的，画得不好的手部，"
    "多余的手指，杂乱的背景"
)

DEFAULT_POSITIVE = (
    "a woman, smooth natural motion, realistic skin, detailed face"
)

# Browser UA required by Cloudflare-protected S3 presign URLs.
_BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wan 2.2 Animate character-replace via RunPod Serverless"
    )
    p.add_argument("ref_image",    help="path to Leila reference image (png/jpg)")
    p.add_argument("driving_video", help="path to driving source reel (mp4)")
    p.add_argument(
        "--prompt", default=DEFAULT_POSITIVE,
        help="positive prompt (default: motion-focus description)",
    )
    p.add_argument(
        "--negative", default=DEFAULT_NEGATIVE,
        help="negative prompt (default: Wan canonical negative)",
    )
    p.add_argument(
        "--frames", type=int, default=49,
        # ponytail: default 49 = short test clip (~3s @16fps).
        # Animate-14B is heavy; 49 frames already saturates VRAM on a single A100.
        # Upgrade path: increase once smoke confirms pose-detection holds on full reel.
        help="num_frames for WanVideoAnimateEmbeds (default 49 ~= 3s test clip)",
    )
    p.add_argument("--seed", type=int, default=42, help="sampler seed")
    p.add_argument(
        "--endpoint", default=None,
        help="RunPod endpoint ID (overrides RUNPOD_VIDEO_ENDPOINT_ID env)",
    )
    p.add_argument(
        "--out", default=None,
        help="output mp4 path (default: .context/leila_animate_<basename>.mp4)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

def _workflow_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "video-worker", "workflows", "wan_animate.json",
    )


def build_workflow(
    prompt: str,
    negative: str,
    num_frames: int,
    seed: int,
) -> dict:
    """Load wan_animate.json and patch runtime values.

    Fails fast with KeyError if a required node is missing (workflow drift).
    """
    with open(_workflow_path(), encoding="utf-8") as f:
        wf = json.load(f)

    required = (
        NODE_REF_IMAGE,
        NODE_DRIVING_VIDEO,
        NODE_TEXT_ENCODE,
        NODE_ANIMATE_EMBEDS,
        NODE_SAMPLER,
    )
    for node_id in required:
        if node_id not in wf:
            raise KeyError(
                f"Node '{node_id}' not found in wan_animate.json. "
                "Workflow may have been edited; update NODE_* constants."
            )

    wf[NODE_REF_IMAGE]["inputs"]["image"]                      = REF_IMAGE_NAME
    wf[NODE_DRIVING_VIDEO]["inputs"]["video"]                  = DRIVING_VID_NAME
    wf[NODE_TEXT_ENCODE]["inputs"]["positive_prompt"]          = prompt
    wf[NODE_TEXT_ENCODE]["inputs"]["negative_prompt"]          = negative
    wf[NODE_ANIMATE_EMBEDS]["inputs"]["num_frames"]            = num_frames
    wf[NODE_SAMPLER]["inputs"]["seed"]                         = seed

    return wf


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def load_key() -> str:
    """Return RUNPOD_API_KEY from env or ~/.claude/.credentials.master.env."""
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key
    cred = os.path.expanduser("~/.claude/.credentials.master.env")
    try:
        with open(cred, encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("RUNPOD_API_KEY"):
                    return line.split("=", 1)[1].strip()
    except FileNotFoundError:
        pass
    sys.exit("RUNPOD_API_KEY not found in env or ~/.claude/.credentials.master.env")


# ---------------------------------------------------------------------------
# HTTP helpers (mirrors gen_faceswap_serverless.py)
# ---------------------------------------------------------------------------

def _http(url: str, key: str, payload: dict | None = None, retries: int = 10) -> dict:
    """GET or POST with retry + capped backoff.

    High retry count because a flaky VPN tunnel must not abort a multi-minute
    job that is already running on the endpoint.
    """
    for attempt in range(retries):
        try:
            body   = json.dumps(payload).encode() if payload is not None else None
            method = "POST" if payload is not None else "GET"
            req    = urllib.request.Request(url, data=body, method=method)
            req.add_header("Authorization", "Bearer " + key)
            if body:
                req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            if attempt < retries - 1:
                time.sleep(min(5 * (attempt + 1), 20))
            else:
                raise
    raise RuntimeError("unreachable")


def _post(url: str, key: str, payload: dict) -> dict:
    return _http(url, key, payload=payload)


def _get(url: str, key: str) -> dict:
    return _http(url, key)


def _encode_file(path: str) -> str:
    """Read a file and return its base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ---------------------------------------------------------------------------
# Output normalisation
# ---------------------------------------------------------------------------

def extract_results(output: object) -> list[dict]:
    """Normalise worker-comfyui output to a list of {type, data, filename} items.

    Animate uses SaveImage (node 60), which surfaces under 'images'.
    We check all known keys in priority order so the function is safe even if
    a future worker version reroutes output.
    Priority: videos > gifs > images > files.
    """
    if not isinstance(output, dict):
        return []
    for key in ("videos", "gifs", "images", "files"):
        items = output.get(key)
        if items:
            return items
    return []


# ---------------------------------------------------------------------------
# Frame assembly (the core of the Animate path)
# ---------------------------------------------------------------------------

def _ffprobe_duration(video_path: str) -> float:
    """Return video duration in seconds via ffprobe. Returns 0.0 on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _download_frame(url: str, dest_path: str) -> None:
    """Download a single frame from a Cloudflare-protected S3 presign URL."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", _BROWSER_UA)
    with urllib.request.urlopen(req, timeout=120) as resp, \
         open(dest_path, "wb") as f:
        f.write(resp.read())


def assemble_video_from_frames(
    images: list[dict],
    driving_video: str,
    out_path: str,
) -> str:
    """Download frames, encode to silent mp4, mux original audio from driving_video.

    Steps:
      1. Sort frames by filename.
      2. Download each frame (s3_url → HTTP fetch; base64 → decode) to tmp dir.
      3. Compute fps = len(frames) / ffprobe(driving_video). Fallback: 16.
      4. ffmpeg: frames → silent mp4 (libx264, yuv420p, crf 18).
      5. ffmpeg: mux audio from driving_video (copy video, aac audio, -shortest).
      6. Clean tmp dir. Return out_path (or silent if audio mux failed).

    If ffmpeg is not found: prints WARNING to stderr and returns the path to the
    first downloaded frame dir (no mp4). Does not raise.
    """
    if not images:
        sys.exit("assemble_video_from_frames: no frames in output.images")

    # Step 1 — sort by filename so ffmpeg gets sequential frames.
    sorted_images = sorted(images, key=lambda x: x.get("filename", ""))

    tmp_dir = tempfile.mkdtemp(prefix="animate_frames_")
    try:
        # Step 2 — download / decode all frames into tmp_dir.
        print(f"  Downloading {len(sorted_images)} frames to {tmp_dir} ...")
        for idx, item in enumerate(sorted_images):
            frame_path = os.path.join(tmp_dir, f"f{idx:05d}.png")
            kind = item.get("type", "")
            data = item.get("data", "")

            if kind in ("s3_url", "url"):
                _download_frame(data, frame_path)
            elif kind == "base64":
                with open(frame_path, "wb") as f:
                    f.write(base64.b64decode(data))
            else:
                print(
                    f"WARNING: unknown frame type '{kind}' for item {idx}, skipping",
                    file=sys.stderr,
                )
                continue

        # Step 3 — compute fps.
        duration = _ffprobe_duration(driving_video)
        if duration > 0:
            fps = len(sorted_images) / duration
        else:
            fps = 16.0  # ponytail: VHS_LoadVideo force_rate=16; safe fallback
            print(
                "WARNING: could not probe driving video duration, using 16 fps",
                file=sys.stderr,
            )

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        silent_mp4 = out_path.replace(".mp4", "_silent.mp4")

        # Step 4 — encode frames → silent mp4.
        frame_pattern = os.path.join(tmp_dir, "f%05d.png")
        encode_cmd = [
            "ffmpeg", "-y",
            "-framerate", f"{fps:.4f}",
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            silent_mp4,
        ]
        try:
            subprocess.run(encode_cmd, check=True, capture_output=True)
            print(f"  Encoded silent mp4 -> {silent_mp4}")
        except FileNotFoundError:
            print(
                "WARNING: ffmpeg not found — frames saved to tmp dir but no mp4 assembled.",
                file=sys.stderr,
            )
            return tmp_dir  # caller gets something rather than nothing
        except subprocess.CalledProcessError as exc:
            print(
                f"WARNING: ffmpeg encode failed (rc={exc.returncode})\n"
                f"  stderr: {exc.stderr.decode(errors='replace')[:400]}",
                file=sys.stderr,
            )
            return tmp_dir

        # Step 5 — mux audio from driving video.
        # -map "1:a:0?" — the quotes around the stream specifier are required
        # so that zsh does not glob-expand the '?'. subprocess passes args as a
        # list so shell expansion is not an issue here; the '?' is intentional
        # (makes the audio stream optional so silent sources don't fail).
        mux_cmd = [
            "ffmpeg", "-y",
            "-i", silent_mp4,
            "-i", driving_video,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            out_path,
        ]
        try:
            subprocess.run(mux_cmd, check=True, capture_output=True)
            print(f"  Muxed audio -> {out_path}")
        except FileNotFoundError:
            print(
                "WARNING: ffmpeg not found for audio mux — output is silent mp4.",
                file=sys.stderr,
            )
            import shutil
            shutil.copy2(silent_mp4, out_path)
        except subprocess.CalledProcessError as exc:
            print(
                f"WARNING: audio mux failed (rc={exc.returncode}) — output is silent mp4.\n"
                f"  stderr: {exc.stderr.decode(errors='replace')[:400]}",
                file=sys.stderr,
            )
            import shutil
            shutil.copy2(silent_mp4, out_path)
        finally:
            # Clean up intermediate silent file.
            if os.path.isfile(silent_mp4) and silent_mp4 != out_path:
                os.unlink(silent_mp4)

    finally:
        # Step 6 — clean tmp dir.
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return out_path


# ---------------------------------------------------------------------------
# Default output path
# ---------------------------------------------------------------------------

def _default_out_path(driving_video: str) -> str:
    """Build default output: .context/leila_animate_<basename>.mp4."""
    basename = os.path.splitext(os.path.basename(driving_video))[0]
    # Engine root is this script's dir; .context is one level up from engine root.
    context_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    return os.path.join(context_dir, f"leila_animate_{basename}.mp4")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ns = parse_args()

    if not os.path.isfile(ns.ref_image):
        sys.exit(f"ref image not found: {ns.ref_image}")
    if not os.path.isfile(ns.driving_video):
        sys.exit(f"driving video not found: {ns.driving_video}")

    eid = ns.endpoint or os.environ.get("RUNPOD_VIDEO_ENDPOINT_ID") or DEFAULT_ENDPOINT
    key = load_key()
    base_url = f"https://api.runpod.ai/v2/{eid}"

    out_path = ns.out or _default_out_path(ns.driving_video)

    print(f"Encoding ref image:      {ns.ref_image}")
    ref_b64 = _encode_file(ns.ref_image)

    print(f"Encoding driving video:  {ns.driving_video}")
    drv_b64 = _encode_file(ns.driving_video)

    wf = build_workflow(ns.prompt, ns.negative, ns.frames, ns.seed)

    job_input = {
        "workflow": wf,
        # worker-comfyui writes each entry to ComfyUI input/<name> verbatim.
        # mp4 bytes are written as-is; VHS_LoadVideo reads driving.mp4 by name.
        "images": [
            {"name": REF_IMAGE_NAME,   "image": ref_b64},
            {"name": DRIVING_VID_NAME, "image": drv_b64},
        ],
    }

    print(
        f"Submitting Animate to endpoint {eid}  "
        f"(frames={ns.frames}, seed={ns.seed}, "
        f"ref={os.path.basename(ns.ref_image)}, "
        f"driving={os.path.basename(ns.driving_video)})..."
    )

    job    = _post(f"{base_url}/run", key, {"input": job_input})
    job_id = job.get("id")
    if not job_id:
        sys.exit(f"No job id in submit response: {json.dumps(job)[:400]}")

    print(f"Job {job_id} submitted. Polling (cold start up to ~5 min)...")

    # Cold start: worker scaled to zero → image pull + Animate-14B load.
    # Default cap ~36 min; POLL_ITERS env overrides.
    max_iters = int(os.environ.get("POLL_ITERS", "720"))
    for i in range(max_iters):
        try:
            st = _get(f"{base_url}/status/{job_id}", key)
        except Exception as exc:
            print(f"  (poll network error, retrying: {type(exc).__name__})")
            time.sleep(5)
            continue

        status = st.get("status")

        if status == "COMPLETED":
            results = extract_results(st.get("output", {}))
            if not results:
                sys.exit(
                    f"Job COMPLETED but no frames in output.\n"
                    f"Raw output (truncated): {json.dumps(st.get('output'))[:600]}\n"
                    "Check SaveImage node 60 and worker-comfyui output keys."
                )
            print(f"  COMPLETED — {len(results)} frame(s) in output.images.")
            path = assemble_video_from_frames(results, ns.driving_video, out_path)
            print(f"OK -> {path}")
            return

        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            sys.exit(f"Job {status}: {json.dumps(st)[:600]}")

        if i == 0 or i % 10 == 0:
            print(f"  {status}... ({i * 3}s elapsed)")
        time.sleep(3)

    sys.exit(f"Timed out after {max_iters * 3}s waiting for job {job_id}")


if __name__ == "__main__":
    main()
