"""
Submit a faceswap job to the RunPod Serverless ComfyUI video endpoint (EP2).

Takes a reference face image (Leila's photo) and a driving video (source reel),
sends both files to the worker via the images[] transport (worker-comfyui saves
any bytes under the provided name into ComfyUI's input/ directory), patches
reactor_faceswap.json, and saves the returned mp4.

Workflow: reactor_faceswap.json
  200 LoadImage          <- ref_face.png  (base64 upload)
  201 VHS_LoadVideo      <- driving.mp4   (base64 upload; worker writes mp4 bytes verbatim)
  202 ReActorFaceBoost   (GFPGANv1.4, config node)
  203 ReActorFaceSwap    (inswapper_128, per-frame)
  204 SaveImage          (secondary: individual swapped frames)
  205 VHS_VideoCombine   (primary: mp4 + original audio from 201 slot 2)

Follows the same submit /run -> poll /status -> extract_results -> save pattern
as gen_video_serverless.py. High retry count to survive a flaky VPN tunnel.

Env:
  RUNPOD_API_KEY           from env or ~/.claude/.credentials.master.env
  RUNPOD_VIDEO_ENDPOINT_ID the serverless video endpoint id (--endpoint overrides)
  POLL_ITERS               max poll iterations (default 720, ~36 min total @ 3s/iter)

Usage:
  python gen_faceswap_serverless.py REF_FACE_IMAGE DRIVING_VIDEO \\
         [--endpoint ID] [--out PATH]

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

# Workflow node ids — must stay in sync with reactor_faceswap.json.
# A KeyError here means the workflow was edited and these ids drifted;
# fail loud rather than silently send a stale graph.
NODE_REF_FACE = "200"    # LoadImage: source identity (Leila)
NODE_DRIVING  = "201"    # VHS_LoadVideo: driving video (source reel)

# Names under which the worker writes uploaded bytes into ComfyUI input/.
# Must match the node inputs.image / inputs.video values in the patched workflow.
REF_FACE_NAME   = "ref_face.png"
DRIVING_VID_NAME = "driving.mp4"

# Default endpoint (EP2 leila-wan-video).
DEFAULT_ENDPOINT = "0gtqrw3xo6w4uz"

# Default output directory relative to this script.
_DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "leila_faceswap",  # .context/leila_faceswap_<basename>.mp4
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Faceswap driving video with Leila's face via RunPod Serverless"
    )
    p.add_argument("ref_face",    help="path to Leila's reference face image (png/jpg)")
    p.add_argument("driving_video", help="path to the driving source reel (mp4)")
    p.add_argument(
        "--endpoint", default=None,
        help="RunPod video endpoint ID (overrides RUNPOD_VIDEO_ENDPOINT_ID env)"
    )
    p.add_argument(
        "--out", default=None,
        help="output mp4 path (default: .context/leila_faceswap_<basename>.mp4)"
    )
    return p.parse_args()


def _workflow_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "video-worker", "workflows", "reactor_faceswap.json",
    )


def build_workflow() -> dict:
    """Load and patch reactor_faceswap.json: replace PLACEHOLDERs with real names.

    The actual file upload happens via the images[] transport in the job input;
    we only need the node inputs to reference the correct names.
    """
    with open(_workflow_path(), encoding="utf-8") as f:
        wf = json.load(f)

    # Validate expected nodes exist — fail fast if workflow drifted.
    for node_id in (NODE_REF_FACE, NODE_DRIVING):
        if node_id not in wf:
            raise KeyError(
                f"Node '{node_id}' not found in reactor_faceswap.json. "
                "Workflow may have been edited; update NODE_* constants."
            )

    wf[NODE_REF_FACE]["inputs"]["image"] = REF_FACE_NAME
    wf[NODE_DRIVING]["inputs"]["video"]  = DRIVING_VID_NAME
    return wf


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


def _http(url: str, key: str, payload: dict | None = None, retries: int = 10) -> dict:
    """HTTP GET or POST with retry + capped backoff.

    High retry count because a flaky VPN tunnel (ConnectionReset on the way to
    RunPod) must not kill a run — the job lives on the endpoint regardless.
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
    raise RuntimeError("unreachable")  # satisfies type-checkers


def _post(url: str, key: str, payload: dict) -> dict:
    return _http(url, key, payload=payload)


def _get(url: str, key: str) -> dict:
    return _http(url, key)


def _encode_file(path: str) -> str:
    """Read a file and return its base64 representation (str)."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def extract_results(output: object) -> list[dict]:
    """Normalise worker-comfyui output to a list of {type, data} items.

    worker-comfyui surfaces VHS_VideoCombine results under different keys
    depending on version and output type:
      - "videos"  — most common for VHS_VideoCombine (h264-mp4)
      - "gifs"    — some VHS versions route combined video here
      - "images"  — frames/SaveImage; also used by older VHS builds
      - "files"   — generic fallback used by some worker versions

    We check all four, in priority order, and return the first non-empty list.
    If none found we return [] so the caller can detect the empty case and abort.
    """
    if not isinstance(output, dict):
        return []
    for key in ("videos", "gifs", "images", "files"):
        items = output.get(key)
        if items:
            return items
    return []


def _default_out_path(driving_video: str) -> str:
    """Build default output path: .context/leila_faceswap_<basename>.mp4."""
    basename = os.path.splitext(os.path.basename(driving_video))[0]
    # Place next to the .context directory (engine root -> ../.. from script).
    context_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    out_dir = os.path.normpath(context_dir)
    return os.path.join(out_dir, f"leila_faceswap_{basename}.mp4")


def save_result(item: dict, out_path: str) -> str:
    """Write a single result item to out_path and return the path."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    kind = item.get("type", "")
    data = item.get("data", "")

    if kind == "base64":
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(data))
    elif kind in ("s3_url", "url"):
        req = urllib.request.Request(data)
        req.add_header(
            "User-Agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        )
        with urllib.request.urlopen(req, timeout=300) as resp, \
             open(out_path, "wb") as f:
            f.write(resp.read())
    else:
        sys.exit(f"Unknown result type '{kind}' in output item: {item!r}")

    return out_path


def main() -> None:
    ns = parse_args()

    if not os.path.isfile(ns.ref_face):
        sys.exit(f"ref face not found: {ns.ref_face}")
    if not os.path.isfile(ns.driving_video):
        sys.exit(f"driving video not found: {ns.driving_video}")

    eid = ns.endpoint or os.environ.get("RUNPOD_VIDEO_ENDPOINT_ID") or DEFAULT_ENDPOINT
    key = load_key()
    base_url = f"https://api.runpod.ai/v2/{eid}"

    out_path = ns.out or _default_out_path(ns.driving_video)

    print(f"Encoding ref face:     {ns.ref_face}")
    ref_b64 = _encode_file(ns.ref_face)

    print(f"Encoding driving video: {ns.driving_video}")
    drv_b64 = _encode_file(ns.driving_video)

    wf = build_workflow()

    job_input = {
        "workflow": wf,
        # worker-comfyui writes each entry to ComfyUI input/<name> verbatim.
        # mp4 bytes are written as-is; VHS_LoadVideo reads driving.mp4 by name.
        "images": [
            {"name": REF_FACE_NAME,    "image": ref_b64},
            {"name": DRIVING_VID_NAME, "image": drv_b64},
        ],
    }

    print(f"Submitting faceswap to endpoint {eid}  "
          f"(ref={os.path.basename(ns.ref_face)}, "
          f"driving={os.path.basename(ns.driving_video)})...")

    job   = _post(f"{base_url}/run", key, {"input": job_input})
    job_id = job.get("id")
    if not job_id:
        sys.exit(f"No job id in submit response: {json.dumps(job)[:400]}")

    print(f"Job {job_id} submitted. Polling...")

    # Cold start (worker scaled to zero -> image pull + model load) can keep the
    # job IN_QUEUE for ~4 min; subsequent polls are every 3s. Default cap ~36 min.
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
                    f"Job COMPLETED but no video in output.\n"
                    f"Raw output (truncated): {json.dumps(st.get('output'))[:600]}\n"
                    "Check VHS_VideoCombine node output key in worker logs."
                )
            path = save_result(results[0], out_path)
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
