# flux-worker — Leila image generation on RunPod Serverless

ComfyUI worker with Flux fp8 + Leila/unlock LoRA **baked into the image**, so the
serverless endpoint runs in any datacenter (no network-volume DC lock — that lock
is what stranded us on on-demand when EU-RO ran out of 4090s).

## How it fits together

```
GitHub repo (this) --push--> GitHub Actions --build--> GHCR image
                                                          |
                                  RunPod Serverless endpoint pulls image
                                                          |
        gen_flux_serverless.py  --HTTPS /run-->  endpoint  --base64 image-->  local file
```

- `Dockerfile` — `runpod/worker-comfyui:5.8.5-base` + Flux fp8 (heavy stable layer)
  + LoRAs (light volatile layer, last so swaps are cheap).
- `.github/workflows/build.yml` — builds & pushes to `ghcr.io/<you>/<repo>:latest`
  on every push to `main`. Uses registry buildcache so LoRA-only changes don't
  re-download the 17 GB Flux layer.
- `models/loras/*.safetensors` — tracked via Git LFS (see `.gitattributes`).
- `gen_flux_serverless.py` — local client; builds the workflow and calls the endpoint.

## First-time setup (what Markus does)

1. Create a **private** repo on github.com (e.g. `flux-worker`). Don't init with a README.
2. Tell me the repo URL — I'll wire up `git`, LFS, and push from `D:\claude\neirodevochki\flux-worker\`.
3. GitHub Actions builds the image automatically (~30–50 min first run; the Flux
   download dominates). No secrets needed — push to GHCR uses the built-in
   `GITHUB_TOKEN`.
4. I create the serverless endpoint via the RunPod API, pointing it at the GHCR
   image (private image → I add GHCR pull credentials to the endpoint once).
5. Test: `python gen_flux_serverless.py "leila_woman, portrait" --endpoint=<id>`.

## Updating a LoRA later (Maya, Leila v3, …)

1. Drop the new `.safetensors` into `models/loras/` (and adjust names in
   `gen_flux_serverless.py` if the filename changes).
2. `git add` + `git commit` + `git push`.
3. Actions rebuilds **only the LoRA layer** (~minutes) and pushes a new `:latest`.
4. The endpoint picks up the new image on its next cold worker.

## Notes

- Flux comes from the public Comfy-Org mirror — no HuggingFace token required.
- `aidmaNSFWunlock` (civitai 780667) fixes weak base-Flux anatomy; strength via
  `UNLOCK` env (default 0.55, `UNLOCK=0` disables).
- If GitHub's free LFS quota (1 GB bandwidth/mo) becomes a bottleneck, move the
  LoRAs to a private HuggingFace repo and download them in the Dockerfile instead.
