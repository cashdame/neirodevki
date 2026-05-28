"""
check_imports.py — import smoke-test for the EP2 video worker image.

Run as part of the Docker build (or entrypoint) to detect broken deps early:
  python check_imports.py

Exit 0: all imports succeeded.
Exit 1: one or more imports failed; full traceback printed to stderr.

This fixes the EP1 Critical issue (silent 10-min timeout on import error).
The entrypoint should catch the non-zero exit and write /tmp/worker_error.json
so the RunPod job returns FAILED with a readable message instead of timing out.

Architecture notes (Section 1 of EP2 spec):
  - CORE_MODULES:  pip-installed packages, valid Python identifiers.
                   Imported via importlib.import_module().
  - NODE_MODULES:  custom-node directory names under CUSTOM_NODES_PATH.
                   May contain dashes (invalid Python identifiers), so they
                   are imported via spec_from_file_location / __init__.py.
                   Fill in real paths after GET /object_info on the ComfyUI
                   instance confirms which modules each node exposes
                   (pod-stand TODO Phase B).
"""

import importlib
import importlib.util
import py_compile
import sys
import traceback
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Custom nodes base path — overridable via env var for testing.
# ---------------------------------------------------------------------------
import os

CUSTOM_NODES_PATH = Path(os.environ.get("CF_CUSTOM_NODES_PATH", "/comfyui/custom_nodes"))

# ---------------------------------------------------------------------------
# Core pip dependencies (pinned in Dockerfile-video)
# All entries must be valid Python identifiers (no dashes).
# ---------------------------------------------------------------------------
CORE_MODULES: list[str] = [
    "torch",
    "huggingface_hub",
    "imageio",
    "imageio_ffmpeg",
    "av",                   # PyAV — video I/O
    "onnxruntime",          # onnxruntime-gpu on the pod
    "insightface",
    "timm",
    "einops",
    "onnx",
]

# ---------------------------------------------------------------------------
# Custom node directories under CUSTOM_NODES_PATH.
#
# Names with dashes are NOT valid Python identifiers — importlib.import_module()
# would raise ModuleNotFoundError. Instead we use _import_node_by_path().
#
# SeedVR2 (ComfyUI-SeedVR2_VideoUpscaler) is excluded from CI:
#   its __init__.py imports triton which is GPU-only and unavailable in the
#   GitHub Actions runner. Will be verified manually on pod in Phase B.
# ---------------------------------------------------------------------------
NODE_MODULES: list[str] = [
    "ComfyUI-WanVideoWrapper",
    "ComfyUI-Frame-Interpolation",
    "ComfyUI-LatentSyncWrapper",
    "comfyui_controlnet_aux",
    "ComfyUI-ReActor",
    "ComfyUI-KJNodes",                 # required by Wan 2.2 i2v reference workflow
    "ComfyUI-VideoHelperSuite",        # VHS_LoadVideo + video export
    "ComfyUI-WanAnimatePreprocess",    # iter17: OnnxDetectionModelLoader + PoseAndFaceDetection + DrawViTPose
    # "ComfyUI-SeedVR2_VideoUpscaler",  # triton GPU-only — verify on pod in Phase B
]


# ---------------------------------------------------------------------------
# Nodes that require a fully-initialized PromptServer (aiohttp app/router).
# On CI (no GPU) we cannot reproduce that environment — use a structural
# (syntax-only) check instead of a live import.
# On the production RunPod worker ComfyUI loads the real server via /start.sh
# before custom nodes are scanned, so these nodes import fine in prod.
# ---------------------------------------------------------------------------
_STRUCTURAL_ONLY: frozenset[str] = frozenset({
    "ComfyUI-KJNodes",
    "ComfyUI-VideoHelperSuite",
    # WanAnimatePreprocess nodes.py imports cv2, torch, and comfy.model_management
    # at module level — requires GPU + full ComfyUI server init. Syntax-only check
    # on CI; full load happens at ComfyUI startup on the RunPod worker.
    "ComfyUI-WanAnimatePreprocess",
})


def _check_node_structural(node_dir: Path) -> None:
    """Verify __init__.py exists and is syntactically valid Python, without executing it.

    Used for nodes that require a fully-initialized ComfyUI server (with aiohttp app),
    which cannot be reproduced on a GPU-less CI runner.

    Args:
        node_dir: Path to the custom node directory.

    Raises:
        FileNotFoundError: If __init__.py is not present in node_dir.
        py_compile.PyCompileError: If __init__.py has a syntax error.
    """
    init_file = node_dir / "__init__.py"
    if not init_file.is_file():
        raise FileNotFoundError(f"__init__.py not found in {node_dir}")
    py_compile.compile(str(init_file), doraise=True)


# ---------------------------------------------------------------------------
# Node importer — handles directories with dashes in their names
# ---------------------------------------------------------------------------

def _import_node_by_path(node_dir_name: str):
    """
    Import a ComfyUI custom node by its directory name.

    Uses spec_from_file_location so directory names with dashes (e.g.
    ComfyUI-WanVideoWrapper) work correctly — importlib.import_module()
    would fail on such names because '-' is not valid in Python identifiers.

    Args:
        node_dir_name: Directory name under CUSTOM_NODES_PATH.

    Returns:
        The loaded module object.

    Raises:
        ImportError: If __init__.py is not found, or spec creation fails.
        Any exception raised by the node's __init__.py propagates unchanged.
    """
    init_file = CUSTOM_NODES_PATH / node_dir_name / "__init__.py"
    if not init_file.is_file():
        raise ImportError(
            f"Node __init__.py not found at {init_file}. "
            f"Is the node directory present under {CUSTOM_NODES_PATH}?"
        )

    # Sanitise the directory name to a valid Python identifier for sys.modules.
    module_key = f"_cf_node_{node_dir_name.replace('-', '_')}"

    spec = importlib.util.spec_from_file_location(
        module_key,
        str(init_file),
        submodule_search_locations=[str(init_file.parent)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for node: {node_dir_name}")

    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec_module so circular imports inside the node work.
    sys.modules[module_key] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Check functions — separated so they can be tested independently
# ---------------------------------------------------------------------------

def check_core_imports(modules: list[str]) -> list[tuple[str, str]]:
    """
    Try to import each pip package via importlib.import_module().

    Returns:
        List of (module_name, traceback_str) for each failure.
    """
    failures: list[tuple[str, str]] = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception:
            failures.append((mod, traceback.format_exc()))
    return failures


def check_node_imports(node_dirs: list[str]) -> list[tuple[str, str]]:
    """
    Try to verify each custom node — either by live import or structural check.

    Nodes listed in _STRUCTURAL_ONLY are verified via _check_node_structural()
    (syntax check only, no execution) because they require a fully-initialized
    PromptServer that cannot be reproduced on a GPU-less CI runner.

    All other nodes are verified via _import_node_by_path() (live import).

    Returns:
        List of (node_dir_name, traceback_str) for each failure.
    """
    failures: list[tuple[str, str]] = []
    for node_dir_name in node_dirs:
        try:
            if node_dir_name in _STRUCTURAL_ONLY:
                _check_node_structural(CUSTOM_NODES_PATH / node_dir_name)
            else:
                _import_node_by_path(node_dir_name)
        except Exception:
            failures.append((node_dir_name, traceback.format_exc()))
    return failures


# ---------------------------------------------------------------------------
# Legacy shim — kept for backward-compat with existing tests in
# tests/test_strip_and_verify.py (TestCheckImports.test_check_imports_*)
# ---------------------------------------------------------------------------

def check_imports(modules: list[str]) -> list[tuple[str, str]]:
    """
    Thin compatibility wrapper — routes to check_core_imports.

    Only used by the old TestCheckImports suite that passes raw module name
    strings. New code should call check_core_imports / check_node_imports.
    """
    return check_core_imports(modules)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Inject custom_nodes path so nodes that do sibling-node imports work.
    # Done inside main() to avoid polluting sys.path in unit test contexts.
    custom_nodes_str = str(CUSTOM_NODES_PATH)
    if custom_nodes_str not in sys.path:
        sys.path.insert(0, custom_nodes_str)

    # ------------------------------------------------------------------
    # Inject stub modules to prevent the utils-package name collision.
    #
    # Problem: one of the custom nodes' transitive deps installs the pip
    # package 'utils' (a flat utils.py), which shadows /comfyui/utils/.
    # When Python tries to import the real server.py it walks:
    #   server.py → app/frontend_management.py → from utils.install_util …
    # and resolves 'utils' to the pip package instead of the directory,
    # causing ModuleNotFoundError: 'utils' is not a package.
    #
    # Fix: register lightweight stubs in sys.modules BEFORE loading nodes.
    # Nodes only read server.PromptServer.instance to register web-routes;
    # they never need the real server.py in headless/CI mode.
    #
    # Safety guard: on a real RunPod worker ComfyUI's /start.sh loads
    # server.py BEFORE custom nodes, so sys.modules['server'] is already
    # populated.  The 'if … not in' guard ensures we never overwrite the
    # live module — production inference is unaffected.
    # ------------------------------------------------------------------
    if "server" not in sys.modules:
        # Real /comfyui/server.py cannot be imported without GPU — it transitively
        # initializes torch.cuda via comfy.model_management. On the production
        # RunPod worker, ComfyUI loads the real server through /start.sh BEFORE
        # custom nodes are scanned, so this stub is never used in prod.
        #
        # KJNodes and VHS — the two nodes that need PromptServer.app — are handled
        # via _check_node_structural() (syntax-only), so they never reach this stub.
        _server_stub = types.ModuleType("server")
        _server_stub.PromptServer = types.SimpleNamespace(
            instance=types.SimpleNamespace(
                prompt_queue=types.SimpleNamespace(),
                send_sync=lambda *a, **kw: None,
                loop=None,
            )
        )
        sys.modules["server"] = _server_stub

    if "app.frontend_management" not in sys.modules:
        sys.modules["app.frontend_management"] = types.ModuleType(
            "app.frontend_management"
        )

    total = len(CORE_MODULES) + len(NODE_MODULES)
    if total == 0:
        print("check_imports: no modules to check (both lists empty).")
        sys.exit(0)

    # --- Core pip packages ---
    core_failures = check_core_imports(CORE_MODULES)

    # CUDA mock for CI smoke-test (no GPU on GitHub runner).
    # Lets ComfyUI's model_management.get_torch_device() pick CPU fallback
    # instead of crashing on RuntimeError("Found no NVIDIA driver") when
    # nodes like WanVideoWrapper import comfy.model_management at module load.
    # On real RunPod worker torch.cuda.is_available() returns True and this
    # patch is skipped — runtime behavior is identical to production.
    # Imported here (after check_core_imports) so a missing torch is already
    # reported above before we reach this point.
    import torch
    if not torch.cuda.is_available():
        import collections

        torch.cuda.is_available = lambda: False
        torch.cuda.current_device = lambda: 0
        torch.cuda.get_device_name = lambda device=None: "cpu-mock"

        # comfy/model_management.py reads memory_stats(device) at import time
        # and expects at least 'reserved_bytes.all.current' (and related keys).
        # defaultdict(int) returns 0 for any key access, so all reads are safe.
        torch.cuda.memory_stats = lambda device=None: collections.defaultdict(int)

        # comfy/model_management.py also calls get_device_properties(device).total_memory.
        # 1 << 34 = 16 GiB keeps any downstream division-by-zero away from zero.
        _MockDeviceProps = types.SimpleNamespace(
            total_memory=1 << 34,
            name="cpu-mock",
            major=0,
            minor=0,
        )
        torch.cuda.get_device_properties = lambda device=None: _MockDeviceProps

        # comfy/model_management.py:241 calls mem_get_info(dev) at top-level import
        # via _lazy_init → torch._C._cuda_init(), which fails with
        # "Found no NVIDIA driver" on CI runners without GPU.
        # Returns (free_bytes, total_bytes) — 16 GiB each keeps downstream
        # memory-fraction calculations sane without hitting div-by-zero.
        torch.cuda.mem_get_info = lambda *a, **kw: (16 << 30, 16 << 30)

        # Prophylactic stubs for other lazy_init entry points that may be hit
        # by transitive imports in custom nodes (e.g. controlnet_aux chain).
        # Guard with hasattr so we don't shadow any real callable on a future
        # torch version that adds these to the no-GPU path.
        if not callable(getattr(torch.cuda, "is_initialized", None)):
            torch.cuda.is_initialized = lambda: True
        if not callable(getattr(torch.cuda, "device_count", None)):
            torch.cuda.device_count = lambda: 1

    # Preload the core ComfyUI 'nodes' module so that custom nodes that do
    # 'from nodes import MAX_RESOLUTION' (e.g. KJNodes) pick up the real core
    # module at /comfyui/nodes.py instead of any stray nodes.py shipped inside
    # a custom-node directory (e.g. ComfyUI-ReActor ships its own nodes.py,
    # which would win the import race if custom_nodes/ is on sys.path first).
    try:
        import nodes  # core ComfyUI nodes module — must be found at /comfyui/nodes.py
        print(f"core nodes preloaded: {nodes.__file__}")
    except Exception as e:
        print(
            f"WARNING: failed to preload core nodes module: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        # Do not abort — some custom nodes may still succeed without it.

    # --- Custom nodes ---
    node_failures = check_node_imports(NODE_MODULES)

    all_failures = core_failures + node_failures

    if not all_failures:
        print(
            f"check_imports: OK — {total} modules "
            f"({len(CORE_MODULES)} core, {len(NODE_MODULES)} node) "
            "all imported successfully."
        )
        sys.exit(0)

    # Print summary to stdout so it shows up in container logs.
    print(
        f"check_imports: FAILED — {len(all_failures)}/{total} module(s) "
        "could not be imported.",
        flush=True,
    )

    # Print full tracebacks to stderr so the entrypoint can capture them.
    for mod, tb in all_failures:
        print(f"\n--- FAILED: {mod} ---", file=sys.stderr)
        print(tb, file=sys.stderr)

    sys.exit(1)


if __name__ == "__main__":
    main()
