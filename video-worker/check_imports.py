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
import sys
import traceback
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
    "ComfyUI-KJNodes",           # required by Wan 2.2 i2v reference workflow
    "ComfyUI-VideoHelperSuite",  # VHS_LoadVideo + video export
    # "ComfyUI-SeedVR2_VideoUpscaler",  # triton GPU-only — verify on pod in Phase B
]


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
    Try to import each custom node by directory path via _import_node_by_path().

    Returns:
        List of (node_dir_name, traceback_str) for each failure.
    """
    failures: list[tuple[str, str]] = []
    for node_dir in node_dirs:
        try:
            _import_node_by_path(node_dir)
        except Exception:
            failures.append((node_dir, traceback.format_exc()))
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
        torch.cuda.is_available = lambda: False
        torch.cuda.current_device = lambda: 0
        torch.cuda.get_device_name = lambda device=None: "cpu-mock"

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
