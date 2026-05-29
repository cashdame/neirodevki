"""
Tests for iter22 RIFE patches — vfi_utils.py and rife_arch.py.

Verifies that neither patched file contains a module-level get_torch_device() call.
A module-level call executes at import time (during ComfyUI node registration scan),
before CUDA is fully initialised on the GPU worker. This causes ComfyUI to silently
drop RIFE_VFI from NODE_CLASS_MAPPINGS via its node-loader try/except wrapper.

Run from video-worker/ directory:
    pytest tests/test_rife_patches.py -v
"""

import ast
import pathlib
import re
import unittest

PATCHES_DIR = pathlib.Path(__file__).parent.parent / "patches"
VFI_UTILS_PATH = PATCHES_DIR / "vfi_utils.py"
RIFE_ARCH_PATH = PATCHES_DIR / "rife_arch.py"


def _extract_module_level_calls(source: str) -> list[tuple[int, str]]:
    """
    Return (lineno, line) for statements at module level that look like a
    bare assignment of a get_torch_device() or DEVICE call.

    We use a simple regex rather than full AST walking because we only need
    to detect the specific problematic pattern, not all possible CUDA calls.
    """
    pattern = re.compile(
        r"^\s*(?:DEVICE|device)\s*=\s*get_torch_device\s*\(\s*\)",
    )
    hits = []
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.lstrip()
        # Skip comment lines.
        if stripped.startswith("#"):
            continue
        if pattern.match(line):
            hits.append((i, line.rstrip()))
    return hits


def _is_valid_python(source: str) -> tuple[bool, str]:
    """Return (True, '') if source parses as valid Python, else (False, error)."""
    try:
        ast.parse(source)
        return True, ""
    except SyntaxError as exc:
        return False, str(exc)


class TestRifePatchFiles(unittest.TestCase):
    """Patch files in patches/ exist and are well-formed."""

    def test_vfi_utils_patch_exists(self):
        """patches/vfi_utils.py must exist."""
        self.assertTrue(
            VFI_UTILS_PATH.is_file(),
            f"patches/vfi_utils.py not found at {VFI_UTILS_PATH}",
        )

    def test_rife_arch_patch_exists(self):
        """patches/rife_arch.py must exist."""
        self.assertTrue(
            RIFE_ARCH_PATH.is_file(),
            f"patches/rife_arch.py not found at {RIFE_ARCH_PATH}",
        )

    def test_vfi_utils_is_valid_python(self):
        """patches/vfi_utils.py must be syntactically valid Python."""
        source = VFI_UTILS_PATH.read_text(encoding="utf-8")
        ok, err = _is_valid_python(source)
        self.assertTrue(ok, f"patches/vfi_utils.py has syntax error: {err}")

    def test_rife_arch_is_valid_python(self):
        """patches/rife_arch.py must be syntactically valid Python."""
        source = RIFE_ARCH_PATH.read_text(encoding="utf-8")
        ok, err = _is_valid_python(source)
        self.assertTrue(ok, f"patches/rife_arch.py has syntax error: {err}")


class TestNoModuleLevelCudaCall(unittest.TestCase):
    """Neither patched file may contain a module-level get_torch_device() assignment."""

    def test_vfi_utils_no_module_level_device_assignment(self):
        """patches/vfi_utils.py must not have DEVICE = get_torch_device() at module level."""
        source = VFI_UTILS_PATH.read_text(encoding="utf-8")
        hits = _extract_module_level_calls(source)
        self.assertEqual(
            hits,
            [],
            f"patches/vfi_utils.py still has module-level get_torch_device() call(s): {hits}",
        )

    def test_rife_arch_no_module_level_device_assignment(self):
        """patches/rife_arch.py must not have device = get_torch_device() at module level."""
        source = RIFE_ARCH_PATH.read_text(encoding="utf-8")
        hits = _extract_module_level_calls(source)
        self.assertEqual(
            hits,
            [],
            f"patches/rife_arch.py still has module-level get_torch_device() call(s): {hits}",
        )


class TestLazyGetterInVfiUtils(unittest.TestCase):
    """patches/vfi_utils.py must define _get_device() and use it instead of bare DEVICE."""

    def setUp(self):
        self.source = VFI_UTILS_PATH.read_text(encoding="utf-8")

    def test_lazy_getter_function_defined(self):
        """patches/vfi_utils.py must define the _get_device() function."""
        self.assertIn(
            "def _get_device():",
            self.source,
            "patches/vfi_utils.py must define _get_device() lazy getter",
        )

    def test_no_bare_device_capitalized_usage(self):
        """patches/vfi_utils.py must not have bare DEVICE token (old module-level var)."""
        # Find all non-comment lines containing bare DEVICE (not inside _get_device)
        bad_lines = []
        for i, line in enumerate(self.source.splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            # A bare DEVICE usage: DEVICE followed by ) or , or newline — not part of _get_device
            if re.search(r"\bDEVICE\b", line) and "_get_device" not in line:
                bad_lines.append((i, line.rstrip()))
        self.assertEqual(
            bad_lines,
            [],
            f"patches/vfi_utils.py has bare DEVICE references (should use _get_device()): {bad_lines}",
        )

    def test_get_device_calls_replace_device(self):
        """patches/vfi_utils.py must have at least 5 calls to _get_device()."""
        count = len(re.findall(r"_get_device\(\)", self.source))
        self.assertGreaterEqual(
            count,
            5,
            f"Expected at least 5 _get_device() calls (replacing 5 DEVICE usages), found {count}",
        )


class TestRifeArchUsesTonFlowDevice(unittest.TestCase):
    """patches/rife_arch.py must use tenFlow.device in warp() instead of global device."""

    def setUp(self):
        self.source = RIFE_ARCH_PATH.read_text(encoding="utf-8")

    def test_no_standalone_device_variable_in_warp(self):
        """rife_arch.py warp() must not use bare `device` variable (the old module-level one)."""
        # Find the warp function body and check for bare `device` references
        # (not tenFlow.device, not tenInput.device, not tenGrid.device)
        bad_lines = []
        in_warp = False
        brace_depth = 0
        for i, line in enumerate(self.source.splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("def warp("):
                in_warp = True
            elif in_warp and stripped.startswith("def ") and not stripped.startswith("def warp("):
                # Next function definition — exit warp scope
                in_warp = False
            if not in_warp:
                continue
            if stripped.startswith("#"):
                continue
            # Look for bare device= or device. NOT preceded by tenFlow or other tensors
            # The forbidden pattern: `device=device` or `.to(device)` where device is bare
            if re.search(r"(?<!\w)device\b(?!\.type\b)", line):
                # Allow: tenFlow.device, tenInput.device, tenGrid.device, k = (...device...)
                if "tenFlow.device" not in line and "tenInput.device" not in line:
                    # Check if it's actually a bare device variable usage (assignment-only context)
                    if re.search(r"(?:device=device|\.to\(device\))", line):
                        bad_lines.append((i, line.rstrip()))
        self.assertEqual(
            bad_lines,
            [],
            f"rife_arch.py warp() still has bare `device` variable usage(s): {bad_lines}",
        )

    def test_tenflow_device_used_in_linspace(self):
        """rife_arch.py must pass tenFlow.device to torch.linspace calls in warp()."""
        count = len(re.findall(r"device=tenFlow\.device", self.source))
        self.assertGreaterEqual(
            count,
            2,
            f"Expected at least 2 'device=tenFlow.device' in linspace calls, found {count}",
        )
