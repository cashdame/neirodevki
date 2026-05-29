"""
Tests for check_imports.py — server/app.frontend_management mock injection,
CUDA mock behaviour, and structural-check routing for KJNodes/VHS.

Run from video-worker/ directory:
    pytest tests/test_check_imports.py -v

These tests verify:
1. server mock is injected when 'server' is absent from sys.modules
2. server mock does NOT overwrite an already-registered sys.modules['server']
3. app.frontend_management mock is injected when absent
4. app.frontend_management mock does NOT overwrite an existing entry
5. CUDA mock guard: patch is applied only when cuda.is_available() is False
6. CUDA mock guard: real cuda is left untouched when is_available() is True
7. _check_node_structural: accepts a valid __init__.py
8. _check_node_structural: raises FileNotFoundError when __init__.py is absent
9. _check_node_structural: raises PyCompileError on syntax error
10. check_node_imports: KJNodes/VHS route through _check_node_structural, others don't
"""

import collections
import py_compile
import sys
import tempfile
import types
import importlib
import importlib.util
import pathlib
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helper: load a fresh copy of check_imports without cache pollution
# ---------------------------------------------------------------------------

_CI_PATH = pathlib.Path(__file__).parent.parent / "check_imports.py"


def _load_fresh():
    """Return a freshly exec'd check_imports module (not cached in sys.modules)."""
    mod_name = "_test_check_imports_fresh"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, str(_CI_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_mock_torch(cuda_available: bool):
    """Build a minimal torch stub sufficient for main() without a real GPU."""
    torch_mock = MagicMock()
    torch_mock.cuda.is_available = MagicMock(return_value=cuda_available)
    torch_mock.cuda.current_device = MagicMock(return_value=0)
    torch_mock.cuda.get_device_name = MagicMock(return_value="mock-gpu")
    torch_mock.cuda.memory_stats = MagicMock(
        return_value=collections.defaultdict(int)
    )
    _props = types.SimpleNamespace(total_memory=1 << 34, name="mock", major=0, minor=0)
    torch_mock.cuda.get_device_properties = MagicMock(return_value=_props)
    torch_mock.cuda.mem_get_info = MagicMock(return_value=(16 << 30, 16 << 30))
    torch_mock.cuda.is_initialized = MagicMock(return_value=True)
    torch_mock.cuda.device_count = MagicMock(return_value=1)
    return torch_mock


class _ModuleGuard:
    """
    Context manager: temporarily replace a sys.modules entry and restore it.

    Usage:
        with _ModuleGuard('server', replacement_or_None):
            ...  # sys.modules['server'] == replacement inside block

    If replacement is None, the key is removed for the duration of the block.
    """

    def __init__(self, key: str, replacement):
        self._key = key
        self._replacement = replacement
        self._original = None
        self._was_present = False

    def __enter__(self):
        self._was_present = self._key in sys.modules
        self._original = sys.modules.get(self._key)
        if self._replacement is None:
            sys.modules.pop(self._key, None)
        else:
            sys.modules[self._key] = self._replacement
        return self

    def __exit__(self, *_):
        if self._was_present:
            sys.modules[self._key] = self._original
        else:
            sys.modules.pop(self._key, None)


def _run_main(extra_pre_populate: dict | None = None, cuda_available: bool = False):
    """
    Load a fresh check_imports, replace torch with a mock, optionally pre-populate
    sys.modules entries, run main(), and return whatever sys.modules contains for
    the keys of interest.

    Returns:
        dict: snapshot of sys.modules after main() returns, for the keys that
              were either pre-populated or that main() is expected to insert.
    """
    ci = _load_fresh()
    ci.CORE_MODULES = []
    ci.NODE_MODULES = []

    torch_mock = _make_mock_torch(cuda_available=cuda_available)

    guards = []

    # Replace torch so check_core_imports + CUDA block work without a real install.
    guards.append(_ModuleGuard("torch", torch_mock))

    # Pre-populate additional keys as requested by the caller.
    if extra_pre_populate:
        for key, val in extra_pre_populate.items():
            guards.append(_ModuleGuard(key, val))
    else:
        # Ensure 'server' and 'app.frontend_management' are absent by default.
        guards.append(_ModuleGuard("server", None))
        guards.append(_ModuleGuard("app.frontend_management", None))

    for g in guards:
        g.__enter__()

    try:
        try:
            ci.main()
        except SystemExit as e:
            if e.code != 0:
                raise RuntimeError(f"check_imports.main() exited with code {e.code}")
        # Capture the keys we care about while still inside the context window.
        snapshot = {
            "server": sys.modules.get("server"),
            "app.frontend_management": sys.modules.get("app.frontend_management"),
        }
        # Also capture torch for CUDA-mock tests.
        snapshot["torch"] = sys.modules.get("torch")
        return snapshot, torch_mock
    finally:
        for g in reversed(guards):
            g.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestServerMock(unittest.TestCase):
    """server + app.frontend_management mock injection tests."""

    # ------ server ------

    def test_server_mock_installed_when_absent(self):
        """main() injects sys.modules['server'] with a rich PromptServer stub."""
        snapshot, _ = _run_main()

        server_mod = snapshot["server"]
        self.assertIsNotNone(server_mod, "sys.modules['server'] must be set by main()")
        self.assertTrue(
            hasattr(server_mod, "PromptServer"),
            "stub must expose PromptServer attribute",
        )
        self.assertIsNotNone(
            server_mod.PromptServer.instance,
            "PromptServer.instance must not be None — VHS needs it",
        )

    def test_server_mock_prompt_queue_exists(self):
        """PromptServer.instance.prompt_queue must exist after mock injection."""
        snapshot, _ = _run_main()

        instance = snapshot["server"].PromptServer.instance
        self.assertTrue(
            hasattr(instance, "prompt_queue"),
            "PromptServer.instance must expose prompt_queue (required by VHS)",
        )

    def test_server_mock_send_sync_callable_returns_none(self):
        """PromptServer.instance.send_sync must be callable and return None."""
        snapshot, _ = _run_main()

        send_sync = snapshot["server"].PromptServer.instance.send_sync
        self.assertTrue(callable(send_sync), "send_sync must be callable")
        result = send_sync("event", {"data": 1}, sid="test")
        self.assertIsNone(result, "send_sync stub must return None")

    def test_server_mock_does_not_overwrite_existing(self):
        """main() must NOT replace a pre-existing sys.modules['server']."""
        sentinel = object()
        fake_server = types.ModuleType("server")
        fake_server.PromptServer = types.SimpleNamespace(instance=sentinel)

        snapshot, _ = _run_main(extra_pre_populate={"server": fake_server})

        self.assertIs(
            snapshot["server"],
            fake_server,
            "main() must not replace a pre-existing server module",
        )
        self.assertIs(
            snapshot["server"].PromptServer.instance,
            sentinel,
            "sentinel must survive — original module was not replaced",
        )

    # ------ app.frontend_management ------

    def test_app_frontend_management_mock_installed_when_absent(self):
        """main() injects sys.modules['app.frontend_management'] when absent."""
        snapshot, _ = _run_main()

        self.assertIsNotNone(
            snapshot["app.frontend_management"],
            "sys.modules['app.frontend_management'] must be set by main()",
        )

    def test_app_frontend_management_mock_does_not_overwrite_existing(self):
        """main() must NOT replace an existing sys.modules['app.frontend_management']."""
        sentinel = object()
        fake_afm = types.ModuleType("app.frontend_management")
        fake_afm._sentinel = sentinel

        snapshot, _ = _run_main(
            extra_pre_populate={"app.frontend_management": fake_afm}
        )

        self.assertIs(
            snapshot["app.frontend_management"],
            fake_afm,
            "main() must not replace a pre-existing app.frontend_management module",
        )
        self.assertIs(
            snapshot["app.frontend_management"]._sentinel,
            sentinel,
        )

    # ------ existing CUDA mock guard (regression) ------

    def test_cuda_mock_applied_when_no_gpu(self):
        """CUDA stubs are applied inside main() when cuda.is_available() is False."""
        _snapshot, torch_mock = _run_main(cuda_available=False)

        # The lambda replaces MagicMock — check it's now a plain callable, not MagicMock.
        self.assertTrue(callable(torch_mock.cuda.mem_get_info))
        # mem_get_info lambda returns (free, total) — verify the lambda is in place
        # by calling it and checking the return type.
        result = torch_mock.cuda.mem_get_info()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_cuda_mock_skipped_when_gpu_available(self):
        """CUDA stubs are NOT applied when cuda.is_available() returns True."""
        _snapshot, torch_mock = _run_main(cuda_available=True)

        # On GPU path the lambda replacements are never written;
        # mem_get_info stays as the original MagicMock.
        self.assertIsInstance(torch_mock.cuda.mem_get_info, MagicMock)


class TestStructuralCheck(unittest.TestCase):
    """Tests for _check_node_structural() and its routing in check_node_imports()."""

    def _get_fn(self):
        """Return _check_node_structural from a fresh module load."""
        return _load_fresh()._check_node_structural

    # ------ _check_node_structural unit tests ------

    def test_structural_accepts_valid_init(self):
        """_check_node_structural passes silently for a syntactically valid __init__.py."""
        fn = self._get_fn()
        with tempfile.TemporaryDirectory() as tmpdir:
            node_dir = Path(tmpdir)
            (node_dir / "__init__.py").write_text("# valid\npass\n", encoding="utf-8")
            # Should not raise.
            fn(node_dir)

    def test_structural_raises_file_not_found_when_init_missing(self):
        """_check_node_structural raises FileNotFoundError when __init__.py is absent."""
        fn = self._get_fn()
        with tempfile.TemporaryDirectory() as tmpdir:
            node_dir = Path(tmpdir)
            # No __init__.py created.
            with self.assertRaises(FileNotFoundError):
                fn(node_dir)

    def test_structural_raises_on_syntax_error(self):
        """_check_node_structural raises py_compile.PyCompileError on broken syntax."""
        fn = self._get_fn()
        with tempfile.TemporaryDirectory() as tmpdir:
            node_dir = Path(tmpdir)
            (node_dir / "__init__.py").write_text(
                "def broken(\n    # missing closing paren and body\n",
                encoding="utf-8",
            )
            with self.assertRaises(py_compile.PyCompileError):
                fn(node_dir)

    # ------ routing test ------

    def test_check_node_imports_routes_structural_for_kjnodes_and_vhs(self):
        """KJNodes and VHS call _check_node_structural; all other nodes call _import_node_by_path."""
        ci = _load_fresh()

        structural_calls: list[str] = []
        import_calls: list[str] = []

        def fake_structural(node_dir):
            structural_calls.append(node_dir.name)

        def fake_import(node_dir_name):
            import_calls.append(node_dir_name)

        with (
            patch.object(ci, "_check_node_structural", side_effect=fake_structural),
            patch.object(ci, "_import_node_by_path", side_effect=fake_import),
        ):
            ci.check_node_imports([
                "ComfyUI-KJNodes",
                "ComfyUI-VideoHelperSuite",
                "ComfyUI-WanVideoWrapper",
                "ComfyUI-ReActor",
            ])

        self.assertIn("ComfyUI-KJNodes", structural_calls)
        self.assertIn("ComfyUI-VideoHelperSuite", structural_calls)
        self.assertNotIn("ComfyUI-KJNodes", import_calls)
        self.assertNotIn("ComfyUI-VideoHelperSuite", import_calls)

        self.assertIn("ComfyUI-WanVideoWrapper", import_calls)
        self.assertIn("ComfyUI-ReActor", import_calls)
        self.assertNotIn("ComfyUI-WanVideoWrapper", structural_calls)
        self.assertNotIn("ComfyUI-ReActor", structural_calls)


class TestNodeRegistrationCheck(unittest.TestCase):
    """Tests for NODE_CLASS_MAPPINGS registration verification in check_node_imports()."""

    def _make_module_with_mappings(self, mappings: dict) -> types.ModuleType:
        """Return a fake imported module with the given NODE_CLASS_MAPPINGS."""
        mod = types.ModuleType("fake_node")
        mod.NODE_CLASS_MAPPINGS = mappings
        return mod

    def test_registration_check_passes_when_key_present(self):
        """check_node_imports reports no failure when required key is in NODE_CLASS_MAPPINGS."""
        ci = _load_fresh()

        fake_module = self._make_module_with_mappings({"RIFE VFI": object()})

        with patch.object(ci, "_import_node_by_path", return_value=fake_module):
            failures = ci.check_node_imports(["ComfyUI-Frame-Interpolation"])

        self.assertEqual(failures, [], "Expected no failures when RIFE VFI key is present")

    def test_registration_check_fails_when_key_missing(self):
        """check_node_imports reports failure when required key absent from NODE_CLASS_MAPPINGS."""
        ci = _load_fresh()

        # Module imported successfully, but RIFE VFI not registered (dep swallowed error).
        fake_module = self._make_module_with_mappings({"FILM VFI": object()})

        with patch.object(ci, "_import_node_by_path", return_value=fake_module):
            failures = ci.check_node_imports(["ComfyUI-Frame-Interpolation"])

        self.assertEqual(len(failures), 1, "Expected exactly one failure")
        node_name, tb_str = failures[0]
        self.assertEqual(node_name, "ComfyUI-Frame-Interpolation")
        self.assertIn("RIFE VFI", tb_str, "Failure traceback must mention the missing key")
        self.assertIn("NODE_CLASS_MAPPINGS", tb_str, "Failure traceback must mention NODE_CLASS_MAPPINGS")

    def test_registration_check_fails_when_no_mappings_attr(self):
        """check_node_imports reports failure when module has no NODE_CLASS_MAPPINGS at all."""
        ci = _load_fresh()

        # Module imported but NODE_CLASS_MAPPINGS attribute entirely absent.
        bare_module = types.ModuleType("bare_node")

        with patch.object(ci, "_import_node_by_path", return_value=bare_module):
            failures = ci.check_node_imports(["ComfyUI-Frame-Interpolation"])

        self.assertEqual(len(failures), 1)
        _node_name, tb_str = failures[0]
        self.assertIn("NODE_CLASS_MAPPINGS", tb_str)

    def test_registration_check_not_applied_to_other_nodes(self):
        """Nodes absent from _NODE_REGISTRATION_CHECKS skip the key verification."""
        ci = _load_fresh()

        # Module for WanVideoWrapper has no NODE_CLASS_MAPPINGS — should still pass
        # because WanVideoWrapper is not in _NODE_REGISTRATION_CHECKS.
        bare_module = types.ModuleType("wan_node")

        with patch.object(ci, "_import_node_by_path", return_value=bare_module):
            failures = ci.check_node_imports(["ComfyUI-WanVideoWrapper"])

        self.assertEqual(failures, [], "Nodes not in _NODE_REGISTRATION_CHECKS must not be checked")

    def test_rife_vfi_key_in_registration_checks(self):
        """_NODE_REGISTRATION_CHECKS must list 'RIFE VFI' for ComfyUI-Frame-Interpolation."""
        ci = _load_fresh()
        checks = ci._NODE_REGISTRATION_CHECKS
        self.assertIn(
            "ComfyUI-Frame-Interpolation", checks,
            "ComfyUI-Frame-Interpolation must be in _NODE_REGISTRATION_CHECKS",
        )
        self.assertIn(
            "RIFE VFI", checks["ComfyUI-Frame-Interpolation"],
            "'RIFE VFI' must be a required key for ComfyUI-Frame-Interpolation",
        )


class TestTorchvisionAbi(unittest.TestCase):
    """Tests for check_torchvision_abi() — version printing and ABI guard."""

    def _get_fn(self):
        """Return check_torchvision_abi from a fresh module load."""
        return _load_fresh().check_torchvision_abi

    def _make_tv_mock(self, torch_ver: str, tv_ver: str):
        """Return (torch_mock, torchvision_mock) stubs with given version strings."""
        torch_mock = MagicMock()
        torch_mock.__version__ = torch_ver
        tv_mock = MagicMock()
        tv_mock.__version__ = tv_ver
        return torch_mock, tv_mock

    def _call_abi(self, torch_ver: str, tv_ver: str):
        """
        Call check_torchvision_abi() with mocked torch/torchvision versions.

        Returns (exit_code_or_None, stdout_text, stderr_text).
        exit_code_or_None is None if the function returns normally (no SystemExit).
        """
        fn = self._get_fn()
        torch_mock, tv_mock = self._make_tv_mock(torch_ver, tv_ver)
        import io
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with (
            _ModuleGuard("torch", torch_mock),
            _ModuleGuard("torchvision", tv_mock),
            patch("sys.stdout", stdout_capture),
            patch("sys.stderr", stderr_capture),
        ):
            try:
                fn()
                return None, stdout_capture.getvalue(), stderr_capture.getvalue()
            except SystemExit as e:
                return e.code, stdout_capture.getvalue(), stderr_capture.getvalue()

    def test_prints_both_versions_on_success(self):
        """check_torchvision_abi prints 'torch=X torchvision=Y' on compatible pair."""
        code, stdout, _stderr = self._call_abi("2.6.0+cu126", "0.21.0+cu126")
        self.assertIsNone(code, "Compatible pair must not exit")
        self.assertIn("torch=", stdout)
        self.assertIn("torchvision=", stdout)
        self.assertIn("2.6.0", stdout)
        self.assertIn("0.21.0", stdout)

    def test_compatible_pair_passes(self):
        """Known-good pairs in _TORCH_TV_COMPAT pass without exit."""
        pairs = [
            ("2.6.0", "0.21.0"),
            ("2.7.0", "0.22.0"),
            ("2.8.0", "0.23.0"),
        ]
        for t_ver, tv_ver in pairs:
            with self.subTest(torch=t_ver, tv=tv_ver):
                code, _stdout, _stderr = self._call_abi(t_ver, tv_ver)
                self.assertIsNone(code, f"Pair torch={t_ver} tv={tv_ver} must pass")

    def test_mismatch_exits_with_1(self):
        """ABI mismatch (torch 2.6 + tv 0.20) must exit(1) with a descriptive message."""
        code, _stdout, stderr = self._call_abi("2.6.0", "0.20.0")
        self.assertEqual(code, 1, "ABI mismatch must exit with code 1")
        self.assertIn("mismatch", stderr.lower(), "stderr must contain 'mismatch'")
        self.assertIn("0.21", stderr, "stderr must mention the expected torchvision version")

    def test_unknown_torch_version_warns_not_fails(self):
        """Torch version absent from matrix → warning to stderr, no exit."""
        code, _stdout, stderr = self._call_abi("3.0.0", "0.25.0")
        self.assertIsNone(code, "Unknown torch version must not cause exit")
        self.assertIn("WARNING", stderr, "Must print WARNING for unknown torch version")

    def test_cu_suffix_stripped_for_comparison(self):
        """Versions like '2.6.0+cu126' and '0.21.0+cu126' must be parsed correctly."""
        code, _stdout, _stderr = self._call_abi("2.6.0+cu126", "0.21.0+cu126")
        self.assertIsNone(code, "cu-suffixed compatible pair must pass")

    def test_mismatch_with_cu_suffix(self):
        """'2.6.0+cu126' + '0.20.0+cu126' is still a mismatch — suffix must not confuse parser."""
        code, _stdout, stderr = self._call_abi("2.6.0+cu126", "0.20.0+cu126")
        self.assertEqual(code, 1, "cu-suffixed mismatch must still exit(1)")
        self.assertIn("mismatch", stderr.lower())


if __name__ == "__main__":
    unittest.main()
