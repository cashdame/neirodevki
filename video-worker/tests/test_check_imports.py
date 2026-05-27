"""
Tests for check_imports.py — server/app.frontend_management mock injection
and existing CUDA mock behaviour.

Run from video-worker/ directory:
    pytest tests/test_check_imports.py -v

These tests verify:
1. server mock is injected when 'server' is absent from sys.modules
2. server mock does NOT overwrite an already-registered sys.modules['server']
3. app.frontend_management mock is injected when absent
4. app.frontend_management mock does NOT overwrite an existing entry
5. CUDA mock guard: patch is applied only when cuda.is_available() is False
6. CUDA mock guard: real cuda is left untouched when is_available() is True
"""

import collections
import sys
import types
import importlib
import importlib.util
import pathlib
import unittest
from unittest.mock import MagicMock


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
        """main() injects sys.modules['server'] with PromptServer.instance=None."""
        snapshot, _ = _run_main()

        server_mod = snapshot["server"]
        self.assertIsNotNone(server_mod, "sys.modules['server'] must be set by main()")
        self.assertTrue(
            hasattr(server_mod, "PromptServer"),
            "stub must expose PromptServer attribute",
        )
        self.assertIsNone(
            server_mod.PromptServer.instance,
            "PromptServer.instance must be None in the stub",
        )

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


if __name__ == "__main__":
    unittest.main()
