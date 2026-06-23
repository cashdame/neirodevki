"""
Tests for gen_faceswap_serverless.py

Covers:
  1. build_workflow — correct node ids are patched, values match expected names
  2. build_workflow — KeyError if a required node is absent from the workflow JSON
  3. extract_results — correct priority order: videos > gifs > images > files
  4. extract_results — returns [] for empty / non-dict output
  5. _encode_file — encodes real bytes to base64
  6. Payload structure — images[] has two entries with correct names
  7. reactor_faceswap.json — JSON-valid, contains VHS_VideoCombine (node 205),
     frame_load_cap=0, force_rate=0, audio wired from slot 2 of node 201
  8. save_result base64 path — writes decoded bytes to disk
  9. save_result url path — fetches URL and writes to disk (mocked)
 10. _default_out_path — embeds driving video basename in output filename

No real HTTP calls are made; RunPod endpoint is never contacted.
"""

import base64
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
import pathlib
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Load gen_faceswap_serverless as a module from an absolute path so the test
# works regardless of the cwd or sys.path state (mirrors test_check_imports.py
# pattern from the same video-worker/tests/ directory).
# ---------------------------------------------------------------------------

_ENGINE_ROOT = pathlib.Path(__file__).parent.parent.parent  # neirodevki-engine/
_MODULE_FILE = _ENGINE_ROOT / "gen_faceswap_serverless.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gen_faceswap_serverless", _MODULE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module()


class TestBuildWorkflow(unittest.TestCase):
    """build_workflow patches the correct nodes with the correct file names."""

    def test_ref_face_node_patched(self):
        wf = _mod.build_workflow()
        self.assertEqual(wf[_mod.NODE_REF_FACE]["inputs"]["image"], _mod.REF_FACE_NAME)

    def test_driving_video_node_patched(self):
        wf = _mod.build_workflow()
        self.assertEqual(wf[_mod.NODE_DRIVING]["inputs"]["video"], _mod.DRIVING_VID_NAME)

    def test_names_match_upload_constants(self):
        # The name in the workflow node must match what we put in images[].
        # If they diverge, the worker will not find the file.
        wf = _mod.build_workflow()
        self.assertEqual(wf["200"]["inputs"]["image"], "ref_face.png")
        self.assertEqual(wf["201"]["inputs"]["video"], "driving.mp4")

    def test_workflow_is_dict(self):
        wf = _mod.build_workflow()
        self.assertIsInstance(wf, dict)

    def test_missing_node_raises_key_error(self):
        # Simulate workflow drift: node 200 missing.
        bad_wf = {"201": {"inputs": {"video": "x"}}}
        wf_path = _mod._workflow_path()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(bad_wf, tf)
            tmp_path = tf.name
        try:
            with patch.object(_mod, "_workflow_path", return_value=tmp_path):
                with self.assertRaises(KeyError):
                    _mod.build_workflow()
        finally:
            os.unlink(tmp_path)


class TestExtractResults(unittest.TestCase):
    """extract_results returns a normalised list from various worker output shapes."""

    def test_videos_key_preferred(self):
        out = {"videos": [{"type": "base64", "data": "abc"}],
               "images": [{"type": "base64", "data": "xyz"}]}
        self.assertEqual(_mod.extract_results(out), out["videos"])

    def test_gifs_key_fallback(self):
        out = {"gifs": [{"type": "url", "data": "http://x"}]}
        self.assertEqual(_mod.extract_results(out), out["gifs"])

    def test_images_key_fallback(self):
        out = {"images": [{"type": "base64", "data": "img"}]}
        self.assertEqual(_mod.extract_results(out), out["images"])

    def test_files_key_fallback(self):
        out = {"files": [{"type": "s3_url", "data": "s3://x"}]}
        self.assertEqual(_mod.extract_results(out), out["files"])

    def test_empty_dict_returns_empty(self):
        self.assertEqual(_mod.extract_results({}), [])

    def test_non_dict_returns_empty(self):
        self.assertEqual(_mod.extract_results(None), [])
        self.assertEqual(_mod.extract_results("string"), [])
        self.assertEqual(_mod.extract_results(["list"]), [])

    def test_empty_list_value_is_skipped(self):
        # An explicit empty list should not be returned; fall through to next key.
        out = {"videos": [], "images": [{"type": "base64", "data": "x"}]}
        self.assertEqual(_mod.extract_results(out), out["images"])


class TestEncodeFile(unittest.TestCase):
    """_encode_file reads a file and returns valid base64."""

    def test_roundtrip(self):
        payload = b"\x00\x01\x02\x03\xff binary content"
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(payload)
            tmp = tf.name
        try:
            encoded = _mod._encode_file(tmp)
            self.assertEqual(base64.b64decode(encoded), payload)
        finally:
            os.unlink(tmp)


class TestSaveResult(unittest.TestCase):
    """save_result writes result items to disk correctly."""

    def test_base64_write(self):
        payload = b"fake mp4 bytes \x00\x01"
        item = {"type": "base64", "data": base64.b64encode(payload).decode()}
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "out.mp4")
            result = _mod.save_result(item, out_path)
            self.assertEqual(result, out_path)
            with open(out_path, "rb") as f:
                self.assertEqual(f.read(), payload)

    def test_url_fetch(self):
        payload = b"mp4 data from url"
        fake_resp = MagicMock()
        fake_resp.read.return_value = payload
        fake_resp.__enter__ = lambda s: s
        fake_resp.__exit__ = MagicMock(return_value=False)

        item = {"type": "url", "data": "https://example.com/video.mp4"}
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "out.mp4")
            with patch("urllib.request.urlopen", return_value=fake_resp):
                result = _mod.save_result(item, out_path)
            self.assertEqual(result, out_path)
            with open(out_path, "rb") as f:
                self.assertEqual(f.read(), payload)

    def test_s3_url_same_as_url(self):
        payload = b"s3 bytes"
        fake_resp = MagicMock()
        fake_resp.read.return_value = payload
        fake_resp.__enter__ = lambda s: s
        fake_resp.__exit__ = MagicMock(return_value=False)

        item = {"type": "s3_url", "data": "https://s3.example.com/v.mp4"}
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "out.mp4")
            with patch("urllib.request.urlopen", return_value=fake_resp):
                _mod.save_result(item, out_path)
            with open(out_path, "rb") as f:
                self.assertEqual(f.read(), payload)


class TestDefaultOutPath(unittest.TestCase):
    """_default_out_path embeds the driving video basename."""

    def test_basename_in_output(self):
        path = _mod._default_out_path("/some/path/DZ45uPDgTCo.mp4")
        self.assertIn("DZ45uPDgTCo", os.path.basename(path))
        self.assertTrue(path.endswith(".mp4"))

    def test_no_double_ext(self):
        path = _mod._default_out_path("video.mp4")
        # Should not produce "video.mp4.mp4"
        self.assertFalse(os.path.basename(path).endswith(".mp4.mp4"))


class TestReactorFaceswapJson(unittest.TestCase):
    """reactor_faceswap.json structural validation."""

    def setUp(self):
        wf_path = _ENGINE_ROOT / "video-worker" / "workflows" / "reactor_faceswap.json"
        with open(wf_path, encoding="utf-8") as f:
            self.wf = json.load(f)

    def test_json_valid(self):
        # If setUp succeeded, JSON is valid. Just assert we have a dict.
        self.assertIsInstance(self.wf, dict)

    def test_vhs_load_video_frame_load_cap_zero(self):
        node = self.wf["201"]
        self.assertEqual(node["inputs"]["frame_load_cap"], 0,
                         "frame_load_cap must be 0 to load all frames of a 14s reel")

    def test_vhs_load_video_force_rate_zero(self):
        node = self.wf["201"]
        self.assertEqual(node["inputs"]["force_rate"], 0,
                         "force_rate=0 preserves original fps for 1-to-1 reel replication")

    def test_vhs_video_combine_present(self):
        self.assertIn("205", self.wf, "VHS_VideoCombine node 205 must be present")
        self.assertEqual(self.wf["205"]["class_type"], "VHS_VideoCombine")

    def test_vhs_video_combine_audio_from_slot_2(self):
        # Audio output of VHS_LoadVideo is slot 2 (IMAGE=0, frame_count=1, audio=2).
        audio_link = self.wf["205"]["inputs"]["audio"]
        self.assertEqual(audio_link, ["201", 2],
                         "audio must be wired from VHS_LoadVideo (201) slot 2")

    def test_vhs_video_combine_images_from_faceswap(self):
        images_link = self.wf["205"]["inputs"]["images"]
        self.assertEqual(images_link[0], "203",
                         "images must come from ReActorFaceSwap node 203")

    def test_vhs_video_combine_format_mp4(self):
        fmt = self.wf["205"]["inputs"]["format"]
        self.assertEqual(fmt, "video/h264-mp4")

    def test_vhs_video_combine_save_output_true(self):
        self.assertTrue(self.wf["205"]["inputs"]["save_output"])

    def test_ref_face_placeholder_replaced(self):
        # After the redesign, the default image name is ref_face.png (not PLACEHOLDER).
        image_val = self.wf["200"]["inputs"]["image"]
        self.assertNotIn("PLACEHOLDER", image_val)

    def test_driving_video_placeholder_replaced(self):
        video_val = self.wf["201"]["inputs"]["video"]
        self.assertNotIn("PLACEHOLDER", video_val)


class TestPayloadStructure(unittest.TestCase):
    """Verify the job payload has both upload entries with correct names."""

    def _make_payload(self, ref_b64: str, drv_b64: str) -> dict:
        wf = _mod.build_workflow()
        return {
            "workflow": wf,
            "images": [
                {"name": _mod.REF_FACE_NAME,    "image": ref_b64},
                {"name": _mod.DRIVING_VID_NAME, "image": drv_b64},
            ],
        }

    def test_two_images_entries(self):
        p = self._make_payload("AAA=", "BBB=")
        self.assertEqual(len(p["images"]), 2)

    def test_ref_face_name(self):
        p = self._make_payload("AAA=", "BBB=")
        names = [e["name"] for e in p["images"]]
        self.assertIn("ref_face.png", names)

    def test_driving_name(self):
        p = self._make_payload("AAA=", "BBB=")
        names = [e["name"] for e in p["images"]]
        self.assertIn("driving.mp4", names)

    def test_node_names_match_upload_names(self):
        # The names in the patched workflow nodes must equal the upload names.
        p = self._make_payload("AAA=", "BBB=")
        wf = p["workflow"]
        upload_names = {e["name"] for e in p["images"]}
        self.assertIn(wf[_mod.NODE_REF_FACE]["inputs"]["image"], upload_names)
        self.assertIn(wf[_mod.NODE_DRIVING]["inputs"]["video"], upload_names)


if __name__ == "__main__":
    unittest.main()
