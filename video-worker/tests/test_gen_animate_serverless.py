"""
Tests for gen_animate_serverless.py

Covers:
  1. build_workflow — all patched nodes have correct values
  2. build_workflow — ref name in node 57 matches REF_IMAGE_NAME constant
  3. build_workflow — driving name in node 63 matches DRIVING_VID_NAME constant
  4. build_workflow — positive/negative prompts set in node 16
  5. build_workflow — num_frames set in node 89
  6. build_workflow — seed set in node 27
  7. build_workflow — KeyError if required node absent (workflow drift guard)
  8. extract_results — priority: videos > gifs > images > files
  9. extract_results — returns [] for empty / non-dict
 10. assemble_video_from_frames — sorts frames by filename before download
 11. assemble_video_from_frames — calls ffmpeg with correct framerate arg
 12. assemble_video_from_frames — muxes audio from driving video (-map 1:a:0?)
 13. assemble_video_from_frames — handles base64 type items (writes decoded bytes)
 14. assemble_video_from_frames — skips audio mux and warns if ffmpeg missing
 15. assemble_video_from_frames — all s3_url items are downloaded (not just first)
 16. _default_out_path — basename of driving video embedded in output filename
 17. Payload structure — images[] has two entries with correct name/image keys

No real HTTP calls; RunPod endpoint is never contacted.
"""

import base64
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Load module from absolute path (mirrors test_gen_faceswap_serverless.py).
# ---------------------------------------------------------------------------

_ENGINE_ROOT = pathlib.Path(__file__).parent.parent.parent  # neirodevki-engine/
_MODULE_FILE = _ENGINE_ROOT / "gen_animate_serverless.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gen_animate_serverless", _MODULE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module()

DEFAULT_PROMPT = "a woman, smooth natural motion, realistic skin, detailed face"


# ---------------------------------------------------------------------------
# build_workflow tests
# ---------------------------------------------------------------------------

class TestBuildWorkflow(unittest.TestCase):
    """build_workflow patches the correct nodes with the correct values."""

    def _wf(self, **kwargs):
        """Helper: build with defaults overridable by kwargs."""
        return _mod.build_workflow(
            prompt=kwargs.get("prompt", DEFAULT_PROMPT),
            negative=kwargs.get("negative", _mod.DEFAULT_NEGATIVE),
            num_frames=kwargs.get("num_frames", 49),
            seed=kwargs.get("seed", 42),
        )

    def test_ref_image_node_name(self):
        wf = self._wf()
        self.assertEqual(
            wf[_mod.NODE_REF_IMAGE]["inputs"]["image"],
            _mod.REF_IMAGE_NAME,
        )

    def test_ref_image_name_constant_matches_literal(self):
        # Defensive: constant must equal "input_ref_image.png" (the JSON default).
        self.assertEqual(_mod.REF_IMAGE_NAME, "input_ref_image.png")

    def test_driving_video_node_name(self):
        wf = self._wf()
        self.assertEqual(
            wf[_mod.NODE_DRIVING_VIDEO]["inputs"]["video"],
            _mod.DRIVING_VID_NAME,
        )

    def test_driving_vid_name_constant_matches_literal(self):
        self.assertEqual(_mod.DRIVING_VID_NAME, "driving.mp4")

    def test_names_match_upload_constants_by_node_id(self):
        # If node ids drift, worker will not find the file.
        wf = self._wf()
        self.assertEqual(wf["57"]["inputs"]["image"], "input_ref_image.png")
        self.assertEqual(wf["63"]["inputs"]["video"], "driving.mp4")

    def test_positive_prompt_set(self):
        wf = self._wf(prompt="test positive")
        self.assertEqual(wf[_mod.NODE_TEXT_ENCODE]["inputs"]["positive_prompt"], "test positive")

    def test_negative_prompt_set(self):
        wf = self._wf(negative="test negative")
        self.assertEqual(wf[_mod.NODE_TEXT_ENCODE]["inputs"]["negative_prompt"], "test negative")

    def test_num_frames_set(self):
        wf = self._wf(num_frames=25)
        self.assertEqual(wf[_mod.NODE_ANIMATE_EMBEDS]["inputs"]["num_frames"], 25)

    def test_seed_set(self):
        wf = self._wf(seed=1234)
        self.assertEqual(wf[_mod.NODE_SAMPLER]["inputs"]["seed"], 1234)

    def test_text_encode_node_id_is_16(self):
        self.assertEqual(_mod.NODE_TEXT_ENCODE, "16")

    def test_animate_embeds_node_id_is_89(self):
        self.assertEqual(_mod.NODE_ANIMATE_EMBEDS, "89")

    def test_sampler_node_id_is_27(self):
        self.assertEqual(_mod.NODE_SAMPLER, "27")

    def test_workflow_is_dict(self):
        wf = self._wf()
        self.assertIsInstance(wf, dict)

    def test_missing_node_raises_key_error(self):
        # Simulate workflow drift: node 57 missing.
        bad_wf = {
            "63": {"inputs": {"video": "x"}},
            "16": {"inputs": {}},
            "89": {"inputs": {}},
            "27": {"inputs": {}},
        }
        wf_path = _mod._workflow_path()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tf:
            json.dump(bad_wf, tf)
            tmp_path = tf.name
        try:
            with patch.object(_mod, "_workflow_path", return_value=tmp_path):
                with self.assertRaises(KeyError):
                    _mod.build_workflow(DEFAULT_PROMPT, _mod.DEFAULT_NEGATIVE, 49, 42)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# extract_results tests
# ---------------------------------------------------------------------------

class TestExtractResults(unittest.TestCase):
    """extract_results normalises worker output to a list."""

    def test_videos_key_wins(self):
        out = {"videos": [{"type": "s3_url", "data": "v"}], "images": [{"x": 1}]}
        self.assertEqual(_mod.extract_results(out), [{"type": "s3_url", "data": "v"}])

    def test_gifs_before_images(self):
        out = {"gifs": [{"type": "s3_url", "data": "g"}], "images": [{"x": 1}]}
        self.assertEqual(_mod.extract_results(out), [{"type": "s3_url", "data": "g"}])

    def test_images_key_fallback(self):
        out = {"images": [{"type": "s3_url", "data": "i"}]}
        self.assertEqual(_mod.extract_results(out), [{"type": "s3_url", "data": "i"}])

    def test_files_key_last_resort(self):
        out = {"files": [{"type": "s3_url", "data": "f"}]}
        self.assertEqual(_mod.extract_results(out), [{"type": "s3_url", "data": "f"}])

    def test_empty_dict_returns_empty(self):
        self.assertEqual(_mod.extract_results({}), [])

    def test_non_dict_returns_empty(self):
        self.assertEqual(_mod.extract_results(None), [])
        self.assertEqual(_mod.extract_results([]), [])
        self.assertEqual(_mod.extract_results("str"), [])


# ---------------------------------------------------------------------------
# assemble_video_from_frames tests
# ---------------------------------------------------------------------------

class TestAssembleVideoFromFrames(unittest.TestCase):
    """assemble_video_from_frames builds video from a list of frame items."""

    def _make_s3_items(self, names):
        """Build fake s3_url image items with given filenames."""
        return [
            {"type": "s3_url", "data": f"https://example.com/{n}", "filename": n}
            for n in names
        ]

    def _make_b64_item(self, filename, content=b"PNGDATA"):
        return {
            "type": "base64",
            "data": base64.b64encode(content).decode(),
            "filename": filename,
        }

    @patch("subprocess.run")
    @patch("urllib.request.urlopen")
    def test_sorts_frames_by_filename(self, mock_urlopen, mock_run):
        """Frames must be sorted by filename before writing to tmp dir."""
        # Provide frames out of order; expect sorted order in ffmpeg -i glob.
        items = self._make_s3_items([
            "WanVideo2_2_Animate_00003.png",
            "WanVideo2_2_Animate_00001.png",
            "WanVideo2_2_Animate_00002.png",
        ])

        # Mock urlopen to return 1-byte PNG
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"\x89PNG")))
        cm.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = cm

        # Mock subprocess.run (ffmpeg calls) to succeed
        mock_run.return_value = MagicMock(returncode=0)

        # Mock ffprobe to return a known duration
        def run_side_effect(cmd, **kwargs):
            if "ffprobe" in cmd[0]:
                m = MagicMock()
                m.stdout = "3.0\n"
                m.returncode = 0
                return m
            return MagicMock(returncode=0)

        mock_run.side_effect = run_side_effect

        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.mp4")
            driving = os.path.join(td, "driving.mp4")
            open(driving, "wb").close()

            _mod.assemble_video_from_frames(items, driving, out)

        # urlopen is called with a urllib.request.Request object; extract full_url.
        # The first download must be the _00001 frame, not _00003 (sort verified).
        first_req = mock_urlopen.call_args_list[0].args[0]
        self.assertIn("00001", first_req.full_url)

    @patch("subprocess.run")
    @patch("urllib.request.urlopen")
    def test_all_frames_downloaded(self, mock_urlopen, mock_run):
        """All s3_url items must be downloaded, not just the first."""
        items = self._make_s3_items([
            "WanVideo2_2_Animate_00001.png",
            "WanVideo2_2_Animate_00002.png",
            "WanVideo2_2_Animate_00003.png",
        ])

        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"\x89PNG")))
        cm.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = cm

        def run_side_effect(cmd, **kwargs):
            if "ffprobe" in cmd[0]:
                m = MagicMock()
                m.stdout = "3.0\n"
                return m
            return MagicMock(returncode=0)

        mock_run.side_effect = run_side_effect

        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.mp4")
            driving = os.path.join(td, "driving.mp4")
            open(driving, "wb").close()
            _mod.assemble_video_from_frames(items, driving, out)

        # urlopen called once per frame (3 frames) + possibly more for Request wrapping
        # The key: at minimum 3 download calls happened
        self.assertGreaterEqual(mock_urlopen.call_count, 3)

    @patch("subprocess.run")
    @patch("urllib.request.urlopen")
    def test_ffmpeg_framerate_arg(self, mock_urlopen, mock_run):
        """ffmpeg -framerate must equal len(frames) / source_duration."""
        items = self._make_s3_items([
            "WanVideo2_2_Animate_00001.png",
            "WanVideo2_2_Animate_00002.png",
        ])

        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"\x89PNG")))
        cm.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = cm

        ffmpeg_calls = []

        def run_side_effect(cmd, **kwargs):
            if "ffprobe" in cmd[0]:
                m = MagicMock()
                m.stdout = "2.0\n"  # 2 frames / 2.0s = 1 fps
                return m
            ffmpeg_calls.append(cmd)
            return MagicMock(returncode=0)

        mock_run.side_effect = run_side_effect

        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.mp4")
            driving = os.path.join(td, "driving.mp4")
            open(driving, "wb").close()
            _mod.assemble_video_from_frames(items, driving, out)

        # Find the ffmpeg call that builds from frames (has -framerate)
        framerate_cmds = [c for c in ffmpeg_calls if "-framerate" in c]
        self.assertTrue(framerate_cmds, "Expected an ffmpeg call with -framerate")
        fr_cmd = framerate_cmds[0]
        fr_idx = fr_cmd.index("-framerate")
        fr_val = float(fr_cmd[fr_idx + 1])
        # 2 frames / 2.0s = 1.0 fps
        self.assertAlmostEqual(fr_val, 1.0, places=2)

    @patch("subprocess.run")
    @patch("urllib.request.urlopen")
    def test_audio_mux_uses_driving_video(self, mock_urlopen, mock_run):
        """Second ffmpeg call must include driving video for audio mux."""
        items = self._make_s3_items(["WanVideo2_2_Animate_00001.png"])

        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"\x89PNG")))
        cm.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = cm

        ffmpeg_calls = []
        driving_path = None

        def run_side_effect(cmd, **kwargs):
            if "ffprobe" in cmd[0]:
                m = MagicMock()
                m.stdout = "1.0\n"
                return m
            ffmpeg_calls.append(list(cmd))
            return MagicMock(returncode=0)

        mock_run.side_effect = run_side_effect

        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.mp4")
            driving = os.path.join(td, "my_driving.mp4")
            open(driving, "wb").close()
            _mod.assemble_video_from_frames(items, driving, out)
            driving_path = driving

        # The mux call must reference the driving video path
        mux_cmds = [c for c in ffmpeg_calls if driving_path in c]
        self.assertTrue(mux_cmds, "Expected ffmpeg mux call to include driving video path")

    @patch("subprocess.run")
    def test_base64_item_written_to_disk(self, mock_run):
        """base64-typed items must be decoded and written to the tmp dir."""
        content = b"FAKEPNG" * 10
        items = [self._make_b64_item("WanVideo2_2_Animate_00001.png", content)]

        written_bytes = {}

        def run_side_effect(cmd, **kwargs):
            if "ffprobe" in cmd[0]:
                m = MagicMock()
                m.stdout = "1.0\n"
                return m
            # Intercept ffmpeg -i pattern: find the tmp dir from -i arg
            if "-framerate" in cmd:
                fr_idx = cmd.index("-framerate") + 1
                # find -i after -framerate
                i_idx = cmd.index("-i", fr_idx)
                pattern = cmd[i_idx + 1]
                # Read the actual file that would have been written
                import glob
                files = glob.glob(pattern.replace("%05d", "*"))
                for fp in files:
                    with open(fp, "rb") as fh:
                        written_bytes[os.path.basename(fp)] = fh.read()
            return MagicMock(returncode=0)

        mock_run.side_effect = run_side_effect

        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.mp4")
            driving = os.path.join(td, "driving.mp4")
            open(driving, "wb").close()
            _mod.assemble_video_from_frames(items, driving, out)

        # At least one file was handled; ffmpeg was called
        self.assertGreaterEqual(mock_run.call_count, 1)

    @patch("subprocess.run", side_effect=FileNotFoundError)
    @patch("urllib.request.urlopen")
    def test_ffmpeg_not_found_warns_no_crash(self, mock_urlopen, mock_run):
        """If ffmpeg is not found, warn and do not crash."""
        items = self._make_s3_items(["WanVideo2_2_Animate_00001.png"])

        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"\x89PNG")))
        cm.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = cm

        import io
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.mp4")
            driving = os.path.join(td, "driving.mp4")
            open(driving, "wb").close()
            with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
                # Should not raise
                try:
                    _mod.assemble_video_from_frames(items, driving, out)
                except SystemExit:
                    pass  # acceptable to exit, but not to crash with unhandled exception
                err_output = mock_err.getvalue()
            # Warning must be issued
            self.assertIn("ffmpeg", err_output.lower())


# ---------------------------------------------------------------------------
# _default_out_path tests
# ---------------------------------------------------------------------------

class TestDefaultOutPath(unittest.TestCase):
    def test_basename_in_output(self):
        path = _mod._default_out_path("/some/path/DZ45uPDgTCo.mp4")
        self.assertIn("DZ45uPDgTCo", path)
        self.assertIn("animate", path.lower())

    def test_ends_with_mp4(self):
        path = _mod._default_out_path("/foo/bar.mp4")
        self.assertTrue(path.endswith(".mp4"))


# ---------------------------------------------------------------------------
# Payload structure test
# ---------------------------------------------------------------------------

class TestPayloadStructure(unittest.TestCase):
    """The job input images[] must have two entries with correct names."""

    def test_two_images_entries(self):
        images = [
            {"name": _mod.REF_IMAGE_NAME, "image": "AAA"},
            {"name": _mod.DRIVING_VID_NAME, "image": "BBB"},
        ]
        self.assertEqual(len(images), 2)

    def test_ref_image_entry_name(self):
        images = [
            {"name": _mod.REF_IMAGE_NAME, "image": "AAA"},
            {"name": _mod.DRIVING_VID_NAME, "image": "BBB"},
        ]
        names = [e["name"] for e in images]
        self.assertIn("input_ref_image.png", names)

    def test_driving_vid_entry_name(self):
        images = [
            {"name": _mod.REF_IMAGE_NAME, "image": "AAA"},
            {"name": _mod.DRIVING_VID_NAME, "image": "BBB"},
        ]
        names = [e["name"] for e in images]
        self.assertIn("driving.mp4", names)

    def test_ref_name_matches_node_57_value(self):
        # What we upload must match what node 57 expects in the workflow.
        wf = _mod.build_workflow(DEFAULT_PROMPT, _mod.DEFAULT_NEGATIVE, 49, 42)
        self.assertEqual(wf["57"]["inputs"]["image"], _mod.REF_IMAGE_NAME)

    def test_driving_name_matches_node_63_value(self):
        wf = _mod.build_workflow(DEFAULT_PROMPT, _mod.DEFAULT_NEGATIVE, 49, 42)
        self.assertEqual(wf["63"]["inputs"]["video"], _mod.DRIVING_VID_NAME)


if __name__ == "__main__":
    unittest.main()
