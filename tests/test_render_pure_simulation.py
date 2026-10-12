import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "render_pure_simulation.py"
SPEC = importlib.util.spec_from_file_location("render_pure_simulation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PureSimulationSweepTest(unittest.TestCase):
    def test_review_video_defaults_to_60_fps(self):
        args = MODULE.build_parser().parse_args(["--output-dir", "/tmp/pure-sim-default-fps-test"])
        self.assertEqual(args.fps, 60)

    def test_full_sweep_shards_cover_each_speed_once(self):
        with tempfile.TemporaryDirectory() as directory:
            shards = MODULE.build_shards(
                Path(directory),
                [0.03, 0.06, 0.09],
                items_per_speed=600,
                shard_size=150,
            )
        self.assertEqual(len(shards), 12)
        for speed in ("speed_003", "speed_006", "speed_009"):
            indices = [
                index
                for shard in shards
                if shard.speed_label == speed
                for index in shard.expected_indices
            ]
            self.assertEqual(indices, list(range(600)))

    def test_speed_label_matches_formal_sweep_names(self):
        self.assertEqual(MODULE.speed_label(0.03), "speed_003")
        self.assertEqual(MODULE.speed_label(0.09), "speed_009")

    def test_qa_reasons_report_collision_and_visibility(self):
        record = {
            "status": "ok",
            "error": None,
            "target_names": ["bowl"],
            "preview_physics": {
                "motion_diagnostics": {
                    "bowl": {
                        "max_platform_path_error_m": 0.001,
                        "max_object_to_platform_error_m": 0.002,
                        "max_tilt_deg": 1.0,
                        "base_bottom_z": 0.9,
                        "min_bottom_z": 0.899,
                    }
                },
                "safety_diagnostics": {
                    "bowl": {
                        "collision_count": 1,
                        "birdview_visible_ratio": 1.0,
                        "agentview_visible_ratio": 0.7,
                    }
                },
            },
            "video": __file__,
        }
        reasons = MODULE.qa_failure_reasons(record)
        self.assertIn("bowl:collisions=1", reasons)
        self.assertIn("bowl:agentview_visible=0.7", reasons)


if __name__ == "__main__":
    unittest.main()
