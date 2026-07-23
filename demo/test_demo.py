from __future__ import annotations

import json
import unittest

from PIL import Image

from demo.app import catalog, catalog_by_name, pipeline_benchmark, resolve_pipeline
from demo.inference_engine import InferenceEngine, MAX_PIPELINE_STEPS, RL_METHOD_NAME
from demo.rl_runtime import (
    AUTO_CANDIDATE_PPO,
    CALIBRATED_CANDIDATE,
    CONTEXTUAL_BANDIT,
    SELECTOR_METHODS,
    TWO_STAGE_PPO,
)
from demo.method_catalog import load_method_catalog
from demo.recovery_runtime import RECOVERY_METHOD_NAME, select_recovery_candidate


class DemoPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = InferenceEngine()

    def test_rl_agent_is_exposed_as_a_described_composable_method(self):
        methods = load_method_catalog(self.engine.available_configs, self.engine.rl_method_info())
        rl = next(item for item in methods if item["name"] == RL_METHOD_NAME)
        self.assertEqual(rl["kind"], "rl_agent")
        self.assertTrue(rl["experimental"])
        self.assertEqual(rl["display_name"], "PixelRL Reinforcement Agent")
        self.assertIn("Reinforcement learning", rl["topic"])
        self.assertEqual(rl["available"], self.engine.rl_checkpoint_path.is_file())

    def test_richardson_lucy_is_not_labeled_as_reinforcement_learning(self):
        methods = load_method_catalog(self.engine.available_configs, self.engine.rl_method_info())
        classical = next(item for item in methods if item["name"] == "richardson_lucy_deblur")
        self.assertIn("Richardson-Lucy", classical["display_name"])
        self.assertNotIn("Reinforcement", classical["display_name"])

    def test_ui_catalog_contains_image_processing_and_learned_methods(self):
        self.assertEqual(len(catalog), 26)
        names = {item["name"] for item in catalog}
        self.assertIn("clahe_gray", names)
        self.assertIn("wavelet_haar", names)
        self.assertIn("freq_highboost", names)
        self.assertIn("morph_tophat", names)
        self.assertIn("component_mask_gray", names)
        self.assertIn(RECOVERY_METHOD_NAME, names)
        self.assertNotIn("clahe_wavelet_haar", names)
        self.assertNotIn("clahe_rl_deblur_bilateral", names)
        self.assertTrue(
            {RL_METHOD_NAME, CALIBRATED_CANDIDATE, CONTEXTUAL_BANDIT, TWO_STAGE_PPO, AUTO_CANDIDATE_PPO}
            <= names
        )
        self.assertEqual(sum(item["filter_group"] == "imp" for item in catalog), 20)
        self.assertEqual(sum(item["filter_group"] == "rl" for item in catalog), 6)

    def test_catalog_ranks_measured_methods_before_unmeasured_blocks(self):
        self.assertEqual([item["rank"] for item in catalog], list(range(1, 27)))
        imp = [item for item in catalog if item["filter_group"] == "imp"]
        measured = [item for item in imp if item["benchmark_available"]]
        self.assertEqual(imp[: len(measured)], measured)
        self.assertEqual(catalog[0]["name"], "homomorphic_filter")
        self.assertEqual(
            [item["exact_acc"] for item in measured],
            sorted((item["exact_acc"] for item in measured), reverse=True),
        )

    def test_learned_selectors_are_available_and_must_run_alone(self):
        for name in SELECTOR_METHODS:
            with self.subTest(name=name):
                item = catalog_by_name[name]
                self.assertTrue(item["available"])
                self.assertTrue(item["exclusive"])
                self.assertFalse(item["comparison_eligible"])
                self.assertEqual(self.engine.normalize_pipeline([name]), [name])
                with self.assertRaises(ValueError):
                    self.engine.normalize_pipeline(["autocontrast", name])

    def test_calibrated_candidate_is_not_mislabeled_as_policy_gradient_rl(self):
        item = catalog_by_name[CALIBRATED_CANDIDATE]
        self.assertEqual(item["kind"], "learned_selector")
        self.assertFalse(item["experimental"])
        self.assertIn("calibrated", item["topic"].lower())

    def test_recovery_ensemble_is_an_exclusive_rl_catalog_entry(self):
        item = catalog_by_name[RECOVERY_METHOD_NAME]
        self.assertEqual(item["kind"], "recovery_selector")
        self.assertEqual(item["catalog_badge"], "RECOVERY")
        self.assertEqual(item["filter_group"], "rl")
        self.assertEqual(item["experimental_label"], "RECOVERY")
        self.assertTrue(item["exclusive"])
        self.assertFalse(item["comparison_eligible"])
        self.assertEqual(self.engine.normalize_pipeline([RECOVERY_METHOD_NAME]), [RECOVERY_METHOD_NAME])
        with self.assertRaises(ValueError):
            self.engine.normalize_pipeline(["autocontrast", RECOVERY_METHOD_NAME])

    def test_recovery_selector_rejects_confident_truncated_text(self):
        selected = select_recovery_candidate(
            [
                {
                    "prediction": "51G7",
                    "confidence": 0.996,
                    "normalized_confidence": 0.999,
                },
                {
                    "prediction": "51G74356",
                    "confidence": 0.979,
                    "normalized_confidence": 0.997,
                },
            ]
        )
        self.assertEqual(selected["prediction"], "51G74356")

    def test_new_standalone_blocks_execute(self):
        image = Image.new("RGB", (96, 48), "gray")
        for name in (
            "wavelet_haar",
            "wiener_deconv",
            "richardson_lucy_deblur",
            "freq_highboost",
            "morph_tophat",
            "otsu_binary",
            "component_mask_gray",
        ):
            with self.subTest(name=name):
                processed, tensor, runtime, timings, _elapsed = self.engine._prepare_pipeline(image, [name])
                self.assertEqual(runtime, [name])
                self.assertEqual(timings[0]["method"], name)
                self.assertEqual(processed.mode, "RGB")
                self.assertEqual(list(tensor.shape), [3, 32, 128])

    def test_pipeline_order_is_preserved_and_duplicate_steps_are_rejected(self):
        pipeline = ["autocontrast", "gamma_0_9", "homomorphic_filter"]
        self.assertEqual(self.engine.normalize_pipeline(pipeline), pipeline)
        with self.assertRaises(ValueError):
            self.engine.normalize_pipeline(["autocontrast", "autocontrast"])

    def test_pipeline_has_a_bounded_number_of_steps(self):
        names = list(self.engine.available_configs)[: MAX_PIPELINE_STEPS + 1]
        with self.assertRaises(ValueError):
            self.engine.normalize_pipeline(names)

    def test_classical_composition_executes_left_to_right(self):
        pipeline = ["autocontrast", "gamma_0_9"]
        processed, tensor, runtime, timings, elapsed = self.engine._prepare_pipeline(
            Image.new("RGB", (96, 48), "gray"), pipeline
        )
        self.assertEqual(runtime, pipeline)
        self.assertEqual([item["method"] for item in timings], pipeline)
        self.assertEqual(list(tensor.shape), [3, 32, 128])
        self.assertEqual(processed.mode, "RGB")
        self.assertGreaterEqual(elapsed, 0.0)

    def test_tight_plate_detection_is_not_recropped(self):
        self.assertTrue(self.engine._is_tight_plate_input((237, 173), [0, 0, 193, 168]))
        self.assertFalse(self.engine._is_tight_plate_input((1280, 720), [400, 300, 800, 450]))

    def test_api_pipeline_parser_and_custom_benchmark_contract(self):
        names = resolve_pipeline(None, json.dumps(["autocontrast", "gamma_0_9"]))
        self.assertEqual(names, ["autocontrast", "gamma_0_9"])
        metadata = pipeline_benchmark(names)
        self.assertFalse(metadata["benchmark_available"])
        self.assertTrue(metadata["is_composition"])
        self.assertIsNone(metadata["exact_acc"])
        measured_name = next(name for name, item in catalog_by_name.items() if item["benchmark_available"])
        single = pipeline_benchmark([measured_name])
        self.assertTrue(single["benchmark_available"])


if __name__ == "__main__":
    unittest.main()
