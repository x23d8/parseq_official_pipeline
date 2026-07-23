"""Runtime adapters for the learned selectors stored in ``D:/NEO/rl_pipeline``.

The demo deliberately reuses its already loaded PARSeq model.  The external
folder supplies the frozen action registries, feature builders and policy
checkpoints, so no model or source file has to be copied into this repository.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image


# Keep the host preprocessing package stable even after the optional PixelRL
# loader prepends its historical source snapshot to ``sys.path``.
LOCAL_ROOT = Path(__file__).resolve().parents[1]
if str(LOCAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ROOT))
import preprocessing_best_config.benchmark_multiscale_selector_phase2  # noqa: E402,F401
import preprocessing_best_config.benchmark_multiscale_tta  # noqa: E402,F401


CALIBRATED_CANDIDATE = "calibrated_candidate_selector"
CONTEXTUAL_BANDIT = "contextual_bandit"
TWO_STAGE_PPO = "two_stage_ppo"
AUTO_CANDIDATE_PPO = "auto_candidate_ppo"
SELECTOR_METHODS = frozenset(
    {CALIBRATED_CANDIDATE, CONTEXTUAL_BANDIT, TWO_STAGE_PPO, AUTO_CANDIDATE_PPO}
)


class LearnedSelectorRuntime:
    """Lazy policy loader that runs against the demo's active PARSeq model."""

    def __init__(self, rl_root: Path, device: torch.device) -> None:
        self.root = Path(rl_root).resolve()
        self.device = device
        self._loaded: dict[str, Any] = {}
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))

        self.paths = {
            CALIBRATED_CANDIDATE: self.root
            / "reinforcement_learning/phase_2_calibrated_selector/results/phase2_selector.joblib",
            CONTEXTUAL_BANDIT: self.root
            / "outputs/rl_restoration/router_seed_123/best_reward_router.pt",
            TWO_STAGE_PPO: self.root
            / "outputs/rl_restoration/ppo_prior_seed_123/best_ppo_restoration_policy.pt",
            AUTO_CANDIDATE_PPO: self.root
            / "reinforcement_learning/auto/results/experiment/run_seed_2028/best_candidate_oof_ppo.pt",
        }

    def availability(self) -> dict[str, dict[str, Any]]:
        source_ready = (self.root / "reinforcement_learning").is_dir() and (
            self.root / "rl_restoration"
        ).is_dir()
        result: dict[str, dict[str, Any]] = {}
        for name, checkpoint in self.paths.items():
            available = source_ready and checkpoint.is_file()
            missing = []
            if not source_ready:
                missing.append(f"RL source under {self.root}")
            if not checkpoint.is_file():
                missing.append(f"checkpoint {checkpoint}")
            result[name] = {
                "available": available,
                "checkpoint": str(checkpoint),
                "unavailable_reason": None if available else "Missing " + " and ".join(missing),
            }
        return result

    @staticmethod
    def _normalized_confidence(value: float, prediction: str) -> float:
        return math.exp(math.log(max(float(value), 1e-12)) / max(len(prediction) + 1, 1))

    @torch.inference_mode()
    def _ocr(self, engine, images: list[Image.Image], deep: bool = False) -> dict[str, Any]:
        tensors = torch.stack([engine._tensor_from_processed(image, "") for image in images]).to(
            self.device
        )
        with engine._lock:
            logits = engine._model(tensors, max_length=engine._model_cfg.max_label_length)
            probabilities = logits.softmax(-1)
            predictions, token_probabilities = engine._model.tokenizer.decode(probabilities)
        from train_no_refinement.parseq_official_anpr_pipeline import normalize_plate_text

        predictions = [normalize_plate_text(value) for value in predictions]
        raw_confidence = np.asarray(
            [float(values.prod().detach().cpu().item()) for values in token_probabilities],
            dtype=np.float32,
        )
        normalized = np.asarray(
            [self._normalized_confidence(value, prediction) for value, prediction in zip(raw_confidence, predictions)],
            dtype=np.float32,
        )
        result: dict[str, Any] = {
            "predictions": predictions,
            "confidence": raw_confidence,
            "normalized_confidence": normalized,
        }
        if deep:
            from rl_restoration.features import parseq_state_features

            result["deep"] = parseq_state_features(engine._model, tensors, predictions, logits).cpu().numpy()
        return result

    @staticmethod
    def _result(
        method: str,
        processed: Image.Image,
        prediction: str,
        confidence: float,
        runtime_steps: list[str],
        started: float,
        trace: dict[str, Any],
    ) -> dict[str, Any]:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "method": method,
            "pipeline": [method],
            "runtime_pipeline": runtime_steps,
            "runtime_method": " -> ".join(runtime_steps),
            "step_timings": [
                {
                    "position": index,
                    "method": step,
                    "runtime_method": step,
                    "milliseconds": elapsed_ms if index == len(runtime_steps) else 0.0,
                }
                for index, step in enumerate(runtime_steps, start=1)
            ],
            "prediction": prediction,
            "confidence": float(confidence),
            "preprocessing_ms": 0.0,
            "inference_ms": elapsed_ms,
            "selector_trace": trace,
            "processed_image_pil": processed,
        }

    def _load_bandit(self):
        if CONTEXTUAL_BANDIT in self._loaded:
            return self._loaded[CONTEXTUAL_BANDIT]
        from rl_restoration.actions import DEFAULT_ACTIONS
        from rl_restoration.policy import RewardRouter

        checkpoint = torch.load(self.paths[CONTEXTUAL_BANDIT], map_location="cpu", weights_only=False)
        action_names = [action.name for action in DEFAULT_ACTIONS]
        if checkpoint["action_names"] != action_names:
            raise ValueError("Contextual-bandit checkpoint action registry is incompatible.")
        policy = RewardRouter(
            checkpoint["input_dim"], len(DEFAULT_ACTIONS), checkpoint["hidden_dim"], checkpoint["dropout"]
        ).to(self.device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        policy.eval()
        payload = (checkpoint, policy, DEFAULT_ACTIONS)
        self._loaded[CONTEXTUAL_BANDIT] = payload
        return payload

    def _run_bandit(self, engine, image: Image.Image) -> dict[str, Any]:
        started = time.perf_counter()
        checkpoint, policy, actions = self._load_bandit()
        from rl_restoration.features import image_quality_features

        baseline_image = actions[0].apply(image)
        baseline = self._ocr(engine, [baseline_image], deep=True)
        raw = np.concatenate((baseline["deep"], image_quality_features(image)[None, :]), axis=1)
        if raw.shape[1] != int(checkpoint["input_dim"]):
            raise RuntimeError(
                f"Contextual-bandit feature mismatch: runtime={raw.shape[1]}, checkpoint={checkpoint['input_dim']}"
            )
        standardized = ((raw - checkpoint["feature_mean"]) / checkpoint["feature_std"]).astype(np.float32)
        with torch.inference_mode():
            rewards = policy(torch.from_numpy(standardized).to(self.device)).cpu().numpy()[0]
        best = int(rewards.argmax())
        gain = float(rewards[best] - rewards[0])
        selected = best if gain >= float(checkpoint["selection_margin"]) else 0
        processed = baseline_image if selected == 0 else actions[selected].apply(image)
        output = baseline if selected == 0 else self._ocr(engine, [processed])
        action_name = actions[selected].name
        return self._result(
            CONTEXTUAL_BANDIT,
            processed,
            output["predictions"][0],
            output["confidence"][0],
            [CONTEXTUAL_BANDIT, action_name],
            started,
            {
                "algorithm": "offline_contextual_bandit",
                "selected_action": action_name,
                "predicted_reward_gain": gain,
                "selection_margin": float(checkpoint["selection_margin"]),
            },
        )

    def _load_ppo(self):
        if TWO_STAGE_PPO in self._loaded:
            return self._loaded[TWO_STAGE_PPO]
        from rl_restoration.actions import DEFAULT_ACTIONS
        from rl_restoration.policy import RewardRouter
        from rl_restoration.ppo_policy import RestorationActorCritic

        checkpoint = torch.load(self.paths[TWO_STAGE_PPO], map_location="cpu", weights_only=False)
        action_names = [action.name for action in DEFAULT_ACTIONS]
        if checkpoint["action_names"] != action_names:
            raise ValueError("Two-stage PPO checkpoint action registry is incompatible.")
        policy = RestorationActorCritic(
            checkpoint["input_dim"], len(DEFAULT_ACTIONS), checkpoint["hidden_dim"], checkpoint["dropout"],
            checkpoint["prior_offset"], checkpoint["prior_scale"],
        ).to(self.device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        policy.eval()
        teacher_path = Path(str(checkpoint.get("teacher_router", "")))
        if not teacher_path.is_file():
            teacher_path = self.paths[CONTEXTUAL_BANDIT]
        teacher_checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=False)
        teacher = RewardRouter(
            teacher_checkpoint["input_dim"], len(DEFAULT_ACTIONS), teacher_checkpoint["hidden_dim"],
            teacher_checkpoint["dropout"],
        ).to(self.device)
        teacher.load_state_dict(teacher_checkpoint["model_state_dict"])
        teacher.eval()
        payload = (checkpoint, policy, teacher_checkpoint, teacher, DEFAULT_ACTIONS)
        self._loaded[TWO_STAGE_PPO] = payload
        return payload

    def _run_ppo(self, engine, image: Image.Image) -> dict[str, Any]:
        started = time.perf_counter()
        checkpoint, policy, teacher_checkpoint, teacher, actions = self._load_ppo()
        from rl_restoration.features import image_quality_features
        from rl_restoration.sequential_env import MAX_PLATE_LENGTH, encode_predictions

        baseline_image = actions[0].apply(image)
        baseline = self._ocr(engine, [baseline_image], deep=True)
        raw = np.concatenate((baseline["deep"], image_quality_features(image)[None, :]), axis=1)
        standardized = (raw - checkpoint["feature_mean"]) / checkpoint["feature_std"]
        teacher_x = (raw - teacher_checkpoint["feature_mean"]) / teacher_checkpoint["feature_std"]
        with torch.inference_mode():
            teacher_rewards = teacher(torch.from_numpy(teacher_x.astype(np.float32)).to(self.device))
        base = np.concatenate((standardized, teacher_rewards.cpu().numpy()), axis=1)[0]
        baseline_prediction = baseline["predictions"][0]

        def state(view: dict[str, Any], action_index: int, step: int) -> torch.Tensor:
            encoded = encode_predictions(np.asarray([[view["predictions"][0]]], dtype=str))[0, 0]
            action_one_hot = np.zeros(len(actions), dtype=np.float32)
            action_one_hot[action_index] = 1.0
            observation = np.concatenate(
                (
                    base,
                    np.asarray(
                        [
                            view["normalized_confidence"][0],
                            len(view["predictions"][0]) / MAX_PLATE_LENGTH,
                            float(view["predictions"][0] != baseline_prediction),
                        ],
                        dtype=np.float32,
                    ),
                    encoded,
                    action_one_hot,
                    [float(step)],
                )
            ).astype(np.float32)
            return torch.from_numpy(observation).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits0, _ = policy(state(baseline, 0, 0))
        best0 = int(logits0.argmax(dim=1).item())
        first_gain = float((logits0[0, best0] - logits0[0, 0]).item())
        first = best0 if best0 != 0 and first_gain >= float(checkpoint["first_margin"]) else 0
        if first == 0:
            final_index, revised, final_image, final = 0, False, baseline_image, baseline
        else:
            intermediate_image = actions[first].apply(image)
            intermediate = self._ocr(engine, [intermediate_image])
            with torch.inference_mode():
                logits1, _ = policy(state(intermediate, first, 1))
            best1 = int(logits1.argmax(dim=1).item())
            revise_gain = float((logits1[0, best1] - logits1[0, first]).item())
            final_index = best1 if revise_gain >= float(checkpoint["revise_margin"]) else first
            revised = final_index != first
            if final_index == first:
                final_image, final = intermediate_image, intermediate
            elif final_index == 0:
                final_image, final = baseline_image, baseline
            else:
                final_image = actions[final_index].apply(image)
                final = self._ocr(engine, [final_image])
        return self._result(
            TWO_STAGE_PPO,
            final_image,
            final["predictions"][0],
            final["confidence"][0],
            [TWO_STAGE_PPO, actions[first].name, actions[final_index].name],
            started,
            {
                "algorithm": "actor_critic_ppo_two_stage",
                "first_action": actions[first].name,
                "final_action": actions[final_index].name,
                "revised": bool(revised),
                "first_gain": first_gain,
            },
        )

    def _load_auto(self):
        if AUTO_CANDIDATE_PPO in self._loaded:
            return self._loaded[AUTO_CANDIDATE_PPO]
        # The auto registry uses the historical config id ``richardson_lucy``;
        # the host catalog renamed the identical pure block to
        # ``richardson_lucy_deblur``.  Register an in-memory compatibility
        # alias without copying or modifying either project's source files.
        from dataclasses import replace
        from preprocessing_best_config import preprocessing as host_preprocessing

        try:
            host_preprocessing.get_preprocessing_config("richardson_lucy")
        except KeyError:
            source = host_preprocessing.get_preprocessing_config("richardson_lucy_deblur")
            host_preprocessing.SWEEP_CONFIGS.append(replace(source, name="richardson_lucy"))
        from reinforcement_learning.auto.action_space import AUTO_VIEWS
        from reinforcement_learning.phase_6_candidate_oof_ppo.model import CandidateSetActorCritic, RewardTeacher

        checkpoint = torch.load(self.paths[AUTO_CANDIDATE_PPO], map_location="cpu", weights_only=False)
        if checkpoint.get("test_used", True):
            raise ValueError("Auto runtime requires a test-free policy checkpoint.")
        if checkpoint["action_names"] != [view.name for view in AUTO_VIEWS]:
            raise ValueError("Auto PPO checkpoint action registry is incompatible.")
        teacher_cfg = checkpoint["teacher_config"]
        teacher = RewardTeacher(
            teacher_cfg["input_dim"], len(AUTO_VIEWS), teacher_cfg["hidden_dim"], teacher_cfg["dropout"]
        ).to(self.device)
        teacher.load_state_dict(checkpoint["teacher_state_dict"])
        teacher.eval()
        model_cfg = checkpoint["model_config"]
        policy = CandidateSetActorCritic(
            model_cfg["candidate_dim"], model_cfg["action_count"], model_cfg["hidden_dim"],
            model_cfg["heads"], model_cfg["layers"], model_cfg["dropout"], model_cfg["prior_scale"],
        ).to(self.device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        policy.eval()
        payload = (checkpoint, policy, teacher, AUTO_VIEWS)
        self._loaded[AUTO_CANDIDATE_PPO] = payload
        return payload

    def _run_auto(self, engine, image: Image.Image) -> dict[str, Any]:
        started = time.perf_counter()
        checkpoint, policy, teacher, views = self._load_auto()
        from reinforcement_learning.auto.action_space import view_metadata
        from reinforcement_learning.phase_6_candidate_oof_ppo.data import candidate_ocr_features
        from reinforcement_learning.phase_6_candidate_oof_ppo.train import policy_selection, teacher_predict

        processed = [view.apply(image) for view in views]
        output = self._ocr(engine, processed, deep=True)
        metadata = np.stack([np.asarray(view_metadata(view), dtype=np.float32) for view in views])
        raw = np.concatenate((output["deep"], metadata), axis=1)[None, :, :]
        predictions = np.asarray(output["predictions"], dtype=str)[None, :]
        normalized = output["normalized_confidence"][None, :]
        if checkpoint.get("candidate_ocr_strings", False):
            raw = np.concatenate(
                (raw, candidate_ocr_features({"predictions": predictions, "normalized_confidence": normalized})),
                axis=2,
            )
        candidates = ((raw - checkpoint["candidate_mean"]) / checkpoint["candidate_std"]).astype(np.float32)
        teacher_x = ((raw[:, 0] - checkpoint["teacher_mean"]) / checkpoint["teacher_std"]).astype(np.float32)
        prior = teacher_predict(teacher, teacher_x, self.device)
        first, selected, revised = policy_selection(
            policy,
            torch.from_numpy(candidates).to(self.device),
            torch.from_numpy(prior).to(self.device),
            checkpoint["first_margin"], checkpoint["revise_margin"], self.device,
            checkpoint["teacher_margin"], checkpoint.get("disagreement_margin"),
            checkpoint.get("final_teacher_gain_margin"),
        )
        first_index, selected_index = int(first[0]), int(selected[0])
        selected_view = views[selected_index]
        return self._result(
            AUTO_CANDIDATE_PPO,
            processed[selected_index],
            output["predictions"][selected_index],
            output["confidence"][selected_index],
            [AUTO_CANDIDATE_PPO, selected_view.name],
            started,
            {
                "algorithm": "candidate_set_ppo_with_oof_teacher_residual",
                "candidate_count": len(views),
                "first_action": views[first_index].name,
                "final_action": selected_view.name,
                "components": list(selected_view.components) or ["train_baseline"],
                "revised": bool(revised[0]),
            },
        )

    @staticmethod
    def _image_features(image: Image.Image) -> pd.DataFrame:
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        aspect = width / max(height, 1)
        scale = "tiny" if width < 64 or height < 24 else "small" if width < 100 or height < 40 else "regular"
        layout = "two_line" if aspect < 1.9 else "single_line"
        row = {
            "image_path": "runtime",
            "image_width": float(width), "image_height": float(height), "aspect_ratio": float(aspect),
            "log_image_area": float(math.log1p(width * height)), "gray_mean": float(gray.mean() / 255.0),
            "gray_std": float(gray.std() / 255.0), "saturation_mean": float(hsv[..., 1].mean() / 255.0),
            "laplacian_log_variance": float(math.log1p(cv2.Laplacian(gray, cv2.CV_64F).var())),
            "is_tiny": float(scale == "tiny"), "is_small": float(scale != "regular"),
            "is_two_line": float(layout == "two_line"), "route_class": f"{scale}_{layout}",
        }
        return pd.DataFrame([row]).set_index("image_path", drop=False)

    def _load_calibrated(self):
        if CALIBRATED_CANDIDATE in self._loaded:
            return self._loaded[CALIBRATED_CANDIDATE]
        artifact = joblib.load(self.paths[CALIBRATED_CANDIDATE])
        self._loaded[CALIBRATED_CANDIDATE] = artifact
        return artifact

    def _run_calibrated(self, engine, image: Image.Image) -> dict[str, Any]:
        started = time.perf_counter()
        artifact = self._load_calibrated()
        from preprocessing import get_preprocessing_config, preprocess_plate_image
        from preprocessing_best_config.benchmark_multiscale_selector_phase2 import (
            build_candidate_features,
            deserialize_reference_stats,
            score_candidates,
            select_predictions,
        )
        from preprocessing_best_config.benchmark_multiscale_tta import (
            apply_center_zoom,
            build_specs,
            unwrap_plate_lines,
            upscale_small_image,
        )

        specs = build_specs()
        processed: list[Image.Image] = []
        for spec in specs:
            view = apply_center_zoom(image.convert("RGB"), spec.zoom)
            view = upscale_small_image(view, spec.upscale)
            if spec.unwrap_two_line:
                view = unwrap_plate_lines(view)
            processed.append(
                preprocess_plate_image(view, get_preprocessing_config(spec.preprocessing)).convert("RGB")
            )
        output = self._ocr(engine, processed)
        rows = []
        for spec, prediction, confidence, normalized in zip(
            specs, output["predictions"], output["confidence"], output["normalized_confidence"]
        ):
            rows.append(
                {
                    "image_path": "runtime", "target": "", "prediction": prediction, "view": spec.name,
                    "confidence": float(confidence), "normalized_confidence": float(normalized), "exact": False,
                    "edit_distance": len(prediction), "target_length": 1, "zoom": spec.zoom,
                    "upscale": spec.upscale, "preprocessing": spec.preprocessing,
                    "unwrap_two_line": spec.unwrap_two_line,
                }
            )
        predictions = pd.DataFrame(rows)
        stats = deserialize_reference_stats(artifact["reference_stats"])
        candidates = build_candidate_features(predictions, self._image_features(image), stats)
        scored = score_candidates(artifact["model"], candidates, artifact["feature_columns"])
        selected = select_predictions(scored, float(artifact["switch_margin"])).iloc[0]
        supporting = str(selected["supporting_views"]).split(";")
        matches = [index for index, row in enumerate(rows) if row["view"] in supporting and row["prediction"] == selected["prediction"]]
        selected_index = max(matches, key=lambda index: rows[index]["confidence"]) if matches else 0
        return self._result(
            CALIBRATED_CANDIDATE,
            processed[selected_index],
            str(selected["prediction"]),
            rows[selected_index]["confidence"],
            [CALIBRATED_CANDIDATE, rows[selected_index]["view"]],
            started,
            {
                "algorithm": "calibrated_pairwise_candidate_selector",
                "candidate_views": len(specs),
                "unique_predictions": int(len(candidates)),
                "selected_view": rows[selected_index]["view"],
                "votes": int(selected["votes"]),
                "switched_from_phase1": bool(selected["switched_from_phase1"]),
                "score_gain_over_phase1": float(selected["score_gain_over_phase1"]),
            },
        )

    def run(self, engine, image: Image.Image, method: str) -> dict[str, Any]:
        if method == CALIBRATED_CANDIDATE:
            return self._run_calibrated(engine, image)
        if method == CONTEXTUAL_BANDIT:
            return self._run_bandit(engine, image)
        if method == TWO_STAGE_PPO:
            return self._run_ppo(engine, image)
        if method == AUTO_CANDIDATE_PPO:
            return self._run_auto(engine, image)
        raise KeyError(f"Unknown learned selector: {method}")
