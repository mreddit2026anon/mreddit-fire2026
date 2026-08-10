"""Model-free contract tests for the Stage 5 Tulu extended-epoch experiment."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_transformer import (
    check_inputs,
    configured_early_stopping_patience,
    load_config,
    output_paths,
    trainer_callbacks,
)
from src.transformer_data import fixed_label_mapping, load_training_task, make_stratified_split


BASE_CONFIG = ROOT / "configs/transformer_baseline.json"
EXTENDED_CONFIG = ROOT / "configs/tulu_extended_epochs.json"
TULU_TASKS = ("tulu_regret", "tulu_domain")


def synthetic_task(task: str, rows_per_class: int = 20) -> pd.DataFrame:
    label2id, _ = fixed_label_mapping(task)
    rows = []
    for label, label_id in label2id.items():
        for index in range(rows_per_class):
            rows.append(
                {
                    "id": f"{task}-{label_id}-{index}",
                    "text": f"[TITLE] title {label_id} {index}\n[TEXT] body {label_id} {index}",
                    "label": label,
                    "label_id": label_id,
                }
            )
    return pd.DataFrame(rows)


class TuluExtendedEpochTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline = load_config(BASE_CONFIG)
        cls.extended = json.loads(EXTENDED_CONFIG.read_text(encoding="utf-8"))

    def test_extended_retry_configuration_is_suspended(self) -> None:
        self.assertFalse(self.extended["execution_enabled"])
        with self.assertRaisesRegex(ValueError, "configuration is suspended"):
            load_config(EXTENDED_CONFIG)

    def test_only_two_tulu_tasks_are_allowed(self) -> None:
        self.assertEqual(self.extended["allowed_tasks"], list(TULU_TASKS))
        self.assertEqual(set(self.extended["tasks"]), set(TULU_TASKS))
        self.assertEqual(self.extended["first_task"], "tulu_regret")
        for task in TULU_TASKS:
            self.assertEqual(self.extended["tasks"][task]["language"], "Tulu")

    def test_training_changes_are_epochs_and_early_stopping_only(self) -> None:
        self.assertEqual(self.baseline["training"]["num_train_epochs"], 5)
        self.assertEqual(self.extended["training"]["num_train_epochs"], 8)
        self.assertEqual(configured_early_stopping_patience(self.baseline), 2)
        self.assertIsNone(configured_early_stopping_patience(self.extended))
        callback_runtime = {
            "EarlyStoppingCallback": lambda **kwargs: kwargs,
        }
        self.assertEqual(trainer_callbacks(callback_runtime, self.extended, smoke=False), [])
        self.assertEqual(
            trainer_callbacks(callback_runtime, self.baseline, smoke=False),
            [{"early_stopping_patience": 2}],
        )
        baseline_training = dict(self.baseline["training"])
        extended_training = dict(self.extended["training"])
        for key in ("num_train_epochs", "early_stopping_patience"):
            baseline_training.pop(key)
            extended_training.pop(key)
        self.assertEqual(extended_training, baseline_training)
        self.assertEqual(self.extended["model_name"], self.baseline["model_name"])
        self.assertEqual(self.extended["seed"], self.baseline["seed"])
        self.assertEqual(self.extended["validation_fraction"], self.baseline["validation_fraction"])
        self.assertEqual(self.extended["text"], self.baseline["text"])
        self.assertEqual(self.extended["class_weights"], self.baseline["class_weights"])
        self.assertEqual(self.extended["smoke"], self.baseline["smoke"])
        self.assertEqual(self.extended["environment"], self.baseline["environment"])
        for runtime_default in ("optim", "lr_scheduler_type", "lr_scheduler_kwargs"):
            self.assertEqual(
                self.extended["training"].get(runtime_default),
                self.baseline["training"].get(runtime_default),
            )

    def test_training_relevant_config_matches_baseline(self) -> None:
        shared_keys = (
            "model_name",
            "seed",
            "validation_fraction",
            "labels",
            "text",
            "class_weights",
            "smoke",
            "environment",
        )
        for key in shared_keys:
            self.assertEqual(self.extended[key], self.baseline[key], key)
        baseline_training = dict(self.baseline["training"])
        extended_training = dict(self.extended["training"])
        for key in ("num_train_epochs", "early_stopping_patience"):
            baseline_training.pop(key)
            extended_training.pop(key)
        self.assertEqual(extended_training, baseline_training)
        for task in TULU_TASKS:
            for key in ("language", "target", "validation"):
                self.assertEqual(self.extended["tasks"][task][key], self.baseline["tasks"][task][key])

    def test_split_is_identical_to_first_round(self) -> None:
        for task in TULU_TASKS:
            with self.subTest(task=task):
                frame = synthetic_task(task)
                baseline_split = make_stratified_split(
                    frame,
                    task,
                    float(self.baseline["validation_fraction"]),
                    int(self.baseline["seed"]),
                )
                extended_split = make_stratified_split(
                    frame,
                    task,
                    float(self.extended["validation_fraction"]),
                    int(self.extended["seed"]),
                )
                self.assertEqual(baseline_split.train["id"].tolist(), extended_split.train["id"].tolist())
                self.assertEqual(
                    baseline_split.validation["id"].tolist(),
                    extended_split.validation["id"].tolist(),
                )

    def test_output_roots_are_isolated_from_baseline(self) -> None:
        expected_roots = {
            "output": ROOT / "outputs/transformer_tulu_extended_repro_retry",
            "checkpoint": ROOT / "checkpoints/transformer_tulu_extended_repro_retry",
            "model": ROOT / "models/transformer_tulu_extended_repro_retry",
            "log": ROOT / "logs/transformer_tulu_extended_repro_retry",
        }
        failed_roots = {
            "output": ROOT / "outputs/transformer_tulu_extended",
            "checkpoint": ROOT / "checkpoints/transformer_tulu_extended",
            "model": ROOT / "models/transformer_tulu_extended",
            "log": ROOT / "logs/transformer_tulu_extended",
        }
        failed_no_early_stop_roots = {
            "output": ROOT / "outputs/transformer_tulu_extended_no_early_stop",
            "checkpoint": ROOT / "checkpoints/transformer_tulu_extended_no_early_stop",
            "model": ROOT / "models/transformer_tulu_extended_no_early_stop",
            "log": ROOT / "logs/transformer_tulu_extended_no_early_stop",
        }
        for task in TULU_TASKS:
            with self.subTest(task=task):
                baseline_paths = output_paths(self.baseline, task)
                extended_paths = output_paths(self.extended, task)
                smoke_paths = output_paths(self.extended, task, smoke=True)
                for key, root in expected_roots.items():
                    suffix = Path(task, "best") if key == "model" else Path(task)
                    self.assertEqual(extended_paths[key], root / suffix)
                    self.assertNotEqual(extended_paths[key], baseline_paths[key])
                    self.assertNotEqual(extended_paths[key], failed_roots[key] / suffix)
                    self.assertNotEqual(
                        extended_paths[key],
                        failed_no_early_stop_roots[key] / suffix,
                    )
                    self.assertIn("smoke", smoke_paths[key].parts)
                    self.assertNotEqual(smoke_paths[key], extended_paths[key])

    def test_baseline_english_and_kannada_contracts_are_unchanged(self) -> None:
        reloaded = load_config(BASE_CONFIG)
        self.assertEqual(
            reloaded["allowed_tasks"],
            ["english_domain", "english_regret", "tulu_regret", "tulu_domain"],
        )
        self.assertEqual(reloaded["first_task"], "english_domain")
        self.assertNotIn("english_domain", self.extended["allowed_tasks"])
        self.assertNotIn("english_regret", self.extended["allowed_tasks"])
        self.assertNotIn("kannada_regret", self.extended["allowed_tasks"])
        self.assertNotIn("kannada_domain", self.extended["allowed_tasks"])

    def test_tulu_training_loader_requests_train_only(self) -> None:
        raw = pd.DataFrame(
            {
                "id": [f"id-{index}" for index in range(10)],
                "title": [f"title {index}" for index in range(10)],
                "text": [f"body {index}" for index in range(10)],
                "Regret": ["Action", "Inaction", "No Regret", "Action", "Inaction"] * 2,
                "Domain": [
                    "Education",
                    "Health",
                    "Career and Finance",
                    "Romance friends and parents",
                    "Other domains",
                ]
                * 2,
            }
        )
        calls: list[tuple[str, str]] = []

        def fake_load_frame(root: Path, language: str, split: str) -> pd.DataFrame:
            calls.append((language, split))
            if split != "train":
                raise AssertionError("Stage 5 attempted to read a non-training split")
            return raw.copy()

        with patch("src.transformer_data.load_frame", side_effect=fake_load_frame):
            for task in TULU_TASKS:
                load_training_task(ROOT, task)
        self.assertEqual(calls, [("Tulu", "train"), ("Tulu", "train")])

    def test_input_check_records_no_test_or_submission(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            config = copy.deepcopy(self.extended)
            temporary = Path(directory)
            config["outputs"] = {
                "root": str(temporary / "outputs"),
                "checkpoints": str(temporary / "checkpoints"),
                "models": str(temporary / "models"),
                "logs": str(temporary / "logs"),
            }
            for task in TULU_TASKS:
                with self.subTest(task=task), patch(
                    "scripts.train_transformer.load_training_task",
                    return_value=synthetic_task(task),
                ):
                    result = check_inputs(config, EXTENDED_CONFIG, task)
                    self.assertFalse(result["test_files_read"])
                    self.assertFalse(result["submission_generated"])
                    task_dir = temporary / "outputs" / task
                    self.assertEqual(
                        {path.name for path in task_dir.iterdir()},
                        {"config_snapshot.json", "input_check.json", "split_manifest.csv"},
                    )


if __name__ == "__main__":
    unittest.main()
