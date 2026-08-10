"""Lightweight, model-download-free tests for the Phase 04 pipeline."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transformer_data import (
    DOMAIN_LABEL2ID,
    REGRET_LABEL2ID,
    TransformerDataError,
    compute_class_weights,
    fixed_label_mapping,
    get_task_spec,
    load_training_task,
    make_stratified_split,
)
from scripts.train_transformer import epoch_metric_rows, load_config, output_paths


def synthetic_english_domain(rows_per_class: int = 20) -> pd.DataFrame:
    rows = []
    for label, label_id in DOMAIN_LABEL2ID.items():
        for index in range(rows_per_class):
            rows.append(
                {
                    "id": f"{label_id}-{index}",
                    "text": f"[TITLE] title {label_id} {index}\n[TEXT] body {label_id} {index}",
                    "label": label,
                    "label_id": label_id,
                }
            )
    return pd.DataFrame(rows)


def synthetic_task(task: str, rows_per_class: int = 20) -> pd.DataFrame:
    label2id, _ = fixed_label_mapping(task)
    rows = []
    for label, label_id in label2id.items():
        for index in range(rows_per_class):
            rows.append(
                {
                    "id": f"{task}-{label_id}-{index}",
                    "text": f"[TITLE] {task} {label_id} {index}\n[TEXT] body {label_id} {index}",
                    "label": label,
                    "label_id": label_id,
                }
            )
    return pd.DataFrame(rows)


class TransformerPipelineTests(unittest.TestCase):
    def test_fixed_label_mappings(self) -> None:
        self.assertEqual(
            REGRET_LABEL2ID,
            {"Action": 0, "Inaction": 1, "No Regret": 2},
        )
        self.assertEqual(
            DOMAIN_LABEL2ID,
            {
                "Education": 0,
                "Health": 1,
                "Career and Finance": 2,
                "Romance friends and parents": 3,
                "Other domains": 4,
            },
        )
        self.assertEqual(fixed_label_mapping("english_domain")[0], DOMAIN_LABEL2ID)
        self.assertEqual(fixed_label_mapping("tulu_domain")[0], DOMAIN_LABEL2ID)
        for task in ("english_regret", "tulu_regret"):
            self.assertEqual(fixed_label_mapping(task)[0], REGRET_LABEL2ID)

    def test_stratified_split_is_reproducible_and_disjoint(self) -> None:
        frame = synthetic_english_domain()
        first = make_stratified_split(frame, "english_domain", 0.2, 2026)
        second = make_stratified_split(frame, "english_domain", 0.2, 2026)
        self.assertEqual(first.train["id"].tolist(), second.train["id"].tolist())
        self.assertEqual(first.validation["id"].tolist(), second.validation["id"].tolist())
        self.assertFalse(set(first.train["id"]) & set(first.validation["id"]))
        self.assertEqual(set(first.train["label"]), set(DOMAIN_LABEL2ID))
        self.assertEqual(set(first.validation["label"]), set(DOMAIN_LABEL2ID))

    def test_training_loader_requests_train_only(self) -> None:
        labels = list(DOMAIN_LABEL2ID)
        raw = pd.DataFrame(
            {
                "id": [f"id-{index}" for index in range(10)],
                "title": [f"title {index}" for index in range(10)],
                "text": [f"body {index}" for index in range(10)],
                "Regret": ["Action"] * 10,
                "Domain": [labels[index % len(labels)] for index in range(10)],
            }
        )
        calls = []

        def fake_load_frame(root: Path, language: str, split: str) -> pd.DataFrame:
            calls.append((language, split))
            if split != "train":
                raise AssertionError("Transformer training loader attempted a non-train split")
            return raw.copy()

        with patch("src.transformer_data.load_frame", side_effect=fake_load_frame):
            loaded = load_training_task(ROOT, "english_domain")
        self.assertEqual(calls, [("English", "train")])
        self.assertEqual(len(loaded), len(raw))

    def test_kannada_domain_is_rejected(self) -> None:
        with self.assertRaisesRegex(TransformerDataError, "Kannada Domain"):
            get_task_spec("kannada_domain")

    def test_class_weights_use_training_labels_only(self) -> None:
        frame = synthetic_english_domain(rows_per_class=25)
        split = make_stratified_split(frame, "english_domain", 0.2, 2026)
        before = compute_class_weights(split.train["label_id"], len(DOMAIN_LABEL2ID))
        changed_validation = split.validation.copy()
        changed_validation["label_id"] = 0
        after = compute_class_weights(split.train["label_id"], len(DOMAIN_LABEL2ID))
        np.testing.assert_allclose(before, after)
        self.assertAlmostEqual(float(before.mean()), 1.0, places=6)
        combined = compute_class_weights(
            pd.concat([split.train["label_id"], changed_validation["label_id"]]),
            len(DOMAIN_LABEL2ID),
        )
        self.assertFalse(np.allclose(before, combined))

    def test_config_contains_fixed_english_domain_parameters(self) -> None:
        config = load_config(ROOT / "configs/transformer_baseline.json")
        self.assertEqual(config["model_name"], "FacebookAI/xlm-roberta-base")
        self.assertEqual(config["first_task"], "english_domain")
        self.assertEqual(
            config["allowed_tasks"],
            ["english_domain", "english_regret", "tulu_regret", "tulu_domain"],
        )
        self.assertEqual(config["text"]["max_length"], 256)
        self.assertEqual(config["training"]["metric_for_best_model"], "macro_f1")
        self.assertTrue(config["training"]["bf16"])

    def test_enabled_task_configs_match_repository_contract(self) -> None:
        config = load_config(ROOT / "configs/transformer_baseline.json")
        expected = {
            "english_domain": ("English", "Domain"),
            "english_regret": ("English", "Regret"),
            "tulu_regret": ("Tulu", "Regret"),
            "tulu_domain": ("Tulu", "Domain"),
        }
        self.assertEqual(set(config["allowed_tasks"]), set(expected))
        for task, (language, target) in expected.items():
            self.assertEqual(config["tasks"][task]["language"], language)
            self.assertEqual(config["tasks"][task]["target"], target)
            self.assertEqual(config["tasks"][task]["validation"], "stratified_80_20")

    def test_new_task_splits_are_reproducible_and_isolated(self) -> None:
        all_ids = set()
        for task in ("english_regret", "tulu_regret", "tulu_domain"):
            frame = synthetic_task(task)
            first = make_stratified_split(frame, task, 0.2, 2026)
            second = make_stratified_split(frame, task, 0.2, 2026)
            self.assertEqual(first.train["id"].tolist(), second.train["id"].tolist())
            self.assertEqual(first.validation["id"].tolist(), second.validation["id"].tolist())
            task_ids = set(first.train["id"]) | set(first.validation["id"])
            self.assertFalse(all_ids & task_ids)
            self.assertFalse(set(first.train["id"]) & set(first.validation["id"]))
            self.assertEqual(set(first.train["label"]), set(first.label2id))
            self.assertEqual(set(first.validation["label"]), set(first.label2id))
            all_ids.update(task_ids)

    def test_output_paths_are_isolated_by_task_and_mode(self) -> None:
        config = load_config(ROOT / "configs/transformer_baseline.json")
        tasks = config["allowed_tasks"]
        formal = {task: output_paths(config, task) for task in tasks}
        smoke = {task: output_paths(config, task, smoke=True) for task in tasks}
        for key in ("output", "checkpoint", "model", "log"):
            self.assertEqual(len({str(formal[task][key]) for task in tasks}), len(tasks))
            self.assertEqual(len({str(smoke[task][key]) for task in tasks}), len(tasks))
            for task in tasks:
                self.assertNotEqual(formal[task][key], smoke[task][key])
                self.assertIn(task, formal[task][key].parts)
                self.assertIn(task, smoke[task][key].parts)

    def test_epoch_metric_history_extraction(self) -> None:
        history = [
            {"loss": 1.0, "epoch": 1.0, "step": 10},
            {
                "eval_macro_f1": 0.5,
                "eval_accuracy": 0.6,
                "eval_weighted_f1": 0.55,
                "eval_loss": 0.9,
                "epoch": 1.0,
                "step": 10,
            },
            {
                "eval_macro_f1": 0.7,
                "eval_accuracy": 0.8,
                "eval_weighted_f1": 0.75,
                "eval_loss": 0.6,
                "epoch": 2.0,
                "step": 20,
            },
        ]
        rows = epoch_metric_rows(history, total_epochs=5)
        self.assertEqual([row["epoch"] for row in rows], [1, 2])
        self.assertEqual([row["progress_percent"] for row in rows], [20.0, 40.0])
        self.assertEqual(rows[1]["macro_f1"], 0.7)

    def test_epoch_progress_uses_configured_total_epochs(self) -> None:
        history = [
            {
                "eval_macro_f1": 0.2,
                "eval_accuracy": 0.3,
                "eval_weighted_f1": 0.25,
                "eval_loss": 1.0,
                "epoch": float(epoch),
                "step": epoch * 137,
            }
            for epoch in range(1, 9)
        ]
        rows = epoch_metric_rows(history, total_epochs=8)
        self.assertEqual(rows[0]["progress_percent"], 12.5)
        self.assertEqual(rows[4]["progress_percent"], 62.5)
        self.assertEqual(rows[-1]["progress_percent"], 100.0)
        self.assertLessEqual(max(row["progress_percent"] for row in rows), 100.0)

    def test_training_cli_remains_separate_from_stage4_inference(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/train_transformer.py"), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--check-inputs", result.stdout)
        self.assertIn("--check-baseline", result.stdout)
        self.assertIn("--smoke", result.stdout)
        self.assertIn("--evaluate-only", result.stdout)
        self.assertNotIn("--predict", result.stdout)
        self.assertNotIn("--submission", result.stdout)
        self.assertTrue((ROOT / "scripts/predict_transformer.py").exists())
        self.assertTrue((ROOT / "scripts/build_mixed_submission.py").exists())
        self.assertFalse((ROOT / "scripts/finalize_transformer_submission.py").exists())


if __name__ == "__main__":
    unittest.main()
