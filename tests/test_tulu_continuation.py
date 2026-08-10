"""Model-free tests for Stage 5 baseline-best continuation fine-tuning."""

from __future__ import annotations

import ast
import copy
import inspect
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
    baseline_reference_metadata,
    build_trainer,
    check_inputs,
    continuation_contract,
    enforce_baseline_metric,
    load_config,
    model_inventory,
    output_paths,
    run_training,
    training_model_source,
)
from src.transformer_data import fixed_label_mapping


CONFIG_PATH = ROOT / "configs/tulu_regret_continuation.json"


def synthetic_regret(rows_per_class: int = 20) -> pd.DataFrame:
    label2id, _ = fixed_label_mapping("tulu_regret")
    rows = []
    for label, label_id in label2id.items():
        for index in range(rows_per_class):
            rows.append(
                {
                    "id": f"continuation-{label_id}-{index}",
                    "text": f"[TITLE] {label_id} {index}\n[TEXT] body",
                    "label": label,
                    "label_id": label_id,
                }
            )
    return pd.DataFrame(rows)


class TuluContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG_PATH)

    def test_exact_continuation_hyperparameters(self) -> None:
        self.assertEqual(self.config["allowed_tasks"], ["tulu_regret"])
        self.assertEqual(self.config["training"]["num_train_epochs"], 3)
        self.assertEqual(self.config["training"]["learning_rate"], 5e-6)
        self.assertEqual(self.config["training"]["warmup_ratio"], 0.0)
        self.assertEqual(self.config["training"]["early_stopping_patience"], 2)
        self.assertEqual(self.config["training"]["metric_for_best_model"], "macro_f1")
        contract = self.config["continuation"]
        self.assertEqual(contract["expected_macro_f1"], 0.5889065170528128)
        self.assertEqual(contract["training_time_macro_f1"], 0.59456571328454)
        self.assertEqual(contract["independent_reload_macro_f1"], 0.5889065170528128)
        self.assertTrue(contract["difference_due_to_one_inaction_prediction"])
        self.assertEqual(contract["baseline_macro_f1_absolute_tolerance"], 1e-6)

    def test_model_and_tokenizer_use_baseline_best_not_base_name(self) -> None:
        source = training_model_source(self.config, "tulu_regret")
        self.assertEqual(source, ROOT / "models/transformer_baseline/tulu_regret/best")
        self.assertNotEqual(str(source), self.config["model_name"])
        implementation = inspect.getsource(build_trainer)
        self.assertEqual(implementation.count("str(model_source)"), 2)

    def test_contract_forbids_all_state_restore(self) -> None:
        contract = continuation_contract(self.config, "tulu_regret")
        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertTrue(contract["initialized_from_baseline_best"])
        self.assertTrue(contract["weights_only"])
        for key in (
            "optimizer_state_restored",
            "scheduler_state_restored",
            "trainer_state_restored",
            "resume_from_checkpoint",
        ):
            self.assertFalse(contract[key])
        tree = ast.parse(inspect.getsource(run_training))
        train_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "train"
        ]
        self.assertEqual(len(train_calls), 1)
        self.assertEqual(train_calls[0].args, [])
        self.assertEqual(train_calls[0].keywords, [])

    def test_baseline_metric_gate_is_strict(self) -> None:
        expected = 0.5889065170528128
        self.assertEqual(enforce_baseline_metric(expected, expected, 1e-6), 0.0)
        self.assertLessEqual(enforce_baseline_metric(expected + 5e-7, expected, 1e-6), 1e-6)
        with self.assertRaisesRegex(RuntimeError, "Baseline reproduction gate failed"):
            enforce_baseline_metric(expected + 2e-6, expected, 1e-6)
        implementation = inspect.getsource(run_training)
        self.assertLess(
            implementation.index("evaluate_continuation_baseline"),
            implementation.index("trainer.train()"),
        )

    def test_baseline_reference_records_training_and_reload_metrics(self) -> None:
        reference = baseline_reference_metadata(self.config["continuation"])
        self.assertEqual(reference["expected_macro_f1"], 0.5889065170528128)
        self.assertEqual(reference["training_time_macro_f1"], 0.59456571328454)
        self.assertEqual(reference["independent_reload_macro_f1"], 0.5889065170528128)
        self.assertTrue(reference["difference_due_to_one_inaction_prediction"])
        self.assertEqual(reference["reference_kind"], "stable_independent_baseline_reload")

    def test_baseline_inventory_requires_nonempty_model_files(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_bytes(b"weights")
            inventory = model_inventory(model_dir)
            self.assertEqual({item["name"] for item in inventory}, {"config.json", "model.safetensors"})
            (model_dir / "model.safetensors").write_bytes(b"")
            with self.assertRaisesRegex(RuntimeError, "empty file"):
                model_inventory(model_dir)

    def test_continuation_outputs_are_fully_isolated(self) -> None:
        paths = output_paths(self.config, "tulu_regret")
        self.assertEqual(paths["output"], ROOT / "outputs/transformer_tulu_continuation/tulu_regret")
        self.assertEqual(
            paths["checkpoint"], ROOT / "checkpoints/transformer_tulu_continuation/tulu_regret"
        )
        self.assertEqual(paths["model"], ROOT / "models/transformer_tulu_continuation/tulu_regret/best")
        self.assertEqual(paths["log"], ROOT / "logs/transformer_tulu_continuation/tulu_regret")
        for path in paths.values():
            self.assertNotIn("transformer_baseline", path.parts)
            self.assertNotIn("transformer_tulu_extended", path.parts)

    def test_input_check_uses_train_only_and_generates_no_submission(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            config = copy.deepcopy(self.config)
            temporary = Path(directory)
            config["outputs"] = {
                "root": str(temporary / "outputs"),
                "checkpoints": str(temporary / "checkpoints"),
                "models": str(temporary / "models"),
                "logs": str(temporary / "logs"),
            }
            with patch("scripts.train_transformer.load_training_task", return_value=synthetic_regret()):
                result = check_inputs(config, CONFIG_PATH, "tulu_regret")
            self.assertFalse(result["test_files_read"])
            self.assertFalse(result["submission_generated"])


if __name__ == "__main__":
    unittest.main()
