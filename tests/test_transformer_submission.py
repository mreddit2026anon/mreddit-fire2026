"""Model-free tests for Stage 4 inference contracts and mixed assembly."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils import DOMAIN_LABELS, REGRET_LABELS, TASKS
from src.transformer_submission import (
    Stage4Error,
    build_candidate_frame,
    canonical_model_path,
    expected_task_ids,
    load_template,
    source_prediction_path,
    validate_candidate_package,
    validate_prediction_frame,
    validate_stage4_config,
    write_candidate_package,
)
from scripts.predict_transformer import task_contract, validate_only


def repeated(labels: tuple[str, ...], count: int, offset: int = 0) -> list[str]:
    return [labels[(index + offset) % len(labels)] for index in range(count)]


class TransformerSubmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = json.loads((ROOT / "configs/transformer_submission.json").read_text(encoding="utf-8"))
        official = self.root / "official"
        official.mkdir(parents=True)
        ids = (
            [f"english-{index}" for index in range(514)]
            + [f"kannada-{index}" for index in range(1370)]
            + [f"tulu-{index}" for index in range(685)]
        )
        pd.DataFrame({"id": ids, "Regret": "", "Domain": ""}).to_csv(
            official / "baseline_predictions.csv", index=False
        )
        template = load_template(self.root, self.config)
        for task in self.config["phase_b_predictions"]:
            task_ids = expected_task_ids(template, self.config, task)
            labels = repeated(TASKS[task]["labels"], len(task_ids), offset=0)
            path = source_prediction_path(self.root, self.config, task, "phase_b")
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"id": task_ids, "prediction": labels}).to_csv(path, index=False)
        for task in self.config["transformer_tasks"]:
            task_ids = expected_task_ids(template, self.config, task)
            labels = repeated(TASKS[task]["labels"], len(task_ids), offset=1)
            path = source_prediction_path(self.root, self.config, task, "transformer")
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "id": task_ids,
                    "predicted_label": labels,
                    "model_task": task,
                    "best_model_path": canonical_model_path(task),
                }
            ).to_csv(path, index=False)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_candidate_source_contracts(self) -> None:
        result = validate_stage4_config(self.config)
        self.assertTrue(result["passed"])
        candidate_a = self.config["candidates"]["candidate_a_phase_b"]["sources"]
        candidate_b = self.config["candidates"]["candidate_b_english_transformer"]["sources"]
        self.assertEqual(set(candidate_a.values()), {"phase_b"})
        self.assertEqual(candidate_b["english_regret"], "transformer")
        self.assertEqual(candidate_b["english_domain"], "transformer")
        self.assertEqual(candidate_b["tulu_regret"], "phase_b")
        self.assertEqual(candidate_b["tulu_domain"], "phase_b")
        self.assertEqual(candidate_b["kannada_regret"], "phase_b")

    def test_source_selection_changes_only_english_for_candidate_b(self) -> None:
        candidate_a, _ = build_candidate_frame(self.root, self.config, "candidate_a_phase_b")
        candidate_b, _ = build_candidate_frame(self.root, self.config, "candidate_b_english_transformer")
        english = candidate_a["id"].str.startswith("english-")
        non_english = ~english
        self.assertTrue((candidate_a.loc[english, "Regret"] != candidate_b.loc[english, "Regret"]).all())
        self.assertTrue((candidate_a.loc[english, "Domain"] != candidate_b.loc[english, "Domain"]).all())
        pd.testing.assert_frame_equal(candidate_a.loc[non_english], candidate_b.loc[non_english])

    def test_prediction_id_alignment_is_strict(self) -> None:
        task = "english_regret"
        path = source_prediction_path(self.root, self.config, task, "transformer")
        frame = pd.read_csv(path, dtype={"id": "string"}, keep_default_na=False)
        frame.loc[[0, 1], "id"] = frame.loc[[1, 0], "id"].to_numpy()
        template = load_template(self.root, self.config)
        with self.assertRaisesRegex(Stage4Error, "IDs do not exactly match"):
            validate_prediction_frame(
                frame,
                task,
                "transformer",
                expected_task_ids(template, self.config, task),
                canonical_model_path(task),
            )

    def test_illegal_prediction_label_is_rejected(self) -> None:
        task = "tulu_domain"
        path = source_prediction_path(self.root, self.config, task, "transformer")
        frame = pd.read_csv(path, dtype={"id": "string"}, keep_default_na=False)
        frame.loc[0, "predicted_label"] = "Not a domain"
        template = load_template(self.root, self.config)
        with self.assertRaisesRegex(Stage4Error, "invalid labels"):
            validate_prediction_frame(
                frame,
                task,
                "transformer",
                expected_task_ids(template, self.config, task),
                canonical_model_path(task),
            )

    def test_kannada_domain_is_empty(self) -> None:
        frame, _ = build_candidate_frame(self.root, self.config, "candidate_b_english_transformer")
        kannada = frame["id"].str.startswith("kannada-")
        self.assertTrue(frame.loc[kannada, "Domain"].eq("").all())
        self.assertTrue(frame.loc[kannada, "Regret"].isin(REGRET_LABELS).all())
        self.assertTrue(frame.loc[~kannada, "Domain"].isin(DOMAIN_LABELS).all())

    def test_zip_contains_only_root_submission_csv_and_crc_passes(self) -> None:
        candidate = "candidate_b_english_transformer"
        frame, metadata = build_candidate_frame(self.root, self.config, candidate)
        result = write_candidate_package(self.root, self.config, candidate, frame, metadata)
        zip_path = self.root / result["zip_path"]
        with zipfile.ZipFile(zip_path, "r") as archive:
            self.assertEqual(archive.namelist(), ["submission.csv"])
            self.assertIsNone(archive.testzip())
        validated = validate_candidate_package(self.root, self.config, candidate)
        self.assertTrue(validated["passed"])
        self.assertFalse(validated["test_text_read"])

    def test_existing_candidate_is_not_overwritten(self) -> None:
        candidate = "candidate_a_phase_b"
        frame, metadata = build_candidate_frame(self.root, self.config, candidate)
        write_candidate_package(self.root, self.config, candidate, frame, metadata)
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            write_candidate_package(self.root, self.config, candidate, frame, metadata)

    def test_config_dry_validation_performs_no_file_reads(self) -> None:
        with patch("pandas.read_csv", side_effect=AssertionError("unexpected read")):
            result = validate_stage4_config(self.config)
        self.assertTrue(result["passed"])
        self.assertFalse(result["test_text_read"])
        self.assertFalse(result["model_loaded"])

    def test_transformer_metadata_is_strict(self) -> None:
        task = "english_domain"
        path = source_prediction_path(self.root, self.config, task, "transformer")
        frame = pd.read_csv(path, dtype={"id": "string"}, keep_default_na=False)
        frame.loc[0, "best_model_path"] = "models/wrong"
        template = load_template(self.root, self.config)
        with self.assertRaisesRegex(Stage4Error, "best_model_path metadata mismatch"):
            validate_prediction_frame(
                frame,
                task,
                "transformer",
                expected_task_ids(template, self.config, task),
                canonical_model_path(task),
            )

    def test_kannada_transformer_inference_is_rejected(self) -> None:
        for task in ("kannada_regret", "kannada_domain"):
            with self.subTest(task=task), self.assertRaisesRegex(Stage4Error, "not allowed"):
                task_contract(self.config, task)

    def test_inference_validate_only_does_not_read_raw_test(self) -> None:
        with (
            patch("scripts.predict_transformer.ROOT", self.root),
            patch("scripts.predict_transformer.load_frame", side_effect=AssertionError("raw test read")),
        ):
            result = validate_only(self.config, "english_domain")
        self.assertTrue(result["passed"])
        self.assertFalse(result["test_text_read"])
        self.assertFalse(result["model_loaded"])


if __name__ == "__main__":
    unittest.main()
