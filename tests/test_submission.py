"""Standard-library tests for strict submission validation."""

from __future__ import annotations

import sys
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_submission import SubmissionError, validate_submission_csv, validate_submission_zip
from src.data_utils import load_frame


class SubmissionValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = tempfile.TemporaryDirectory()
        self.fixture_root = Path(self.fixture.name)
        official_dir = self.fixture_root / "official"
        test_dir = self.fixture_root / "data/raw/test_data"
        official_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)

        english_ids = [f"synthetic-en-{index:04d}" for index in range(514)]
        kannada_ids = [f"synthetic-kn-{index:04d}" for index in range(1370)]
        tulu_ids = [f"synthetic-tu-{index:04d}" for index in range(685)]
        pd.DataFrame(
            {"id": english_ids, "title": ["title"] * len(english_ids), "text": ["text"] * len(english_ids)}
        ).to_csv(
            test_dir / "english_test_data.csv", index=False
        )
        pd.DataFrame({"id": kannada_ids, "Text": ["text"] * len(kannada_ids)}).to_csv(
            test_dir / "kannada_test_data.csv", index=False
        )
        pd.DataFrame(
            {"id": tulu_ids, "title": ["title"] * len(tulu_ids), "text": ["text"] * len(tulu_ids)}
        ).to_csv(
            test_dir / "tulu_test_data.csv", index=False
        )

        ids = english_ids + kannada_ids + tulu_ids
        self.template = pd.DataFrame(
            {
                "id": ids,
                "Regret": ["Action"] * len(ids),
                "Domain": ["Education"] * len(english_ids)
                + [""] * len(kannada_ids)
                + ["Education"] * len(tulu_ids),
            }
        )
        self.template.to_csv(official_dir / "baseline_predictions.csv", index=False, lineterminator="\n")

    def tearDown(self) -> None:
        self.fixture.cleanup()

    def write_csv(self, directory: Path, frame: pd.DataFrame) -> Path:
        path = directory / "submission.csv"
        frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
        return path

    def test_synthetic_template_is_valid(self) -> None:
        result = validate_submission_csv(
            self.fixture_root / "official/baseline_predictions.csv", self.fixture_root
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["rows"], 2569)

    def test_rejects_id_order_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            frame = self.template.copy()
            frame.iloc[[0, 1]] = frame.iloc[[1, 0]].to_numpy()
            path = self.write_csv(Path(temporary), frame)
            with self.assertRaisesRegex(SubmissionError, "ID order mismatch"):
                validate_submission_csv(path, self.fixture_root)

    def test_rejects_nonempty_kannada_domain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            frame = self.template.copy()
            kannada_id = load_frame(self.fixture_root, "Kannada", "test")["id"].astype(str).iloc[0]
            frame.loc[frame["id"].astype(str).eq(kannada_id), "Domain"] = "Other domains"
            path = self.write_csv(Path(temporary), frame)
            with self.assertRaisesRegex(SubmissionError, "Kannada Domain"):
                validate_submission_csv(path, self.fixture_root)

    def test_valid_zip_has_one_root_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            csv_path = self.write_csv(directory, self.template)
            zip_path = directory / "submission.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(csv_path, arcname="submission.csv")
            result = validate_submission_zip(zip_path, self.fixture_root)
            self.assertEqual(result["zip_members"], ["submission.csv"])

    def test_rejects_nested_zip_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            csv_path = self.write_csv(directory, self.template)
            zip_path = directory / "submission.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.write(csv_path, arcname="nested/submission.csv")
            with self.assertRaisesRegex(SubmissionError, "root submission.csv"):
                validate_submission_zip(zip_path, self.fixture_root)

    def test_cli_returns_nonzero_for_structural_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            frame = self.template.drop(columns=["Domain"])
            path = self.write_csv(Path(temporary), frame)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/validate_submission.py"),
                    "--csv",
                    str(path),
                    "--root",
                    str(self.fixture_root),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SUBMISSION VALIDATION FAILED", result.stderr)


if __name__ == "__main__":
    unittest.main()
