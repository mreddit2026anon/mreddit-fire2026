"""Merge five prediction files by ID and create a validated Codabench ZIP."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_submission import validate_submission_csv, validate_submission_zip
from src.data_utils import TASKS, load_frame


PREDICTION_DIR = ROOT / "outputs/cpu_baseline/predictions"
SUBMISSION_DIR = ROOT / "outputs/cpu_baseline/submission"


def load_prediction(task_name: str) -> pd.DataFrame:
    path = PREDICTION_DIR / f"{task_name}.csv"
    frame = pd.read_csv(path, dtype={"id": "string"}, keep_default_na=False)
    if tuple(frame.columns) != ("id", "prediction"):
        raise RuntimeError(f"{path}: expected columns id,prediction")
    spec = TASKS[task_name]
    test = load_frame(ROOT, spec["language"], "test")
    if frame["id"].astype(str).tolist() != test["id"].astype(str).tolist():
        raise RuntimeError(f"{path}: IDs do not exactly match the source test order")
    if frame["id"].duplicated().any():
        raise RuntimeError(f"{path}: duplicate IDs")
    invalid = sorted(set(frame["prediction"]) - set(spec["labels"]))
    if invalid:
        raise RuntimeError(f"{path}: invalid labels {invalid}")
    return frame


def main() -> int:
    predictions = {task: load_prediction(task) for task in TASKS}
    template = pd.read_csv(
        ROOT / "official/baseline_predictions.csv", dtype={"id": "string"}, keep_default_na=False
    )
    submission = pd.DataFrame({"id": template["id"].astype(str), "Regret": "", "Domain": ""})

    for task_name, frame in predictions.items():
        target = TASKS[task_name]["target"]
        mapping = dict(zip(frame["id"].astype(str), frame["prediction"], strict=True))
        mask = submission["id"].isin(mapping)
        if int(mask.sum()) != len(frame):
            raise RuntimeError(f"{task_name}: prediction IDs do not map one-to-one into template")
        submission.loc[mask, target] = submission.loc[mask, "id"].map(mapping)

    if submission["Regret"].eq("").any():
        raise RuntimeError("Unfilled Regret predictions")
    english_tulu_ids = set(load_frame(ROOT, "English", "test")["id"].astype(str)) | set(
        load_frame(ROOT, "Tulu", "test")["id"].astype(str)
    )
    if submission.loc[submission["id"].isin(english_tulu_ids), "Domain"].eq("").any():
        raise RuntimeError("Unfilled English/Tulu Domain predictions")

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SUBMISSION_DIR / "submission.csv"
    zip_path = SUBMISSION_DIR / "submission.zip"
    submission.to_csv(csv_path, index=False, encoding="utf-8", lineterminator="\n")
    csv_result = validate_submission_csv(csv_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname="submission.csv")
    zip_result = validate_submission_zip(zip_path)
    print(f"CSV: {csv_path} -> {csv_result}")
    print(f"ZIP: {zip_path} -> {zip_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
