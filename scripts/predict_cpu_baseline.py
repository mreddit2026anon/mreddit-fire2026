"""Predict the three official test sets with the five finalized models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils import TASKS, build_text, load_frame


PREDICTION_DIR = ROOT / "outputs/cpu_baseline/predictions"
MODEL_DIR = ROOT / "models/cpu_baseline"


def main() -> int:
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    distributions = {}
    for task_name, spec in TASKS.items():
        model_path = MODEL_DIR / f"{task_name}.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing finalized model: {model_path}")
        frame = load_frame(ROOT, spec["language"], "test")
        texts = build_text(frame, spec["language"])
        model = joblib.load(model_path)
        prediction = model.predict(texts)
        invalid = sorted(set(prediction) - set(spec["labels"]))
        if invalid:
            raise RuntimeError(f"{task_name}: invalid predictions {invalid}")
        output = pd.DataFrame({"id": frame["id"].astype(str), "prediction": prediction})
        if output["id"].tolist() != frame["id"].astype(str).tolist():
            raise RuntimeError(f"{task_name}: test ID order changed")
        path = PREDICTION_DIR / f"{task_name}.csv"
        output.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
        distributions[task_name] = output["prediction"].value_counts().sort_index().to_dict()
        print(f"{task_name}: {len(output)} rows -> {path}; distribution={distributions[task_name]}")
        del model
    (PREDICTION_DIR / "prediction_distributions.json").write_text(
        json.dumps(distributions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
