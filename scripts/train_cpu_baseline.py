"""Train, evaluate, and finalize the fixed Phase A CPU baselines."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline_models import build_model, evaluate_predictions
from src.data_utils import TASKS, build_text, fixed_stratified_split, load_frame, load_task_data, text_hash, validate_all_inputs


EVALUATION_DIR = ROOT / "outputs/cpu_baseline/evaluation"
MODEL_DIR = ROOT / "models/cpu_baseline"
LOG_DIR = ROOT / "logs/cpu_baseline"
TABLE_DIR = ROOT / "reports/tables"
REPORT_PATH = ROOT / "reports/cpu_baseline_results.md"


def load_config() -> dict:
    return json.loads((ROOT / "configs/cpu_baseline.json").read_text(encoding="utf-8"))


def setup_logging(mode: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = LOG_DIR / f"{stamp}_{mode}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return path


def save_run_record(name: str, config: dict, payload: dict) -> Path:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    path = EVALUATION_DIR / f"{name}.json"
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "scikit_learn": sklearn.__version__,
        "config": config,
        **payload,
    }
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def smoke_test(config: dict) -> None:
    results = []
    smoke_dir = MODEL_DIR / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    for task_name, spec in TASKS.items():
        texts, labels, _ = load_task_data(ROOT, task_name)
        sample = pd.DataFrame({"text": texts, "label": labels}).groupby("label", group_keys=False).sample(
            n=6, random_state=config["seed"]
        )
        train_parts = []
        validation_parts = []
        for _, group in sample.groupby("label", sort=False):
            train_parts.append(group.iloc[:4])
            validation_parts.append(group.iloc[4:])
        train = pd.concat(train_parts, ignore_index=True)
        validation = pd.concat(validation_parts, ignore_index=True)
        model = build_model(config)
        model.fit(train["text"], train["label"])
        prediction = model.predict(validation["text"])
        base = {"task": task_name, "language": spec["language"], "target": spec["target"], "eval_split": "smoke", "fold": ""}
        metric, _, _ = evaluate_predictions(validation["label"], prediction, spec["labels"], base)
        path = smoke_dir / f"{task_name}.joblib"
        joblib.dump(model, path)
        loaded = joblib.load(path)
        reloaded_prediction = loaded.predict(validation["text"])
        if not np.array_equal(prediction, reloaded_prediction):
            raise RuntimeError(f"{task_name}: predictions changed after model save/load")
        results.append({**metric, "model_path": str(path.relative_to(ROOT)), "save_load_match": True})
        logging.info("Smoke %s passed: macro_f1=%.4f", task_name, metric["macro_f1"])
        del model, loaded
        gc.collect()
    save_run_record("smoke", config, {"results": results})


def evaluate_standard_task(task_name: str, config: dict) -> dict:
    spec = TASKS[task_name]
    texts, labels, _ = load_task_data(ROOT, task_name)
    train_indices, validation_indices = fixed_stratified_split(labels)
    if set(labels.iloc[validation_indices]) != set(spec["labels"]):
        raise RuntimeError(f"{task_name}: stratified validation does not contain every label")
    model = build_model(config)
    model.fit(texts.iloc[train_indices], labels.iloc[train_indices])
    prediction = model.predict(texts.iloc[validation_indices])
    base = {
        "task": task_name,
        "language": spec["language"],
        "target": spec["target"],
        "eval_split": "stratified_80_20",
        "fold": "",
        "n_train": len(train_indices),
        "n_eval": len(validation_indices),
    }
    metric, per_class, confusion = evaluate_predictions(labels.iloc[validation_indices], prediction, spec["labels"], base)
    logging.info("%s macro_f1=%.6f accuracy=%.6f weighted_f1=%.6f", task_name, metric["macro_f1"], metric["accuracy"], metric["weighted_f1"])
    del model
    gc.collect()
    return {"metrics": [metric], "per_class": per_class, "confusion": confusion}


def evaluate_kannada(config: dict) -> dict:
    task_name = "kannada_regret"
    spec = TASKS[task_name]
    train = load_frame(ROOT, "Kannada", "train")
    texts = build_text(train, "Kannada")
    labels = train["Regret"].astype(str)
    groups = texts.map(text_hash)
    cv_config = config["kannada_group_cv"]
    splitter = StratifiedGroupKFold(
        n_splits=cv_config["n_splits"],
        shuffle=cv_config["shuffle"],
        random_state=cv_config["random_state"],
    )
    metrics: list[dict] = []
    per_class: list[dict] = []
    confusion: list[dict] = []
    for fold, (train_indices, validation_indices) in enumerate(splitter.split(texts, labels, groups), start=1):
        train_groups = set(groups.iloc[train_indices])
        validation_groups = set(groups.iloc[validation_indices])
        if train_groups & validation_groups:
            raise RuntimeError(f"Kannada fold {fold}: normalized-text groups overlap")
        if set(labels.iloc[validation_indices]) != set(spec["labels"]):
            raise RuntimeError(f"Kannada fold {fold}: validation does not contain every label")
        model = build_model(config)
        model.fit(texts.iloc[train_indices], labels.iloc[train_indices])
        prediction = model.predict(texts.iloc[validation_indices])
        base = {
            "task": task_name,
            "language": "Kannada",
            "target": "Regret",
            "eval_split": "stratified_group_5fold",
            "fold": fold,
            "n_train": len(train_indices),
            "n_eval": len(validation_indices),
        }
        metric, class_rows, matrix_rows = evaluate_predictions(labels.iloc[validation_indices], prediction, spec["labels"], base)
        metrics.append(metric)
        per_class.extend(class_rows)
        confusion.extend(matrix_rows)
        logging.info("Kannada fold %d macro_f1=%.6f", fold, metric["macro_f1"])
        del model
        gc.collect()

    fold_metrics = pd.DataFrame(metrics)
    for statistic, values in (("mean", fold_metrics[["macro_f1", "accuracy", "weighted_f1"]].mean()), ("std", fold_metrics[["macro_f1", "accuracy", "weighted_f1"]].std(ddof=1))):
        metrics.append(
            {
                "task": task_name,
                "language": "Kannada",
                "target": "Regret",
                "eval_split": f"stratified_group_5fold_{statistic}",
                "fold": "",
                "n_train": len(train),
                "n_eval": len(train),
                "macro_f1": float(values["macro_f1"]),
                "accuracy": float(values["accuracy"]),
                "weighted_f1": float(values["weighted_f1"]),
            }
        )

    official_validation = load_frame(ROOT, "Kannada", "validation")
    validation_texts = build_text(official_validation, "Kannada")
    model = build_model(config)
    model.fit(texts, labels)
    prediction = model.predict(validation_texts)
    base = {
        "task": task_name,
        "language": "Kannada",
        "target": "Regret",
        "eval_split": "official_validation_reference_only",
        "fold": "",
        "n_train": len(train),
        "n_eval": len(official_validation),
    }
    metric, class_rows, matrix_rows = evaluate_predictions(
        official_validation["Regret"].astype(str), prediction, spec["labels"], base
    )
    metrics.append(metric)
    per_class.extend(class_rows)
    confusion.extend(matrix_rows)
    logging.info("Kannada official validation reference-only macro_f1=%.6f", metric["macro_f1"])
    del model
    gc.collect()
    return {"metrics": metrics, "per_class": per_class, "confusion": confusion}


def evaluate_task(task_name: str, config: dict) -> None:
    payload = evaluate_kannada(config) if task_name == "kannada_regret" else evaluate_standard_task(task_name, config)
    save_run_record(task_name, config, payload)


def finalize_task(task_name: str, config: dict) -> None:
    include_validation = task_name == "kannada_regret"
    texts, labels, ids = load_task_data(ROOT, task_name, include_kannada_validation=include_validation)
    model = build_model(config)
    model.fit(texts, labels)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{task_name}.joblib"
    joblib.dump(model, model_path)
    metadata = {
        "task": task_name,
        "language": TASKS[task_name]["language"],
        "target": TASKS[task_name]["target"],
        "labels": list(TASKS[task_name]["labels"]),
        "training_rows": len(texts),
        "training_id_sha256": __import__("hashlib").sha256("\n".join(ids).encode("utf-8")).hexdigest(),
        "includes_kannada_official_validation": include_validation,
        "config": config,
    }
    (MODEL_DIR / f"{task_name}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Finalized %s with %d labeled rows -> %s", task_name, len(texts), model_path)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    shown = frame.loc[:, columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in shown.itertuples(index=False, name=None):
        values = [f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(config: dict) -> None:
    required = list(TASKS)
    missing = [task for task in required if not (EVALUATION_DIR / f"{task}.json").is_file()]
    if missing:
        raise RuntimeError(f"Missing evaluation records: {missing}")
    records = [json.loads((EVALUATION_DIR / f"{task}.json").read_text(encoding="utf-8")) for task in required]
    metrics = pd.DataFrame([row for record in records for row in record["metrics"]])
    per_class = pd.DataFrame([row for record in records for row in record["per_class"]])
    confusion = pd.DataFrame([row for record in records for row in record["confusion"]])
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(TABLE_DIR / "cpu_baseline_metrics.csv", index=False, encoding="utf-8", lineterminator="\n")
    per_class.to_csv(TABLE_DIR / "cpu_baseline_per_class.csv", index=False, encoding="utf-8", lineterminator="\n")
    confusion.to_csv(TABLE_DIR / "cpu_baseline_confusion.csv", index=False, encoding="utf-8", lineterminator="\n")
    key_rows = metrics[
        metrics["eval_split"].isin(["stratified_80_20", "stratified_group_5fold_mean", "stratified_group_5fold_std", "official_validation_reference_only"])
    ]
    report = f"""# Phase A CPU Baseline Results

Generated with seed `{config['seed']}` using character TF-IDF `(3, 5)` and balanced LinearSVC (`C=1.0`). Test data were not used for fitting or model selection.

## Validation summary

{markdown_table(key_rows, ['task', 'eval_split', 'macro_f1', 'accuracy', 'weighted_f1', 'n_train', 'n_eval'])}

Kannada model selection uses the five-fold normalized-text-hash group-aware result. The official Kannada validation score is reference-only because exact normalized texts overlap the training set.

## Final model policy

All five tasks use the fixed Phase A model. English and Tulu final models use their full training sets. The Kannada final model uses the full Kannada train and official validation sets after evaluation is complete. Kannada Domain is never trained or predicted.

Detailed metrics are in `reports/tables/cpu_baseline_metrics.csv`, `cpu_baseline_per_class.csv`, and `cpu_baseline_confusion.csv`.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    logging.info("Wrote metrics tables and %s", REPORT_PATH)


def main() -> int:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check-inputs", action="store_true")
    actions.add_argument("--smoke", action="store_true")
    actions.add_argument("--task", choices=list(TASKS))
    actions.add_argument("--finalize", choices=list(TASKS))
    actions.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    mode = "check_inputs" if args.check_inputs else "smoke" if args.smoke else f"task_{args.task}" if args.task else f"finalize_{args.finalize}" if args.finalize else "write_report"
    log_path = setup_logging(mode)
    config = load_config()
    logging.info("Using config %s; log=%s", ROOT / "configs/cpu_baseline.json", log_path)
    if args.check_inputs:
        result = validate_all_inputs(ROOT)
        save_run_record("input_check", config, {"result": result})
        logging.info("Input validation passed: %s", result)
    elif args.smoke:
        smoke_test(config)
    elif args.task:
        evaluate_task(args.task, config)
    elif args.finalize:
        finalize_task(args.finalize, config)
    else:
        write_report(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
