"""Finalize the fixed Phase B task-level model mixture and submission.

No model selection or hyperparameter search is performed here. English and
Tulu Regret use the already-selected Phase B configurations. English/Tulu
Domain and Kannada Regret are copied byte-for-byte from the Phase A final
models so their behavior remains unchanged.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import logging
import shutil
import subprocess
import sys
import time
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_submission import (
    SubmissionError,
    validate_submission_bytes,
    validate_submission_csv,
    validate_submission_zip,
)
from src.baseline_models import build_model as build_phase_a_model
from src.data_utils import (
    DOMAIN_LABELS,
    LANGUAGE_ORDER,
    REGRET_LABELS,
    TASKS,
    build_text,
    load_frame,
    load_task_data,
    validate_all_inputs,
)


CONFIG_PATH = ROOT / "configs/phase_b_final.json"
PHASE_A_CONFIG_PATH = ROOT / "configs/cpu_baseline.json"
PHASE_B_METRICS_PATH = ROOT / "reports/tables/phase_b_cv_metrics.csv"
MODEL_DIR = ROOT / "models/phase_b_final"
PREDICTION_DIR = ROOT / "outputs/phase_b_final/predictions"
SUBMISSION_DIR = ROOT / "outputs/phase_b_final/submission"
STATE_DIR = ROOT / "outputs/phase_b_final/state"
SMOKE_DIR = ROOT / "outputs/phase_b_final/smoke"
LOG_DIR = ROOT / "logs/phase_b_final"
REPORT_PATH = ROOT / "reports/phase_b_final_submission.md"
INPUT_CHECK_PATH = STATE_DIR / "input_check.json"
SMOKE_PATH = STATE_DIR / "smoke.json"
PREDICTION_SUMMARY_PATH = STATE_DIR / "prediction_summary.json"
SUBMISSION_VALIDATION_PATH = STATE_DIR / "submission_validation.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config() -> dict[str, Any]:
    return load_json(CONFIG_PATH)


def load_phase_a_config() -> dict[str, Any]:
    return load_json(PHASE_A_CONFIG_PATH)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(ids: pd.Series) -> str:
    return hashlib.sha256("\n".join(ids.astype(str)).encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


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


def tfidf(spec: dict[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=spec["analyzer"],
        ngram_range=tuple(spec["ngram_range"]),
        min_df=spec["min_df"],
        sublinear_tf=spec["sublinear_tf"],
        max_features=spec["max_features"],
        dtype=np.float32,
    )


def build_phase_b_model(config: dict[str, Any], task: str) -> Pipeline:
    task_config = config["tasks"][task]
    if task_config["action"] != "train":
        return build_phase_a_model(load_phase_a_config())
    feature_name = task_config["feature"]
    if feature_name == "word_char":
        feature_config = config["features"]["word_char"]
        features: Any = FeatureUnion(
            [
                ("char", tfidf(feature_config["char"])),
                ("word", tfidf(feature_config["word"])),
            ],
            n_jobs=1,
        )
    elif feature_name == "char":
        features = tfidf(config["features"]["char"])
    else:
        raise ValueError(f"Unsupported fixed feature for {task}: {feature_name}")
    classifier_config = config["classifiers"][task_config["classifier"]]
    base = LogisticRegression(
        solver=classifier_config["solver"],
        C=classifier_config["C"],
        max_iter=classifier_config["max_iter"],
        class_weight=classifier_config["class_weight"],
        random_state=classifier_config["random_state"],
    )
    classifier = OneVsRestClassifier(base, n_jobs=1)
    return Pipeline([("features", features), ("classifier", classifier)])


def model_feature_count(model: Pipeline) -> int:
    if "features" in model.named_steps:
        features = model.named_steps["features"]
        if isinstance(features, FeatureUnion):
            return int(sum(len(transformer.vocabulary_) for _, transformer in features.transformer_list))
        return int(len(features.vocabulary_))
    return int(len(model.named_steps["tfidf"].vocabulary_))


def phase_a_inventory() -> dict[str, str]:
    roots = [
        ROOT / "models/cpu_baseline",
        ROOT / "outputs/cpu_baseline/predictions",
        ROOT / "outputs/cpu_baseline/submission",
    ]
    files = sorted(path for directory in roots for path in directory.glob("*") if path.is_file())
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in files}


def raw_hashes(config: dict[str, Any]) -> dict[str, str]:
    return {relative: sha256_file(ROOT / relative) for relative in config["expected_raw_sha256"]}


def verify_fixed_cv_evidence(config: dict[str, Any]) -> dict[str, Any]:
    metrics = pd.read_csv(PHASE_B_METRICS_PATH)
    summaries = metrics[metrics["record_type"].eq("summary")]
    expected = {
        "english_regret": "word_char_logreg_balanced",
        "tulu_regret": "char_logreg_balanced",
        "english_domain": "char_linearsvc_balanced",
        "tulu_domain": "char_linearsvc_balanced",
    }
    evidence: dict[str, Any] = {}
    for task, candidate_id in expected.items():
        rows = summaries[(summaries["task"] == task) & (summaries["candidate_id"] == candidate_id)]
        if len(rows) != 1:
            raise RuntimeError(f"Missing fixed CV evidence for {task}/{candidate_id}")
        row = rows.iloc[0]
        if not bool(row["local_winner"]):
            raise RuntimeError(f"Configured final model is not recorded local winner: {task}/{candidate_id}")
        expected_mean = config["tasks"][task]["phase_b_cv_macro_f1_mean"]
        if not np.isclose(float(row["macro_f1_mean"]), expected_mean, atol=5e-7):
            raise RuntimeError(f"CV evidence changed for {task}: {row['macro_f1_mean']} != {expected_mean}")
        evidence[task] = {
            "candidate_id": candidate_id,
            "macro_f1_mean": float(row["macro_f1_mean"]),
            "macro_f1_std": float(row["macro_f1_std"]),
            "local_winner": True,
        }
    return evidence


def check_inputs(config: dict[str, Any]) -> dict[str, Any]:
    if config["task_order"] != [
        "english_regret",
        "tulu_regret",
        "english_domain",
        "tulu_domain",
        "kannada_regret",
    ]:
        raise RuntimeError("Unexpected Phase B final task order")
    official = validate_all_inputs(ROOT)
    actual_raw = raw_hashes(config)
    if actual_raw != config["expected_raw_sha256"]:
        mismatch = {
            key: {"expected": config["expected_raw_sha256"][key], "actual": actual_raw[key]}
            for key in actual_raw
            if actual_raw[key] != config["expected_raw_sha256"][key]
        }
        raise RuntimeError(f"Raw input hash mismatch: {mismatch}")
    for task in ("english_domain", "tulu_domain", "kannada_regret"):
        task_config = config["tasks"][task]
        for key in ("source_model", "source_metadata"):
            if not (ROOT / task_config[key]).is_file():
                raise FileNotFoundError(f"Missing Phase A source for {task}: {task_config[key]}")
    evidence = verify_fixed_cv_evidence(config)
    payload = {
        "created_utc": utc_now(),
        "python": sys.version,
        "scikit_learn": sklearn.__version__,
        "official_input_validation": official,
        "raw_sha256": actual_raw,
        "phase_a_artifact_sha256_before": phase_a_inventory(),
        "fixed_cv_evidence": evidence,
        "new_search_performed": False,
    }
    atomic_write_json(INPUT_CHECK_PATH, payload)
    return payload


def stratified_smoke_sample(task: str, include_validation: bool, seed: int) -> tuple[pd.Series, ...]:
    texts, labels, _ = load_task_data(ROOT, task, include_kannada_validation=include_validation)
    sample = pd.DataFrame({"text": texts, "label": labels}).groupby(
        "label", group_keys=False
    ).sample(n=10, random_state=seed)
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    for _, group in sample.groupby("label", sort=False):
        train_parts.append(group.iloc[:8])
        validation_parts.append(group.iloc[8:])
    train = pd.concat(train_parts, ignore_index=True)
    validation = pd.concat(validation_parts, ignore_index=True)
    return train["text"], train["label"], validation["text"], validation["label"]


def validator_self_checks() -> dict[str, Any]:
    template = pd.read_csv(
        ROOT / "official/baseline_predictions.csv", dtype={"id": "string"}, keep_default_na=False
    )
    synthetic = pd.DataFrame(
        {
            "id": template["id"].astype(str),
            "Regret": "Action",
            "Domain": "Other domains",
        }
    )
    kannada_ids = set(load_frame(ROOT, "Kannada", "test")["id"].astype(str))
    synthetic.loc[synthetic["id"].isin(kannada_ids), "Domain"] = ""
    result = validate_submission_bytes(
        synthetic.to_csv(index=False, lineterminator="\n").encode("utf-8")
    )
    if not any("one class" in warning for warning in result["warnings"]):
        raise RuntimeError("Single-class validator warning self-check did not fire")

    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    bad_path = SMOKE_DIR / "structurally_invalid_submission.csv"
    synthetic.drop(columns=["Domain"]).to_csv(
        bad_path, index=False, encoding="utf-8", lineterminator="\n"
    )
    command = [
        sys.executable,
        str(ROOT / "scripts/validate_submission.py"),
        "--csv",
        str(bad_path),
    ]
    cli = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if cli.returncode == 0 or "SUBMISSION VALIDATION FAILED" not in cli.stderr:
        raise RuntimeError("Structural-error CLI self-check did not return nonzero")
    return {
        "single_class_warnings": result["warnings"],
        "structural_error_return_code": cli.returncode,
        "structural_error_message": cli.stderr.strip(),
    }


def smoke_test(config: dict[str, Any]) -> dict[str, Any]:
    if not INPUT_CHECK_PATH.is_file():
        raise RuntimeError("Run --check-inputs before --smoke")
    results: list[dict[str, Any]] = []
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    for task in config["task_order"]:
        task_config = config["tasks"][task]
        train_text, train_labels, validation_text, validation_labels = stratified_smoke_sample(
            task, task_config["include_kannada_validation"], config["seed"]
        )
        model = build_phase_b_model(config, task)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(train_text, train_labels)
        feature_step = model.named_steps.get("features", model.named_steps.get("tfidf"))
        transformed = feature_step.transform(validation_text)
        if not sparse.issparse(transformed):
            raise RuntimeError(f"{task}: smoke feature matrix became dense")
        prediction = model.predict(validation_text)
        invalid = sorted(set(prediction) - set(TASKS[task]["labels"]))
        if invalid:
            raise RuntimeError(f"{task}: invalid smoke predictions {invalid}")
        path = SMOKE_DIR / f"{task}.joblib"
        joblib.dump(model, path)
        loaded = joblib.load(path)
        loaded_prediction = loaded.predict(validation_text)
        if not np.array_equal(prediction, loaded_prediction):
            raise RuntimeError(f"{task}: smoke predictions changed after save/load")
        results.append(
            {
                "task": task,
                "action": task_config["action"],
                "n_features": model_feature_count(model),
                "sparse_matrix": True,
                "prediction_rows": len(prediction),
                "labels_legal": True,
                "save_load_match": True,
                "warnings": [f"{item.category.__name__}: {item.message}" for item in caught],
            }
        )
        logging.info("Smoke passed %s", task)
        del model, loaded, transformed
        gc.collect()
    payload = {
        "created_utc": utc_now(),
        "passed": True,
        "tasks": results,
        "validator_self_checks": validator_self_checks(),
    }
    atomic_write_json(SMOKE_PATH, payload)
    return payload


def assert_prerequisites() -> None:
    if not INPUT_CHECK_PATH.is_file() or not SMOKE_PATH.is_file():
        raise RuntimeError("Successful input check and smoke artifacts are required")
    if not load_json(SMOKE_PATH).get("passed"):
        raise RuntimeError("Smoke test did not pass")


def finalize_task(config: dict[str, Any], task: str) -> dict[str, Any]:
    assert_prerequisites()
    task_config = config["tasks"][task]
    texts, labels, ids = load_task_data(
        ROOT, task, include_kannada_validation=task_config["include_kannada_validation"]
    )
    if len(texts) != task_config["training_rows"]:
        raise RuntimeError(f"{task}: training rows changed: {len(texts)}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{task}.joblib"
    metadata_path = MODEL_DIR / f"{task}.json"
    started = time.perf_counter()
    fit_warnings: list[str] = []

    if task_config["action"] == "train":
        model = build_phase_b_model(config, task)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(texts, labels)
        fit_warnings = [f"{item.category.__name__}: {item.message}" for item in caught]
        joblib.dump(model, model_path)
        source_model_sha256 = None
    else:
        source = ROOT / task_config["source_model"]
        shutil.copy2(source, model_path)
        source_model_sha256 = sha256_file(source)
        if sha256_file(model_path) != source_model_sha256:
            raise RuntimeError(f"{task}: copied model hash mismatch")
        model = joblib.load(model_path)

    fit_seconds = time.perf_counter() - started
    verification_texts = texts.iloc[: min(64, len(texts))]
    prediction = model.predict(verification_texts)
    invalid = sorted(set(prediction) - set(TASKS[task]["labels"]))
    if invalid:
        raise RuntimeError(f"{task}: invalid verification predictions {invalid}")
    loaded = joblib.load(model_path)
    if not np.array_equal(prediction, loaded.predict(verification_texts)):
        raise RuntimeError(f"{task}: prediction changed after final model save/load")
    if task_config["action"] == "copy_phase_a":
        source_loaded = joblib.load(ROOT / task_config["source_model"])
        if not np.array_equal(prediction, source_loaded.predict(verification_texts)):
            raise RuntimeError(f"{task}: copied model behavior differs from Phase A source")
        del source_loaded

    metadata = {
        "created_utc": utc_now(),
        "task": task,
        "language": TASKS[task]["language"],
        "target": TASKS[task]["target"],
        "labels": list(TASKS[task]["labels"]),
        "action": task_config["action"],
        "training_rows": len(texts),
        "training_id_sha256": sha256_ids(ids),
        "includes_kannada_official_validation": task_config["include_kannada_validation"],
        "feature": task_config["feature"],
        "classifier": task_config["classifier"],
        "n_features": model_feature_count(model),
        "fit_seconds": fit_seconds,
        "fit_warnings": fit_warnings,
        "save_load_match": True,
        "model_sha256": sha256_file(model_path),
        "source_model_sha256": source_model_sha256,
        "config": task_config,
    }
    atomic_write_json(metadata_path, metadata)
    logging.info(
        "Finalized %s action=%s rows=%d features=%d model=%s",
        task,
        task_config["action"],
        len(texts),
        metadata["n_features"],
        model_path,
    )
    del model, loaded
    gc.collect()
    return metadata


def prediction_distribution(series: pd.Series, labels: tuple[str, ...]) -> dict[str, int]:
    counts = series.value_counts().reindex(labels, fill_value=0)
    return {label: int(counts[label]) for label in labels}


def compare_with_phase_a(task: str, current: pd.DataFrame) -> dict[str, Any]:
    phase_a_path = ROOT / f"outputs/cpu_baseline/predictions/{task}.csv"
    phase_a = pd.read_csv(phase_a_path, dtype={"id": "string"}, keep_default_na=False)
    if tuple(phase_a.columns) != ("id", "prediction"):
        raise RuntimeError(f"Invalid Phase A prediction columns: {phase_a_path}")
    if phase_a["id"].astype(str).tolist() != current["id"].astype(str).tolist():
        raise RuntimeError(f"{task}: Phase A/B IDs differ")
    old = phase_a["prediction"].astype(str)
    new = current["prediction"].astype(str)
    labels = TASKS[task]["labels"]
    old_distribution = prediction_distribution(old, labels)
    new_distribution = prediction_distribution(new, labels)
    changed = int((old != new).sum())
    transitions = (
        pd.DataFrame({"old": old, "new": new})
        .value_counts()
        .sort_index()
        .to_dict()
    )
    return {
        "changed_predictions": changed,
        "total_predictions": len(current),
        "change_rate": changed / len(current),
        "phase_a_distribution": old_distribution,
        "phase_b_distribution": new_distribution,
        "distribution_delta": {
            label: new_distribution[label] - old_distribution[label] for label in labels
        },
        "transitions": {f"{old_label} -> {new_label}": int(count) for (old_label, new_label), count in transitions.items()},
        "phase_a_prediction_sha256": sha256_file(phase_a_path),
    }


def predict_all(config: dict[str, Any]) -> dict[str, Any]:
    assert_prerequisites()
    missing = [task for task in config["task_order"] if not (MODEL_DIR / f"{task}.joblib").is_file()]
    if missing:
        raise RuntimeError(f"Missing final models: {missing}")
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    distributions: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    warnings_list: list[str] = []
    for task in config["task_order"]:
        spec = TASKS[task]
        test = load_frame(ROOT, spec["language"], "test")
        texts = build_text(test, spec["language"])
        model = joblib.load(MODEL_DIR / f"{task}.joblib")
        prediction = model.predict(texts)
        invalid = sorted(set(prediction) - set(spec["labels"]))
        if invalid:
            raise RuntimeError(f"{task}: invalid test predictions {invalid}")
        output = pd.DataFrame({"id": test["id"].astype(str), "prediction": prediction})
        if output["id"].tolist() != test["id"].astype(str).tolist():
            raise RuntimeError(f"{task}: test ID order changed")
        if output["id"].duplicated().any():
            raise RuntimeError(f"{task}: duplicate prediction IDs")
        if output["prediction"].nunique() == 1:
            warnings_list.append(f"{task}: predictions contain one class")
        path = PREDICTION_DIR / f"{task}.csv"
        output.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
        reloaded = pd.read_csv(path, dtype={"id": "string"}, keep_default_na=False)
        if not output.astype(str).equals(reloaded.astype(str)):
            raise RuntimeError(f"{task}: UTF-8 prediction roundtrip mismatch")
        distributions[task] = prediction_distribution(output["prediction"], spec["labels"])
        comparisons[task] = compare_with_phase_a(task, output)
        expected_same = config["tasks"][task]["action"] == "copy_phase_a"
        if expected_same and comparisons[task]["changed_predictions"] != 0:
            raise RuntimeError(f"{task}: copied Phase A task changed predictions")
        if not expected_same and comparisons[task]["changed_predictions"] == 0:
            raise RuntimeError(f"{task}: replacement model produced no prediction changes")
        logging.info(
            "Predicted %s rows=%d changed_vs_phase_a=%d distribution=%s",
            task,
            len(output),
            comparisons[task]["changed_predictions"],
            distributions[task],
        )
        del model
        gc.collect()
    payload = {
        "created_utc": utc_now(),
        "distributions": distributions,
        "phase_a_comparisons": comparisons,
        "single_class_warnings": warnings_list,
        "descriptive_only_no_tuning": True,
    }
    atomic_write_json(PREDICTION_SUMMARY_PATH, payload)
    return payload


def load_prediction(task: str) -> pd.DataFrame:
    path = PREDICTION_DIR / f"{task}.csv"
    frame = pd.read_csv(path, dtype={"id": "string"}, keep_default_na=False)
    if tuple(frame.columns) != ("id", "prediction"):
        raise RuntimeError(f"{path}: expected columns id,prediction")
    test = load_frame(ROOT, TASKS[task]["language"], "test")
    if frame["id"].astype(str).tolist() != test["id"].astype(str).tolist():
        raise RuntimeError(f"{path}: IDs do not exactly match test order")
    if frame["id"].duplicated().any():
        raise RuntimeError(f"{path}: duplicate IDs")
    invalid = sorted(set(frame["prediction"]) - set(TASKS[task]["labels"]))
    if invalid:
        raise RuntimeError(f"{path}: invalid labels {invalid}")
    return frame


def make_submission(config: dict[str, Any]) -> dict[str, Any]:
    if not PREDICTION_SUMMARY_PATH.is_file():
        raise RuntimeError("Run --predict before --submission")
    predictions = {task: load_prediction(task) for task in config["task_order"]}
    template_path = ROOT / "official/baseline_predictions.csv"
    template = pd.read_csv(template_path, dtype={"id": "string"}, keep_default_na=False)
    if tuple(template.columns) != ("id", "Regret", "Domain"):
        raise RuntimeError("Official template columns changed")
    submission = pd.DataFrame({"id": template["id"].astype(str), "Regret": "", "Domain": ""})
    for task, frame in predictions.items():
        target = TASKS[task]["target"]
        mapping = dict(zip(frame["id"].astype(str), frame["prediction"], strict=True))
        mask = submission["id"].isin(mapping)
        if int(mask.sum()) != len(frame):
            raise RuntimeError(f"{task}: prediction IDs do not map one-to-one into official template")
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
    reloaded = pd.read_csv(csv_path, dtype={"id": "string"}, keep_default_na=False)
    if tuple(reloaded.columns) != ("id", "Regret", "Domain") or any(
        column.startswith("Unnamed:") for column in reloaded.columns
    ):
        raise RuntimeError("CSV columns or Pandas index-column check failed")
    if not submission.astype(str).equals(reloaded.astype(str)):
        raise RuntimeError("CSV UTF-8 roundtrip data mismatch")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname="submission.csv")
    with zipfile.ZipFile(zip_path, "r") as archive:
        if archive.namelist() != ["submission.csv"]:
            raise RuntimeError(f"Unexpected ZIP members: {archive.namelist()}")
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC test failed")
        zip_csv_bytes = archive.read("submission.csv")
    if zip_csv_bytes != csv_path.read_bytes():
        raise RuntimeError("ZIP member bytes differ from validated submission.csv")
    zip_result = validate_submission_zip(zip_path)
    zip_frame = pd.read_csv(io.BytesIO(zip_csv_bytes), dtype={"id": "string"}, keep_default_na=False)
    if not reloaded.astype(str).equals(zip_frame.astype(str)):
        raise RuntimeError("ZIP-extracted CSV differs from source CSV")

    payload = {
        "created_utc": utc_now(),
        "template": "official/baseline_predictions.csv",
        "template_sha256": sha256_file(template_path),
        "csv_path": csv_path.relative_to(ROOT).as_posix(),
        "zip_path": zip_path.relative_to(ROOT).as_posix(),
        "csv_sha256": sha256_file(csv_path),
        "zip_sha256": sha256_file(zip_path),
        "csv_validation": csv_result,
        "zip_validation": zip_result,
        "strict_columns": True,
        "no_pandas_index_column": True,
        "utf8_roundtrip": True,
        "zip_member_bytes_match_csv": True,
        "zip_crc_passed": True,
    }
    atomic_write_json(SUBMISSION_VALIDATION_PATH, payload)
    return payload


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def assert_phase_a_unchanged() -> None:
    input_check = load_json(INPUT_CHECK_PATH)
    current = phase_a_inventory()
    if current != input_check["phase_a_artifact_sha256_before"]:
        raise RuntimeError("Phase A model/prediction/submission artifacts changed during finalization")


def write_report(config: dict[str, Any]) -> dict[str, Any]:
    required = [INPUT_CHECK_PATH, SMOKE_PATH, PREDICTION_SUMMARY_PATH, SUBMISSION_VALIDATION_PATH]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing finalization state: {missing}")
    assert_phase_a_unchanged()
    actual_raw = raw_hashes(config)
    if actual_raw != config["expected_raw_sha256"]:
        raise RuntimeError("data/raw hashes changed before report generation")
    prediction_summary = load_json(PREDICTION_SUMMARY_PATH)
    validation = load_json(SUBMISSION_VALIDATION_PATH)
    metadata = {task: load_json(MODEL_DIR / f"{task}.json") for task in config["task_order"]}

    model_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    for task in config["task_order"]:
        task_config = config["tasks"][task]
        classifier = config["classifiers"][task_config["classifier"]]
        classifier_text = classifier["type"]
        if classifier["type"] == "LogisticRegression":
            classifier_text = "OneVsRest LogisticRegression(liblinear)"
        model_rows.append(
            {
                "task": task,
                "action": task_config["action"],
                "feature": task_config["feature"],
                "classifier": classifier_text,
                "training_rows": metadata[task]["training_rows"],
                "n_features": metadata[task]["n_features"],
                "model_sha256": metadata[task]["model_sha256"],
            }
        )
        for label, count in prediction_summary["distributions"][task].items():
            comparison = prediction_summary["phase_a_comparisons"][task]
            distribution_rows.append(
                {
                    "task": task,
                    "label": label,
                    "phase_a": comparison["phase_a_distribution"][label],
                    "phase_b": count,
                    "delta": comparison["distribution_delta"][label],
                }
            )
        comparison = prediction_summary["phase_a_comparisons"][task]
        difference_rows.append(
            {
                "task": task,
                "changed": comparison["changed_predictions"],
                "total": comparison["total_predictions"],
                "change_rate": comparison["change_rate"],
                "single_class": len([count for count in prediction_summary["distributions"][task].values() if count > 0]) == 1,
            }
        )

    file_rows: list[dict[str, Any]] = []
    artifact_paths = [
        *(MODEL_DIR / f"{task}.joblib" for task in config["task_order"]),
        *(PREDICTION_DIR / f"{task}.csv" for task in config["task_order"]),
        SUBMISSION_DIR / "submission.csv",
        SUBMISSION_DIR / "submission.zip",
    ]
    for path in artifact_paths:
        file_rows.append(
            {
                "file": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    report = f"""# Phase B Final Mixed-Model Submission

This finalization uses the fixed Phase B five-fold winners without any additional model search, cross-validation, or test-driven tuning. Test data were used only for final prediction and structural validation. No Codabench upload was performed.

## Final task models

{markdown_table(model_rows, ['task', 'action', 'feature', 'classifier', 'training_rows', 'n_features', 'model_sha256'])}

Only English Regret and Tulu Regret were replaced because their selected Phase B candidates improved five-fold Macro F1 by `+0.026130` and `+0.013813`, respectively. English Domain and Tulu Domain retained Phase A because the Phase A configuration remained the local five-fold winner. Kannada Regret also retains the Phase A train-plus-official-validation model and does not use a test-text lookup.

## Kannada caveat

Kannada final training uses 4,793 training rows plus 685 official validation rows (5,478 total). Exact normalized-text overlap across Kannada splits can inflate overlap-sensitive results. The copied Phase A model applies its learned TF-IDF/LinearSVC pipeline normally; no manual label copying or test duplicate lookup is used.

## Prediction differences from Phase A

These comparisons are descriptive only and were not used to alter any model or parameter.

{markdown_table(difference_rows, ['task', 'changed', 'total', 'change_rate', 'single_class'])}

## Prediction label distributions

{markdown_table(distribution_rows, ['task', 'label', 'phase_a', 'phase_b', 'delta'])}

No single-class degeneration warning was produced by the actual Phase B predictions: `{prediction_summary['single_class_warnings']}`.

## Submission validation

- Template: `official/baseline_predictions.csv` (the only submission template used).
- CSV path: `{validation['csv_path']}`.
- ZIP path: `{validation['zip_path']}`.
- CSV validation passed: `{validation['csv_validation']['passed']}`; rows: `{validation['csv_validation']['rows']}`; warnings: `{validation['csv_validation']['warnings']}`.
- ZIP validation passed: `{validation['zip_validation']['passed']}`; members: `{validation['zip_validation']['zip_members']}`; warnings: `{validation['zip_validation']['warnings']}`.
- Columns are exactly `id,Regret,Domain`; IDs are unique and match template set/order; all labels are legal; English/Tulu Domain is filled; Kannada Domain is empty.
- UTF-8 roundtrip, absence of a Pandas index column, ZIP root-only member rule, ZIP CRC, extracted-byte equality, and extracted CSV revalidation all passed.
- Validator self-test confirmed that single-class predictions create warnings and structural errors return a nonzero CLI status.

## Artifact hashes

{markdown_table(file_rows, ['file', 'bytes', 'sha256'])}

## Integrity and scope

- All seven `data/raw` SHA-256 values match `reports/data_audit.md`.
- Phase A models, predictions, and submission hashes are unchanged from the pre-run inventory.
- English/Tulu training rows: 2,911 / 2,740. Kannada final training rows: 5,478.
- No final model was selected using the first official test score or Phase B test prediction distribution.
- No Transformer, translation, stemming, stop-word removal, test lookup, dependency installation, commit, push, or upload occurred.
"""
    atomic_write_text(REPORT_PATH, report)
    return {
        "report": REPORT_PATH.relative_to(ROOT).as_posix(),
        "model_count": len(model_rows),
        "prediction_file_count": len(config["task_order"]),
        "artifact_count": len(file_rows),
        "raw_hashes_match": True,
        "phase_a_artifacts_unchanged": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check-inputs", action="store_true")
    actions.add_argument("--smoke", action="store_true")
    actions.add_argument("--task", choices=[
        "english_regret",
        "tulu_regret",
        "english_domain",
        "tulu_domain",
        "kannada_regret",
    ])
    actions.add_argument("--predict", action="store_true")
    actions.add_argument("--submission", action="store_true")
    actions.add_argument("--report", action="store_true")
    args = parser.parse_args()
    mode = (
        "check_inputs"
        if args.check_inputs
        else "smoke"
        if args.smoke
        else f"task_{args.task}"
        if args.task
        else "predict"
        if args.predict
        else "submission"
        if args.submission
        else "report"
    )
    log_path = setup_logging(mode)
    config = load_config()
    logging.info("Phase B final mode=%s config=%s log=%s", mode, CONFIG_PATH, log_path)
    if args.check_inputs:
        result = check_inputs(config)
    elif args.smoke:
        result = smoke_test(config)
    elif args.task:
        result = finalize_task(config, args.task)
    elif args.predict:
        result = predict_all(config)
    elif args.submission:
        result = make_submission(config)
    else:
        result = write_report(config)
    logging.info("Result: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
