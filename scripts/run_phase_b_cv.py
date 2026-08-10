"""Run resumable Phase B five-fold traditional-model experiments.

Only labeled English and Tulu training files are loaded. Official test files are
never read by this script. Candidate workers are executed serially and write an
atomic checkpoint after every completed fold.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import gc
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils import TASKS, load_frame, load_task_data


CONFIG_PATH = ROOT / "configs/phase_b_traditional.json"
OUTPUT_DIR = ROOT / "outputs/phase_b_traditional"
RECORD_DIR = OUTPUT_DIR / "candidate_records"
FOLD_DIR = OUTPUT_DIR / "fold_assignments"
LOG_DIR = ROOT / "logs/phase_b_traditional"
TABLE_DIR = ROOT / "reports/tables"
REPORT_PATH = ROOT / "reports/phase_b_traditional_results.md"
METRICS_PATH = TABLE_DIR / "phase_b_cv_metrics.csv"
PER_CLASS_PATH = TABLE_DIR / "phase_b_per_class.csv"
CONFUSION_PATH = TABLE_DIR / "phase_b_confusion.csv"
RUN_STATE_PATH = OUTPUT_DIR / "run_state.json"
RESOURCE_OBSERVATION_PATH = OUTPUT_DIR / "external_resource_observation.json"
SMOKE_PATH = OUTPUT_DIR / "smoke.json"
INPUT_CHECK_PATH = OUTPUT_DIR / "input_check.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    temporary.replace(path)


def setup_logging(mode: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = LOG_DIR / f"{stamp}_{mode}_{os.getpid()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return path


def record_path(task: str, candidate_id: str) -> Path:
    return RECORD_DIR / f"{task}__{candidate_id}.json"


def candidate_sequence(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (task, candidate)
        for task in config["task_order"]
        for candidate in config["tasks"][task]["candidates"]
    ]


def find_candidate(config: dict[str, Any], task: str, candidate_id: str) -> dict[str, Any]:
    matches = [item for item in config["tasks"][task]["candidates"] if item["id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one candidate {task}/{candidate_id}, found {len(matches)}")
    return matches[0]


def tfidf_from_config(spec: dict[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=spec["analyzer"],
        ngram_range=tuple(spec["ngram_range"]),
        min_df=spec["min_df"],
        sublinear_tf=spec["sublinear_tf"],
        max_features=spec["max_features"],
        dtype=np.float32,
    )


def build_pipeline(
    config: dict[str, Any], candidate: dict[str, Any], use_reduced_union: bool = False
) -> tuple[Pipeline, str, str]:
    feature_name = candidate["feature"]
    if feature_name == "word_char":
        profile = "word_char_reduced" if use_reduced_union else "word_char"
        feature_config = config["features"][profile]
        features: Any = FeatureUnion(
            [
                ("char", tfidf_from_config(feature_config["char"])),
                ("word", tfidf_from_config(feature_config["word"])),
            ],
            n_jobs=1,
        )
    else:
        profile = feature_name
        features = tfidf_from_config(config["features"][feature_name])

    class_weight = candidate["class_weight"]
    if candidate["model"] == "LinearSVC":
        classifier_config = config["classifiers"]["LinearSVC"]
        classifier: Any = LinearSVC(
            C=classifier_config["C"],
            class_weight=class_weight,
            random_state=classifier_config["random_state"],
        )
        solver = "LinearSVC"
    else:
        classifier_config = config["classifiers"]["LogisticRegression"]
        base_classifier = LogisticRegression(
            solver=classifier_config["solver"],
            C=classifier_config["C"],
            max_iter=classifier_config["max_iter"],
            class_weight=class_weight,
            random_state=classifier_config["random_state"],
        )
        classifier = OneVsRestClassifier(base_classifier, n_jobs=1)
        solver = f"{classifier_config['solver']} + OneVsRestClassifier"
    return Pipeline([("features", features), ("classifier", classifier)]), profile, solver


def feature_count(model: Pipeline) -> int:
    features = model.named_steps["features"]
    if isinstance(features, FeatureUnion):
        return int(sum(len(transformer.vocabulary_) for _, transformer in features.transformer_list))
    return int(len(features.vocabulary_))


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def memory_snapshot() -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "system_memory_load_percent": None,
        "system_available_gib": None,
        "process_working_set_mib": None,
        "process_peak_working_set_mib": None,
    }
    if os.name != "nt":
        return result
    try:
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
        kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError()
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        result.update(
            {
                "system_memory_load_percent": float(status.dwMemoryLoad),
                "system_available_gib": float(status.ullAvailPhys / 1024**3),
                "process_working_set_mib": float(counters.WorkingSetSize / 1024**2),
                "process_peak_working_set_mib": float(counters.PeakWorkingSetSize / 1024**2),
            }
        )
    except Exception as error:  # pragma: no cover - platform-specific telemetry fallback
        logging.warning("Memory telemetry unavailable: %s", error)
    return result


def should_reduce_union(config: dict[str, Any], candidate: dict[str, Any]) -> tuple[bool, dict[str, float | None]]:
    snapshot = memory_snapshot()
    if candidate["feature"] != "word_char":
        return False, snapshot
    guards = config["resource_guards"]
    load = snapshot["system_memory_load_percent"]
    available = snapshot["system_available_gib"]
    reduce = bool(
        (load is not None and load >= guards["memory_load_percent_for_reduced_union"])
        or (available is not None and available < guards["minimum_available_memory_gib"])
    )
    return reduce, snapshot


def metric_rows(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str], base: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    metric = {
        **base,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = [
        {
            **base,
            "label": label,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    ]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    confusion = [
        {
            **base,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "count": int(matrix[row_index, column_index]),
        }
        for row_index, true_label in enumerate(labels)
        for column_index, predicted_label in enumerate(labels)
    ]
    return metric, per_class, confusion


def make_splitter(config: dict[str, Any]) -> StratifiedKFold:
    cv = config["cv"]
    return StratifiedKFold(
        n_splits=cv["n_splits"], shuffle=cv["shuffle"], random_state=cv["random_state"]
    )


def validate_config(config: dict[str, Any]) -> None:
    sequence = candidate_sequence(config)
    if len(sequence) != 25:
        raise ValueError(f"Phase B matrix must contain 25 candidates, found {len(sequence)}")
    expected_counts = {"english_domain": 12, "tulu_regret": 8, "english_regret": 3, "tulu_domain": 2}
    actual_counts = {task: len(config["tasks"][task]["candidates"]) for task in config["task_order"]}
    if actual_counts != expected_counts:
        raise ValueError(f"Candidate counts differ: {actual_counts}")
    for task, candidate in sequence:
        if task not in TASKS or TASKS[task]["language"] not in {"English", "Tulu"}:
            raise ValueError(f"Phase B task is outside allowed scope: {task}")
        if candidate["feature"] not in {"char", "word", "word_char"}:
            raise ValueError(f"Unsupported feature: {candidate}")
        if candidate["model"] not in {"LinearSVC", "LogisticRegression"}:
            raise ValueError(f"Unsupported classifier: {candidate}")
        if candidate["class_weight"] not in {None, "balanced"}:
            raise ValueError(f"Unsupported class weight: {candidate}")
    for task in config["task_order"]:
        ids = [item["id"] for item in config["tasks"][task]["candidates"]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate candidate ID in {task}")


def write_fold_assignments(config: dict[str, Any], task: str) -> Path:
    _, labels, ids = load_task_data(ROOT, task)
    assignment = np.zeros(len(labels), dtype=np.int16)
    splitter = make_splitter(config)
    for fold, (_, validation_indices) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
        assignment[validation_indices] = fold
    if set(assignment.tolist()) != set(range(1, config["cv"]["n_splits"] + 1)):
        raise RuntimeError(f"Invalid fold assignments for {task}")
    frame = pd.DataFrame(
        {
            "task": task,
            "id_sha256": [hashlib.sha256(value.encode("utf-8")).hexdigest() for value in ids],
            "validation_fold": assignment,
        }
    )
    path = FOLD_DIR / f"{task}.csv"
    atomic_write_csv(path, frame)
    return path


def check_inputs(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    files: dict[str, Any] = {}
    # Deliberately inspect only labeled train files; no test file is opened here.
    for language in ("English", "Tulu"):
        frame = load_frame(ROOT, language, "train")
        relative = Path("data/raw/training_data") / f"{language.lower()}_train_data.csv"
        path = ROOT / relative
        files[language] = {
            "relative_path": relative.as_posix(),
            "rows": len(frame),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    fold_files = {task: str(write_fold_assignments(config, task).relative_to(ROOT)) for task in config["task_order"]}
    payload = {
        "created_utc": utc_now(),
        "python": sys.version,
        "scikit_learn": sklearn.__version__,
        "config_sha256": config_sha256(config),
        "candidate_count": len(candidate_sequence(config)),
        "train_files": files,
        "fold_assignment_files": fold_files,
        "test_files_read": False,
    }
    atomic_write_json(INPUT_CHECK_PATH, payload)
    return payload


def stratified_smoke_sample(task: str, seed: int) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    texts, labels, _ = load_task_data(ROOT, task)
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


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    if not INPUT_CHECK_PATH.is_file():
        raise RuntimeError("Run --check-inputs before --smoke")
    results: list[dict[str, Any]] = []
    for task, candidate in candidate_sequence(config):
        train_text, train_labels, validation_text, validation_labels = stratified_smoke_sample(
            task, config["seed"]
        )
        model, profile, solver = build_pipeline(config, candidate)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(train_text, train_labels)
        transformed = model.named_steps["features"].transform(validation_text)
        if not sparse.issparse(transformed):
            raise RuntimeError(f"{task}/{candidate['id']}: feature matrix became dense")
        prediction = model.predict(validation_text)
        if len(prediction) != len(validation_labels):
            raise RuntimeError(f"{task}/{candidate['id']}: smoke prediction length mismatch")
        results.append(
            {
                "task": task,
                "candidate_id": candidate["id"],
                "feature_profile": profile,
                "solver": solver,
                "n_features": feature_count(model),
                "sparse_matrix": True,
                "warnings": [f"{item.category.__name__}: {item.message}" for item in caught],
                "passed": True,
            }
        )
        logging.info("Smoke passed %s/%s", task, candidate["id"])
        del model, transformed
        gc.collect()
    payload = {
        "created_utc": utc_now(),
        "python": sys.version,
        "scikit_learn": sklearn.__version__,
        "config_sha256": config_sha256(config),
        "candidate_count": len(results),
        "passed": all(row["passed"] for row in results),
        "results": results,
        "test_files_read": False,
    }
    atomic_write_json(SMOKE_PATH, payload)
    return payload


def base_fields(task: str, candidate: dict[str, Any], solver: str, profile: str) -> dict[str, Any]:
    spec = TASKS[task]
    return {
        "task": task,
        "language": spec["language"],
        "target": spec["target"],
        "candidate_id": candidate["id"],
        "feature_type": candidate["feature"],
        "model_type": candidate["model"],
        "class_weight": candidate["class_weight"] if candidate["class_weight"] is not None else "None",
        "solver": solver,
        "feature_profile": profile,
    }


def candidate_worker(config: dict[str, Any], task: str, candidate_id: str) -> int:
    candidate = find_candidate(config, task, candidate_id)
    path = record_path(task, candidate_id)
    started_utc = utc_now()
    started = time.perf_counter()
    reduced, initial_memory = should_reduce_union(config, candidate)
    model_probe, profile, solver = build_pipeline(config, candidate, reduced)
    del model_probe
    base = base_fields(task, candidate, solver, profile)
    record: dict[str, Any] = {
        "created_utc": started_utc,
        "updated_utc": started_utc,
        "python": sys.version,
        "scikit_learn": sklearn.__version__,
        "config_sha256": config_sha256(config),
        "status": "running",
        "error_summary": "",
        "candidate": candidate,
        "base": base,
        "memory_reduction_triggered": reduced,
        "initial_memory": initial_memory,
        "peak_working_set_mib": initial_memory["process_peak_working_set_mib"],
        "fold_metrics": [],
        "per_class": [],
        "confusion": [],
        "warnings": [],
        "completed_folds": 0,
        "wall_seconds": 0.0,
        "test_files_read": False,
    }
    atomic_write_json(path, record)
    try:
        texts, labels, _ = load_task_data(ROOT, task)
        label_order = TASKS[task]["labels"]
        splits = list(make_splitter(config).split(texts, labels))
        oof_prediction = np.full(len(labels), None, dtype=object)
        for fold, (train_indices, validation_indices) in enumerate(splits, start=1):
            if set(labels.iloc[validation_indices]) != set(label_order):
                raise RuntimeError(f"Fold {fold} does not contain every label")
            model, fold_profile, fold_solver = build_pipeline(config, candidate, reduced)
            if fold_profile != profile or fold_solver != solver:
                raise RuntimeError("Candidate configuration changed between folds")
            fit_started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(texts.iloc[train_indices], labels.iloc[train_indices])
            fit_seconds = time.perf_counter() - fit_started
            predict_started = time.perf_counter()
            prediction = model.predict(texts.iloc[validation_indices])
            predict_seconds = time.perf_counter() - predict_started
            oof_prediction[validation_indices] = prediction
            n_features = feature_count(model)
            fold_base = {
                **base,
                "record_type": "fold",
                "fold": fold,
                "n_train": len(train_indices),
                "n_validation": len(validation_indices),
            }
            metric, per_class, confusion = metric_rows(
                labels.iloc[validation_indices], prediction, label_order, fold_base
            )
            metric.update(
                {
                    "fit_seconds": float(fit_seconds),
                    "predict_seconds": float(predict_seconds),
                    "n_features": n_features,
                }
            )
            record["fold_metrics"].append(metric)
            record["per_class"].extend(per_class)
            record["confusion"].extend(confusion)
            record["warnings"].extend(
                [
                    {
                        "fold": fold,
                        "category": item.category.__name__,
                        "message": str(item.message),
                        "is_convergence_warning": issubclass(item.category, ConvergenceWarning),
                    }
                    for item in caught
                ]
            )
            memory = memory_snapshot()
            peaks = [record.get("peak_working_set_mib"), memory.get("process_peak_working_set_mib")]
            available_peaks = [value for value in peaks if value is not None]
            record["peak_working_set_mib"] = max(available_peaks) if available_peaks else None
            record["completed_folds"] = fold
            record["wall_seconds"] = float(time.perf_counter() - started)
            record["updated_utc"] = utc_now()
            atomic_write_json(path, record)
            logging.info(
                "%s/%s fold=%d macro_f1=%.6f fit=%.2fs predict=%.2fs features=%d",
                task,
                candidate_id,
                fold,
                metric["macro_f1"],
                fit_seconds,
                predict_seconds,
                n_features,
            )
            del model
            gc.collect()

        if any(value is None for value in oof_prediction):
            raise RuntimeError("OOF predictions are incomplete")
        oof_base = {
            **base,
            "record_type": "oof_aggregate",
            "fold": "",
            "n_train": len(labels),
            "n_validation": len(labels),
        }
        _, oof_per_class, oof_confusion = metric_rows(labels, oof_prediction, label_order, oof_base)
        record["per_class"].extend(oof_per_class)
        record["confusion"].extend(oof_confusion)
        record["status"] = "complete"
        record["completed_folds"] = len(splits)
        record["wall_seconds"] = float(time.perf_counter() - started)
        record["updated_utc"] = utc_now()
        atomic_write_json(path, record)
        return 0
    except Exception as error:
        record["status"] = "failed"
        record["error_summary"] = f"{type(error).__name__}: {error}"[:1000]
        record["traceback"] = traceback.format_exc(limit=20)
        record["wall_seconds"] = float(time.perf_counter() - started)
        record["updated_utc"] = utc_now()
        atomic_write_json(path, record)
        logging.exception("Candidate failed %s/%s", task, candidate_id)
        return 1


def read_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def mark_timeout(path: Path, reason: str, elapsed: float) -> None:
    record = read_record(path) or {}
    record.update(
        {
            "status": "timeout",
            "error_summary": reason,
            "updated_utc": utc_now(),
            "wall_seconds": float(elapsed),
            "test_files_read": False,
        }
    )
    atomic_write_json(path, record)


def append_run_session(payload: dict[str, Any]) -> None:
    state = read_record(RUN_STATE_PATH) or {"sessions": []}
    state["sessions"].append(payload)
    state["updated_utc"] = utc_now()
    atomic_write_json(RUN_STATE_PATH, state)


def run_all(config: dict[str, Any]) -> dict[str, Any]:
    smoke = read_record(SMOKE_PATH)
    if not smoke or not smoke.get("passed") or smoke.get("config_sha256") != config_sha256(config):
        raise RuntimeError("A successful smoke test for the current config is required before --run")
    guards = config["resource_guards"]
    terminal = {"complete", "failed", "timeout"}
    existing_records = [
        read_record(record_path(task, candidate["id"])) for task, candidate in candidate_sequence(config)
    ]
    budget_used = sum(
        float(record.get("wall_seconds", 0.0))
        for record in existing_records
        if record and record.get("status") in terminal
    )
    session_started_utc = utc_now()
    session_started = time.perf_counter()
    executed = 0
    skipped = 0
    stop_reason = "all candidates reached a terminal status"
    for task, candidate in candidate_sequence(config):
        path = record_path(task, candidate["id"])
        record = read_record(path)
        if record and record.get("status") in terminal and record.get("config_sha256") == config_sha256(config):
            skipped += 1
            logging.info("Resume skip %s/%s status=%s", task, candidate["id"], record["status"])
            continue
        elapsed_this_session = time.perf_counter() - session_started
        remaining = guards["total_experiment_seconds"] - budget_used - elapsed_this_session
        if remaining <= 0:
            stop_reason = "150-minute cumulative experiment limit reached"
            logging.warning(stop_reason)
            break
        timeout = min(float(guards["candidate_timeout_seconds"]), float(remaining))
        command = [sys.executable, str(Path(__file__).resolve()), "--candidate", task, candidate["id"]]
        logging.info("Start %s/%s timeout=%.1fs", task, candidate["id"], timeout)
        candidate_started = time.perf_counter()
        try:
            result = subprocess.run(command, cwd=ROOT, check=False, timeout=timeout)
            executed += 1
            if result.returncode != 0:
                logging.error("Candidate exited nonzero %s/%s code=%d", task, candidate["id"], result.returncode)
        except subprocess.TimeoutExpired:
            executed += 1
            candidate_elapsed = time.perf_counter() - candidate_started
            reason = (
                "candidate exceeded 12-minute limit"
                if timeout >= guards["candidate_timeout_seconds"] - 1
                else "total 150-minute experiment limit reached during candidate"
            )
            mark_timeout(path, reason, candidate_elapsed)
            logging.error("Stopped %s/%s: %s", task, candidate["id"], reason)
            if timeout < guards["candidate_timeout_seconds"] - 1:
                stop_reason = reason
                break
    session = {
        "started_utc": session_started_utc,
        "finished_utc": utc_now(),
        "session_wall_seconds": float(time.perf_counter() - session_started),
        "budget_used_before_session_seconds": float(budget_used),
        "executed_candidates": executed,
        "resume_skipped_candidates": skipped,
        "stop_reason": stop_reason,
    }
    append_run_session(session)
    return session


def load_all_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task, candidate in candidate_sequence(config):
        record = read_record(record_path(task, candidate["id"]))
        if record is None:
            _, profile, solver = build_pipeline(config, candidate)
            record = {
                "status": "not_run",
                "error_summary": "not executed",
                "base": base_fields(task, candidate, solver, profile),
                "candidate": candidate,
                "fold_metrics": [],
                "per_class": [],
                "confusion": [],
                "warnings": [],
                "completed_folds": 0,
                "wall_seconds": 0.0,
                "peak_working_set_mib": None,
            }
        records.append(record)
    return records


def candidate_summary(record: dict[str, Any], control_id: str) -> dict[str, Any]:
    base = record["base"]
    fold_frame = pd.DataFrame(record.get("fold_metrics", []))
    row: dict[str, Any] = {
        **base,
        "record_type": "summary",
        "fold": "",
        "status": record["status"],
        "error_summary": record.get("error_summary", ""),
        "phase_a_control": base["candidate_id"] == control_id,
        "local_winner": False,
        "stable_alternative": False,
        "completed_folds": record.get("completed_folds", 0),
        "wall_seconds": record.get("wall_seconds", 0.0),
        "peak_working_set_mib": record.get("peak_working_set_mib"),
        "memory_reduction_triggered": record.get("memory_reduction_triggered", False),
        "warning_count": len(record.get("warnings", [])),
        "convergence_warning_count": sum(
            bool(item.get("is_convergence_warning")) for item in record.get("warnings", [])
        ),
    }
    if not fold_frame.empty:
        for metric in ("accuracy", "macro_f1", "weighted_f1"):
            row[f"{metric}_mean"] = float(fold_frame[metric].mean())
            row[f"{metric}_std"] = float(fold_frame[metric].std(ddof=1)) if len(fold_frame) > 1 else np.nan
        row["total_fit_seconds"] = float(fold_frame["fit_seconds"].sum())
        row["total_predict_seconds"] = float(fold_frame["predict_seconds"].sum())
        row["n_features_mean"] = float(fold_frame["n_features"].mean())
    if base["target"] == "Domain":
        class_frame = pd.DataFrame(record.get("per_class", []))
        minority_labels = {"Health", "Education", "Career and Finance"}
        minority_rows = (
            class_frame[
                (class_frame["record_type"] == "oof_aggregate")
                & class_frame["label"].isin(minority_labels)
            ]
            if not class_frame.empty
            else class_frame
        )
        if len(minority_rows) == len(minority_labels):
            row["domain_minority_recall_mean"] = float(minority_rows["recall"].mean())
            row["domain_minority_f1_mean"] = float(minority_rows["f1"].mean())
    return row


def minority_scores(record: dict[str, Any], labels: Sequence[str]) -> tuple[float, float]:
    frame = pd.DataFrame(record.get("per_class", []))
    if frame.empty:
        return float("nan"), float("nan")
    rows = frame[(frame["record_type"] == "oof_aggregate") & frame["label"].isin(labels)]
    return float(rows["recall"].mean()), float(rows["f1"].mean())


def simpler_key(row: pd.Series) -> tuple[int, int, float]:
    feature_complexity = {"word": 0, "char": 1, "word_char": 2}
    model_complexity = {"LinearSVC": 0, "LogisticRegression": 1}
    return (
        feature_complexity.get(row["feature_type"], 99),
        model_complexity.get(row["model_type"], 99),
        float(row.get("total_fit_seconds", float("inf"))),
    )


def choose_for_task(
    config: dict[str, Any], task: str, task_summaries: pd.DataFrame, record_map: dict[tuple[str, str], dict[str, Any]]
) -> tuple[str | None, str | None]:
    complete = task_summaries[
        (task_summaries["status"] == "complete") & (task_summaries["completed_folds"] == config["cv"]["n_splits"])
    ].sort_values(["macro_f1_mean", "macro_f1_std"], ascending=[False, True])
    if complete.empty:
        return None, None
    top = complete.iloc[0]
    winner = top
    threshold = config["selection"]["macro_f1_close_threshold"]
    std_threshold = config["selection"]["std_close_threshold"]
    if len(complete) > 1:
        second = complete.iloc[1]
        if float(top["macro_f1_mean"] - second["macro_f1_mean"]) < threshold:
            std_gap = abs(float(top["macro_f1_std"] - second["macro_f1_std"]))
            if std_gap >= std_threshold:
                winner = top if top["macro_f1_std"] < second["macro_f1_std"] else second
            elif TASKS[task]["target"] == "Domain":
                minority = config["selection"]["domain_minority_labels"]
                top_recall, top_f1 = minority_scores(record_map[(task, top["candidate_id"])], minority)
                second_recall, second_f1 = minority_scores(record_map[(task, second["candidate_id"])], minority)
                winner = top if (top_recall, top_f1) >= (second_recall, second_f1) else second
            else:
                winner = min((top, second), key=simpler_key)
    winner_id = str(winner["candidate_id"])
    alternatives = complete[complete["candidate_id"] != winner_id].copy()
    if alternatives.empty:
        return winner_id, None
    window = config["selection"]["stable_alternative_window"]
    near = alternatives[
        alternatives["macro_f1_mean"] >= float(winner["macro_f1_mean"]) - window
    ]
    pool = near if not near.empty else alternatives.head(1)
    stable = pool.sort_values(["macro_f1_std", "macro_f1_mean"], ascending=[True, False]).iloc[0]
    return winner_id, str(stable["candidate_id"])


def add_selection_flags(
    config: dict[str, Any], summaries: pd.DataFrame, records: list[dict[str, Any]]
) -> tuple[pd.DataFrame, dict[str, tuple[str | None, str | None]]]:
    record_map = {(record["base"]["task"], record["base"]["candidate_id"]): record for record in records}
    selections: dict[str, tuple[str | None, str | None]] = {}
    for task in config["task_order"]:
        winner, stable = choose_for_task(config, task, summaries[summaries["task"] == task], record_map)
        selections[task] = (winner, stable)
        summaries.loc[(summaries["task"] == task) & (summaries["candidate_id"] == winner), "local_winner"] = True
        summaries.loc[
            (summaries["task"] == task) & (summaries["candidate_id"] == stable), "stable_alternative"
        ] = True
    summaries["macro_f1_rank"] = summaries.groupby("task")["macro_f1_mean"].rank(
        method="min", ascending=False
    )
    return summaries, selections


def common_output_fields(
    row: dict[str, Any], summary: pd.Series, control_id: str
) -> dict[str, Any]:
    return {
        **row,
        "status": summary["status"],
        "error_summary": summary["error_summary"],
        "phase_a_control": row["candidate_id"] == control_id,
        "local_winner": bool(summary["local_winner"]),
        "stable_alternative": bool(summary["stable_alternative"]),
    }


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 6) -> str:
    if frame.empty:
        return "No completed candidates."
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for values in frame.loc[:, columns].itertuples(index=False, name=None):
        shown: list[str] = []
        for value in values:
            if pd.isna(value):
                shown.append("")
            elif isinstance(value, (float, np.floating)):
                shown.append(f"{value:.{digits}f}")
            else:
                shown.append(str(value))
        lines.append("| " + " | ".join(shown) + " |")
    return "\n".join(lines)


def selection_label(row: pd.Series) -> str:
    labels: list[str] = []
    if row["local_winner"]:
        labels.append("local_winner")
    if row["stable_alternative"]:
        labels.append("stable_alternative")
    if row["phase_a_control"]:
        labels.append("phase_a_control")
    return ", ".join(labels)


def comparison_rows(
    config: dict[str, Any], task: str, winner_id: str | None, record_map: dict[tuple[str, str], dict[str, Any]], labels: Sequence[str]
) -> pd.DataFrame:
    control_id = config["phase_a_control_candidate"]
    rows: list[dict[str, Any]] = []
    for label in labels:
        values: dict[str, Any] = {"label": label}
        for name, candidate_id in (("control", control_id), ("winner", winner_id)):
            record = record_map.get((task, candidate_id)) if candidate_id else None
            frame = pd.DataFrame(record.get("per_class", [])) if record else pd.DataFrame()
            selected = frame[(frame["record_type"] == "oof_aggregate") & (frame["label"] == label)] if not frame.empty else frame
            values[f"{name}_recall"] = float(selected.iloc[0]["recall"]) if len(selected) else np.nan
            values[f"{name}_f1"] = float(selected.iloc[0]["f1"]) if len(selected) else np.nan
        values["recall_delta"] = values["winner_recall"] - values["control_recall"]
        values["f1_delta"] = values["winner_f1"] - values["control_f1"]
        rows.append(values)
    if len(rows) > 1:
        average = {"label": "Minority average"}
        for column in (
            "control_recall",
            "winner_recall",
            "recall_delta",
            "control_f1",
            "winner_f1",
            "f1_delta",
        ):
            average[column] = float(np.mean([row[column] for row in rows]))
        rows.append(average)
    return pd.DataFrame(rows)


def write_report(
    config: dict[str, Any], summaries: pd.DataFrame, selections: dict[str, tuple[str | None, str | None]], records: list[dict[str, Any]]
) -> None:
    record_map = {(record["base"]["task"], record["base"]["candidate_id"]): record for record in records}
    sections: list[str] = []
    control_id = config["phase_a_control_candidate"]
    for task in config["task_order"]:
        table = summaries[summaries["task"] == task].copy()
        table["selection"] = table.apply(selection_label, axis=1)
        table = table.sort_values(["macro_f1_mean", "macro_f1_std"], ascending=[False, True], na_position="last")
        metric_columns = [
            "macro_f1_rank",
            "candidate_id",
            "status",
            "macro_f1_mean",
            "macro_f1_std",
            "accuracy_mean",
            "weighted_f1_mean",
        ]
        if TASKS[task]["target"] == "Domain":
            metric_columns.extend(["domain_minority_recall_mean", "domain_minority_f1_mean"])
        metric_columns.extend(["total_fit_seconds", "selection"])
        sections.append(
            f"## {task}\n\n"
            + markdown_table(table, metric_columns)
        )

    minority_labels = config["selection"]["domain_minority_labels"]
    english_comparison = comparison_rows(
        config, "english_domain", selections["english_domain"][0], record_map, minority_labels
    )
    tulu_comparison = comparison_rows(
        config, "tulu_regret", selections["tulu_regret"][0], record_map, ["Inaction"]
    )
    counts = summaries["status"].value_counts().to_dict()
    completed = int(counts.get("complete", 0))
    failed = int(counts.get("failed", 0))
    timed_out = int(counts.get("timeout", 0))
    not_run = int(counts.get("not_run", 0))
    candidate_seconds = float(summaries["wall_seconds"].fillna(0).sum())
    worker_peak = summaries["peak_working_set_mib"].max()
    external_resource = read_record(RESOURCE_OBSERVATION_PATH) or {}
    external_peak = external_resource.get("peak_working_set_mib")
    valid_peaks = [float(value) for value in (worker_peak, external_peak) if pd.notna(value) and float(value) > 0]
    peak = max(valid_peaks) if valid_peaks else float("nan")
    peak_source = (
        "external Get-Process observation during execution"
        if external_peak is not None and float(external_peak) == peak
        else "worker telemetry"
    )
    run_state = read_record(RUN_STATE_PATH) or {"sessions": []}
    session_seconds = sum(float(item.get("session_wall_seconds", 0.0)) for item in run_state["sessions"])
    recommendations: list[str] = []
    for task in config["task_order"]:
        winner_id = selections[task][0]
        winner = summaries[(summaries["task"] == task) & (summaries["candidate_id"] == winner_id)]
        control = summaries[(summaries["task"] == task) & (summaries["candidate_id"] == control_id)]
        if len(winner) and len(control):
            delta = float(winner.iloc[0]["macro_f1_mean"] - control.iloc[0]["macro_f1_mean"])
            recommendations.append(f"- {task}: winner `{winner_id}` vs control Macro F1 delta `{delta:+.6f}`.")
    control_ranks = summaries[summaries["phase_a_control"]].copy()
    solver_note = (
        "scikit-learn 1.8.0 rejects direct multiclass use of `solver=liblinear`; all LogisticRegression "
        "candidates therefore use the same `OneVsRestClassifier(LogisticRegression(solver=\"liblinear\"))` "
        "wrapper while retaining C=1.0, max_iter=2000, class weight, and seed 2026."
    )
    report = f"""# Phase B Traditional Model Five-Fold Results

Generated from labeled English and Tulu training data only. Official test files were not read by the Phase B experiment script, and no final full-data model, official test prediction, or submission was produced.

## Protocol

- CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)`.
- Every candidate in a task uses the same saved fold assignment; only SHA-256 hashes of training IDs are stored.
- Each fold fits a fresh complete sparse TF-IDF Pipeline. No vectorizer is fitted before splitting and no sparse matrix is converted to dense.
- Primary selection metric: five-fold Macro F1 mean. Accuracy and Weighted F1 are secondary.
- Candidate count: 25; complete: {completed}; failed: {failed}; timeout: {timed_out}; not run: {not_run}.
- Sum of candidate wall times: {candidate_seconds:.2f} seconds; orchestration session time: {session_seconds:.2f} seconds; maximum observed worker peak working set: {peak:.2f} MiB ({peak_source}).
- LogisticRegression compatibility: {solver_note}

## Selection rule

The local winner follows the configured sequence: highest Macro F1 mean; within 0.005 compare standard deviation; for Domain candidates that remain close compare the OOF average recall/F1 of Health, Education, and Career and Finance; then prefer the simpler/faster model. `stable_alternative` is the lowest-variance non-winner within 0.01 Macro F1 of the winner, or otherwise the raw runner-up. The fixed Phase A configuration is tagged `phase_a_control`.

{chr(10).join(sections)}

## Phase A control ranks under five-fold CV

{markdown_table(control_ranks, ['task', 'candidate_id', 'macro_f1_rank', 'macro_f1_mean', 'macro_f1_std'])}

## English Domain minority-class OOF comparison

Winner: `{selections['english_domain'][0]}`. Deltas are winner minus the Phase A control under the same five folds.

{markdown_table(english_comparison, ['label', 'control_recall', 'winner_recall', 'recall_delta', 'control_f1', 'winner_f1', 'f1_delta'])}

## Tulu Regret Inaction OOF comparison

Winner: `{selections['tulu_regret'][0]}`. Deltas are winner minus the Phase A control under the same five folds.

{markdown_table(tulu_comparison, ['label', 'control_recall', 'winner_recall', 'recall_delta', 'control_f1', 'winner_f1', 'f1_delta'])}

## Local recommendation evidence

{chr(10).join(recommendations)}

These are local candidate recommendations only. No final model has been fitted and official test results were not used for candidate selection. A second submission should be considered only after reviewing the task-level gains, variance, and minority-class trade-offs in this report.

## Artifacts and limitations

- Fold assignments: `outputs/phase_b_traditional/fold_assignments/*.csv` (training ID hashes only).
- Resumable candidate records: `outputs/phase_b_traditional/candidate_records/*.json`.
- Detailed tables: `reports/tables/phase_b_cv_metrics.csv`, `phase_b_per_class.csv`, and `phase_b_confusion.csv`.
- Runtime protection can leave timeout/failed/not-run candidates ineligible for selection; their statuses and error summaries remain in the metrics table.
- The per-worker Windows memory counter returned zero during this run. Peak memory is therefore the highest independent `Get-Process` observation made while workers were active; the corrected telemetry function applies to future runs, and completed candidates were not repeated.
- Results estimate training-distribution performance; English/Tulu have no official validation split, and five-fold CV does not remove every possible distribution-shift risk.
"""
    atomic_write_text(REPORT_PATH, report)


def summarize(config: dict[str, Any]) -> dict[str, Any]:
    records = load_all_records(config)
    control_id = config["phase_a_control_candidate"]
    summaries = pd.DataFrame([candidate_summary(record, control_id) for record in records])
    summaries, selections = add_selection_flags(config, summaries, records)
    summary_lookup = {
        (row["task"], row["candidate_id"]): row for _, row in summaries.iterrows()
    }

    metric_rows_output: list[dict[str, Any]] = []
    per_class_output: list[dict[str, Any]] = []
    confusion_output: list[dict[str, Any]] = []
    for record in records:
        key = (record["base"]["task"], record["base"]["candidate_id"])
        summary = summary_lookup[key]
        for row in record.get("fold_metrics", []):
            metric_rows_output.append(common_output_fields(row, summary, control_id))
        metric_rows_output.append(summary.to_dict())
        for row in record.get("per_class", []):
            per_class_output.append(common_output_fields(row, summary, control_id))
        for row in record.get("confusion", []):
            confusion_output.append(common_output_fields(row, summary, control_id))

    atomic_write_csv(METRICS_PATH, pd.DataFrame(metric_rows_output))
    atomic_write_csv(PER_CLASS_PATH, pd.DataFrame(per_class_output))
    atomic_write_csv(CONFUSION_PATH, pd.DataFrame(confusion_output))
    write_report(config, summaries, selections, records)
    return {
        "counts": summaries["status"].value_counts().to_dict(),
        "selections": selections,
        "metrics_rows": len(metric_rows_output),
        "per_class_rows": len(per_class_output),
        "confusion_rows": len(confusion_output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check-inputs", action="store_true")
    actions.add_argument("--smoke", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--summarize", action="store_true")
    actions.add_argument("--candidate", nargs=2, metavar=("TASK", "CANDIDATE_ID"))
    args = parser.parse_args()
    mode = (
        "check_inputs"
        if args.check_inputs
        else "smoke"
        if args.smoke
        else "run"
        if args.run
        else "summarize"
        if args.summarize
        else f"candidate_{args.candidate[0]}_{args.candidate[1]}"
    )
    log_path = setup_logging(mode)
    config = load_config()
    validate_config(config)
    logging.info("Phase B mode=%s config=%s log=%s", mode, CONFIG_PATH, log_path)
    if args.check_inputs:
        result = check_inputs(config)
    elif args.smoke:
        result = run_smoke(config)
    elif args.run:
        result = run_all(config)
    elif args.summarize:
        result = summarize(config)
    else:
        result = {"candidate_return_code": candidate_worker(config, args.candidate[0], args.candidate[1])}
    logging.info("Result: %s", result)
    if args.candidate and result["candidate_return_code"] != 0:
        return int(result["candidate_return_code"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
