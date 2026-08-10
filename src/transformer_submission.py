"""Pure validation and assembly helpers for Stage 4 mixed submissions."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_utils import TASKS


SUBMISSION_TASKS = (
    "english_regret",
    "english_domain",
    "kannada_regret",
    "tulu_regret",
    "tulu_domain",
)
TRANSFORMER_TASKS = (
    "english_regret",
    "english_domain",
    "tulu_regret",
    "tulu_domain",
)
TRANSFORMER_COLUMNS = ("id", "predicted_label", "model_task", "best_model_path")
PHASE_B_COLUMNS = ("id", "prediction")
SUBMISSION_COLUMNS = ("id", "Regret", "Domain")


class Stage4Error(RuntimeError):
    """Raised when inference or mixed-submission contracts are violated."""


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def canonical_model_path(task: str) -> str:
    return f"models/transformer_baseline/{task}/best"


def canonical_transformer_prediction_path(task: str) -> str:
    return f"outputs/transformer_baseline/predictions/{task}_test_predictions.csv"


def validate_stage4_config(config: dict[str, Any]) -> dict[str, Any]:
    if tuple(config.get("language_order", [])) != ("English", "Kannada", "Tulu"):
        raise Stage4Error("Language order must be English, Kannada, Tulu")
    counts = config.get("language_row_counts", {})
    if counts != {"English": 514, "Kannada": 1370, "Tulu": 685}:
        raise Stage4Error("Official language row counts changed")
    if set(config.get("transformer_tasks", {})) != set(TRANSFORMER_TASKS):
        raise Stage4Error("Transformer inference task set must contain exactly four English/Tulu tasks")
    for task in TRANSFORMER_TASKS:
        item = config["transformer_tasks"][task]
        if item.get("model_dir") != canonical_model_path(task):
            raise Stage4Error(f"{task}: best-model path is not canonical")
        if item.get("prediction_file") != canonical_transformer_prediction_path(task):
            raise Stage4Error(f"{task}: Transformer prediction path is not canonical")
    if "kannada_domain" in config.get("transformer_tasks", {}) or "kannada_regret" in config.get("transformer_tasks", {}):
        raise Stage4Error("Kannada Transformer inference is disabled in Stage 4")
    if set(config.get("phase_b_predictions", {})) != set(SUBMISSION_TASKS):
        raise Stage4Error("Phase B predictions must cover exactly five submission tasks")
    candidates = config.get("candidates", {})
    required = {"candidate_a_phase_b", "candidate_b_english_transformer"}
    if not required.issubset(candidates):
        raise Stage4Error("Candidate A and Candidate B configurations are required")
    for name, candidate in candidates.items():
        sources = candidate.get("sources", {})
        if set(sources) != set(SUBMISSION_TASKS):
            raise Stage4Error(f"{name}: source mapping must contain exactly five tasks")
        for task, source in sources.items():
            if source not in {"phase_b", "transformer"}:
                raise Stage4Error(f"{name}/{task}: unsupported prediction source {source}")
            if source == "transformer" and task not in TRANSFORMER_TASKS:
                raise Stage4Error(f"{name}/{task}: Transformer source is not allowed")
        expected_dir = f"outputs/transformer_mixed_submissions/{name}"
        if candidate.get("output_dir") != expected_dir:
            raise Stage4Error(f"{name}: output directory is not isolated/canonical")
    expected_a = {task: "phase_b" for task in SUBMISSION_TASKS}
    if candidates["candidate_a_phase_b"]["sources"] != expected_a:
        raise Stage4Error("Candidate A must use Phase B for all tasks")
    expected_b = dict(expected_a)
    expected_b.update({"english_regret": "transformer", "english_domain": "transformer"})
    if candidates["candidate_b_english_transformer"]["sources"] != expected_b:
        raise Stage4Error("Candidate B must replace only the two English tasks")
    inference = config.get("inference", {})
    if int(inference.get("batch_size", 0)) <= 0 or int(inference.get("max_length", 0)) <= 0:
        raise Stage4Error("Inference batch_size and max_length must be positive")
    if not inference.get("cuda_required") or not inference.get("bf16"):
        raise Stage4Error("Stage 4 formal inference requires CUDA and bf16")
    return {
        "passed": True,
        "transformer_tasks": list(TRANSFORMER_TASKS),
        "candidates": {name: candidate["sources"] for name, candidate in candidates.items()},
        "test_text_read": False,
        "model_loaded": False,
    }


def load_template(root: Path, config: dict[str, Any]) -> pd.DataFrame:
    path = resolve_path(root, config["template"])
    frame = pd.read_csv(path, dtype={"id": "string"}, keep_default_na=False)
    if tuple(frame.columns) != SUBMISSION_COLUMNS:
        raise Stage4Error("Official template columns must be id,Regret,Domain")
    expected_rows = sum(config["language_row_counts"].values())
    if len(frame) != expected_rows or frame["id"].duplicated().any():
        raise Stage4Error("Official template row count or ID uniqueness changed")
    return frame


def language_ids(template: pd.DataFrame, config: dict[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    start = 0
    for language in config["language_order"]:
        end = start + int(config["language_row_counts"][language])
        result[language] = template.iloc[start:end]["id"].astype(str).tolist()
        start = end
    if start != len(template):
        raise Stage4Error("Language row ranges do not cover the template exactly")
    return result


def expected_task_ids(template: pd.DataFrame, config: dict[str, Any], task: str) -> list[str]:
    if task not in TASKS:
        raise Stage4Error(f"Unknown task: {task}")
    return language_ids(template, config)[TASKS[task]["language"]]


def validate_prediction_frame(
    frame: pd.DataFrame,
    task: str,
    source: str,
    expected_ids: list[str],
    expected_model_path: str | None = None,
) -> pd.Series:
    if task not in SUBMISSION_TASKS:
        raise Stage4Error(f"Unsupported submission task: {task}")
    if source == "transformer":
        if tuple(frame.columns) != TRANSFORMER_COLUMNS:
            raise Stage4Error(f"{task}: expected Transformer columns {TRANSFORMER_COLUMNS}")
        label_column = "predicted_label"
        if not frame["model_task"].astype(str).eq(task).all():
            raise Stage4Error(f"{task}: model_task metadata mismatch")
        if expected_model_path is None or not frame["best_model_path"].astype(str).eq(expected_model_path).all():
            raise Stage4Error(f"{task}: best_model_path metadata mismatch")
    elif source == "phase_b":
        if tuple(frame.columns) != PHASE_B_COLUMNS:
            raise Stage4Error(f"{task}: expected Phase B columns {PHASE_B_COLUMNS}")
        label_column = "prediction"
    else:
        raise Stage4Error(f"Unsupported prediction source: {source}")
    ids = frame["id"].astype(str).tolist()
    if frame["id"].duplicated().any():
        raise Stage4Error(f"{task}: duplicate prediction IDs")
    if ids != expected_ids:
        raise Stage4Error(f"{task}: IDs do not exactly match official task order")
    labels = frame[label_column].astype(str)
    invalid = sorted(set(labels) - set(TASKS[task]["labels"]))
    if invalid:
        raise Stage4Error(f"{task}: invalid labels {invalid}")
    return labels


def source_prediction_path(root: Path, config: dict[str, Any], task: str, source: str) -> Path:
    if source == "phase_b":
        return resolve_path(root, config["phase_b_predictions"][task])
    if source == "transformer" and task in TRANSFORMER_TASKS:
        return resolve_path(root, config["transformer_tasks"][task]["prediction_file"])
    raise Stage4Error(f"{task}: unsupported source {source}")


def read_source_prediction(
    root: Path,
    config: dict[str, Any],
    template: pd.DataFrame,
    task: str,
    source: str,
) -> pd.DataFrame:
    path = source_prediction_path(root, config, task, source)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {source} prediction for {task}: {path}")
    frame = pd.read_csv(path, dtype={"id": "string"}, keep_default_na=False)
    model_path = canonical_model_path(task) if source == "transformer" else None
    validate_prediction_frame(frame, task, source, expected_task_ids(template, config, task), model_path)
    return frame


def validate_submission_frame(
    submission: pd.DataFrame,
    template: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    if tuple(submission.columns) != SUBMISSION_COLUMNS:
        raise Stage4Error("Submission columns must be exactly id,Regret,Domain")
    if submission["id"].astype(str).tolist() != template["id"].astype(str).tolist():
        raise Stage4Error("Submission IDs/order differ from official template")
    if submission["id"].duplicated().any():
        raise Stage4Error("Submission contains duplicate IDs")
    ids = language_ids(template, config)
    kannada_mask = submission["id"].astype(str).isin(set(ids["Kannada"]))
    if not submission.loc[kannada_mask, "Domain"].eq("").all():
        raise Stage4Error("Kannada Domain must be empty")
    if submission.loc[~kannada_mask, "Domain"].eq("").any():
        raise Stage4Error("English/Tulu Domain contains empty values")
    invalid_regret = sorted(set(submission["Regret"]) - set(TASKS["english_regret"]["labels"]))
    invalid_domain = sorted(set(submission.loc[~kannada_mask, "Domain"]) - set(TASKS["english_domain"]["labels"]))
    if invalid_regret or invalid_domain:
        raise Stage4Error(f"Invalid submission labels: Regret={invalid_regret}, Domain={invalid_domain}")
    return {"passed": True, "rows": len(submission), "kannada_domain_nonempty": 0}


def build_candidate_frame(
    root: Path,
    config: dict[str, Any],
    candidate_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validate_stage4_config(config)
    if candidate_name not in config["candidates"]:
        raise Stage4Error(f"Unknown candidate: {candidate_name}")
    template = load_template(root, config)
    submission = pd.DataFrame({"id": template["id"].astype(str), "Regret": "", "Domain": ""})
    candidate = config["candidates"][candidate_name]
    used: dict[str, Any] = {}
    for task in SUBMISSION_TASKS:
        source = candidate["sources"][task]
        frame = read_source_prediction(root, config, template, task, source)
        label_column = "predicted_label" if source == "transformer" else "prediction"
        mapping = dict(zip(frame["id"].astype(str), frame[label_column].astype(str), strict=True))
        mask = submission["id"].isin(mapping)
        if int(mask.sum()) != len(frame):
            raise Stage4Error(f"{candidate_name}/{task}: prediction IDs do not map one-to-one")
        submission.loc[mask, TASKS[task]["target"]] = submission.loc[mask, "id"].map(mapping)
        used[task] = {"source": source, "path": source_prediction_path(root, config, task, source).relative_to(root).as_posix()}
    validation = validate_submission_frame(submission, template, config)
    return submission, {"candidate": candidate_name, "sources": used, "validation": validation}


def write_candidate_package(
    root: Path,
    config: dict[str, Any],
    candidate_name: str,
    submission: pd.DataFrame,
    metadata: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    validate_stage4_config(config)
    if candidate_name not in config["candidates"]:
        raise Stage4Error(f"Unknown candidate: {candidate_name}")
    template = load_template(root, config)
    validate_submission_frame(submission, template, config)
    output_dir = resolve_path(root, config["candidates"][candidate_name]["output_dir"])
    csv_path = output_dir / "submission.csv"
    zip_path = output_dir / "submission.zip"
    metadata_path = output_dir / "build_metadata.json"
    existing = list(output_dir.iterdir()) if output_dir.is_dir() else []
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing candidate artifacts: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(csv_path, index=False, encoding="utf-8", lineterminator="\n")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname="submission.csv")
    with zipfile.ZipFile(zip_path, "r") as archive:
        if archive.namelist() != ["submission.csv"] or archive.testzip() is not None:
            raise Stage4Error("Candidate ZIP must contain exactly root submission.csv with valid CRC")
        if archive.read("submission.csv") != csv_path.read_bytes():
            raise Stage4Error("ZIP submission.csv differs from source CSV")
    payload = {
        **metadata,
        "csv_path": csv_path.relative_to(root).as_posix(),
        "zip_path": zip_path.relative_to(root).as_posix(),
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
        "zip_members": ["submission.csv"],
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def validate_candidate_package(root: Path, config: dict[str, Any], candidate_name: str) -> dict[str, Any]:
    validate_stage4_config(config)
    if candidate_name not in config["candidates"]:
        raise Stage4Error(f"Unknown candidate: {candidate_name}")
    template = load_template(root, config)
    output_dir = resolve_path(root, config["candidates"][candidate_name]["output_dir"])
    csv_path = output_dir / "submission.csv"
    zip_path = output_dir / "submission.zip"
    if not csv_path.is_file() or not zip_path.is_file():
        raise FileNotFoundError(f"Missing candidate CSV/ZIP in {output_dir}")
    frame = pd.read_csv(csv_path, dtype={"id": "string"}, keep_default_na=False)
    validation = validate_submission_frame(frame, template, config)
    with zipfile.ZipFile(zip_path, "r") as archive:
        if archive.namelist() != ["submission.csv"] or archive.testzip() is not None:
            raise Stage4Error("Invalid candidate ZIP structure or CRC")
        if archive.read("submission.csv") != csv_path.read_bytes():
            raise Stage4Error("Candidate ZIP bytes differ from CSV")
    return {**validation, "zip_members": ["submission.csv"], "test_text_read": False, "model_loaded": False}
