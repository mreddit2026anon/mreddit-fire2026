"""Leakage-safe data, split, weighting, and metrics helpers for Phase 04.

This module deliberately exposes labeled training-data operations only. Test
loading and submission construction do not belong to the Transformer training
stage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from src.data_utils import DOMAIN_LABELS, REGRET_LABELS, TASKS, build_text, load_frame


REGRET_LABEL2ID = {
    "Action": 0,
    "Inaction": 1,
    "No Regret": 2,
}
DOMAIN_LABEL2ID = {
    "Education": 0,
    "Health": 1,
    "Career and Finance": 2,
    "Romance friends and parents": 3,
    "Other domains": 4,
}
LABEL2ID_BY_TARGET = {"Regret": REGRET_LABEL2ID, "Domain": DOMAIN_LABEL2ID}
DOMAIN_MINORITY_LABELS = ("Health", "Education", "Career and Finance")


class TransformerDataError(RuntimeError):
    """Raised when Transformer-stage data violates the fixed task contract."""


@dataclass(frozen=True)
class TaskSplit:
    task: str
    train: pd.DataFrame
    validation: pd.DataFrame
    label_names: tuple[str, ...]
    label2id: dict[str, int]


def get_task_spec(task: str) -> dict[str, Any]:
    if task == "kannada_domain":
        raise TransformerDataError("Kannada Domain is not an applicable task")
    if task not in TASKS:
        raise TransformerDataError(f"Unknown Transformer task: {task}")
    spec = TASKS[task]
    if spec["language"] == "Kannada" and spec["target"] == "Domain":
        raise TransformerDataError("Kannada Domain is not an applicable task")
    return spec


def fixed_label_mapping(task: str) -> tuple[dict[str, int], dict[int, str]]:
    spec = get_task_spec(task)
    label2id = dict(LABEL2ID_BY_TARGET[spec["target"]])
    expected = tuple(spec["labels"])
    if tuple(label2id) != expected or tuple(label2id.values()) != tuple(range(len(expected))):
        raise TransformerDataError(f"Fixed label mapping disagrees with repository contract for {task}")
    return label2id, {index: label for label, index in label2id.items()}


def load_training_task(root: Path, task: str) -> pd.DataFrame:
    """Load one labeled train split; this function never accepts a split name."""
    spec = get_task_spec(task)
    frame = load_frame(root, spec["language"], "train")
    text = build_text(frame, spec["language"])
    label2id, _ = fixed_label_mapping(task)
    labels = frame[spec["target"]].astype(str)
    invalid = sorted(set(labels) - set(label2id))
    if invalid:
        raise TransformerDataError(f"{task}: invalid labels {invalid}")
    result = pd.DataFrame(
        {
            "id": frame["id"].astype(str),
            "text": text.astype(str),
            "label": labels,
            "label_id": labels.map(label2id).astype("int64"),
        }
    )
    if result["id"].duplicated().any() or result["text"].eq("").any():
        raise TransformerDataError(f"{task}: duplicate IDs or empty normalized text")
    return result


def make_stratified_split(
    frame: pd.DataFrame,
    task: str,
    validation_fraction: float,
    seed: int,
) -> TaskSplit:
    spec = get_task_spec(task)
    label2id, _ = fixed_label_mapping(task)
    if not 0.0 < validation_fraction < 1.0:
        raise TransformerDataError("validation_fraction must be between 0 and 1")
    required = {"id", "text", "label", "label_id"}
    if not required.issubset(frame.columns):
        raise TransformerDataError(f"Missing split columns: {sorted(required - set(frame.columns))}")
    indices = np.arange(len(frame))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=frame["label_id"],
    )
    train = frame.iloc[train_indices].reset_index(drop=True)
    validation = frame.iloc[validation_indices].reset_index(drop=True)
    expected_labels = set(spec["labels"])
    if set(train["label"]) != expected_labels:
        raise TransformerDataError(f"{task}: training split does not contain every label")
    if set(validation["label"]) != expected_labels:
        raise TransformerDataError(f"{task}: validation split does not contain every label")
    overlap = set(train["id"]) & set(validation["id"])
    if overlap:
        raise TransformerDataError(f"{task}: train/validation ID overlap")
    return TaskSplit(task, train, validation, tuple(spec["labels"]), label2id)


def make_smoke_split(
    frame: pd.DataFrame,
    task: str,
    train_per_class: int,
    eval_per_class: int,
    seed: int,
) -> TaskSplit:
    if train_per_class < 1 or eval_per_class < 1:
        raise TransformerDataError("Smoke sampling needs positive train/eval counts")
    spec = get_task_spec(task)
    train_parts: list[pd.DataFrame] = []
    eval_parts: list[pd.DataFrame] = []
    for label in spec["labels"]:
        group = frame.loc[frame["label"].eq(label)].sample(
            n=train_per_class + eval_per_class,
            random_state=seed,
        )
        train_parts.append(group.iloc[:train_per_class])
        eval_parts.append(group.iloc[train_per_class:])
    train = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    validation = pd.concat(eval_parts, ignore_index=True).reset_index(drop=True)
    if set(train["id"]) & set(validation["id"]):
        raise TransformerDataError("Smoke train/eval ID overlap")
    label2id, _ = fixed_label_mapping(task)
    return TaskSplit(task, train, validation, tuple(spec["labels"]), label2id)


def compute_class_weights(label_ids: Iterable[int], num_labels: int) -> np.ndarray:
    """Compute sqrt-balanced weights from the supplied training labels only."""
    values = np.asarray(list(label_ids), dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise TransformerDataError("Training label IDs must be a non-empty vector")
    counts = np.bincount(values, minlength=num_labels).astype(np.float64)
    if len(counts) != num_labels or np.any(counts <= 0):
        raise TransformerDataError("Every class must occur in the training split")
    weights = np.sqrt(values.size / (num_labels * counts))
    weights /= weights.mean()
    return weights.astype(np.float32)


def class_weight_record(split: TaskSplit) -> dict[str, Any]:
    weights = compute_class_weights(split.train["label_id"], len(split.label_names))
    counts = split.train["label"].value_counts()
    return {
        "source": "training_split_only",
        "formula": "sqrt(N / (K * count_c)), normalized to mean 1",
        "training_rows": len(split.train),
        "counts": {label: int(counts[label]) for label in split.label_names},
        "weights": {label: float(weights[index]) for index, label in enumerate(split.label_names)},
    }


def metric_bundle(y_true: Sequence[int], y_pred: Sequence[int], label_names: Sequence[str]) -> dict[str, Any]:
    label_ids = list(range(len(label_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)
    per_class = [
        {
            "label": label,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(label_names)
    ]
    confusion = [
        {
            "true_label": true_label,
            "predicted_label": predicted_label,
            "count": int(matrix[row_index, column_index]),
        }
        for row_index, true_label in enumerate(label_names)
        for column_index, predicted_label in enumerate(label_names)
    ]
    summary = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0)
        ),
    }
    minority = {
        row["label"]: {"recall": row["recall"], "f1": row["f1"]}
        for row in per_class
        if row["label"] in DOMAIN_MINORITY_LABELS
    }
    return {"summary": summary, "per_class": per_class, "confusion": confusion, "domain_minority": minority}


def trainer_metric_scalars(y_true: Sequence[int], y_pred: Sequence[int], label_names: Sequence[str]) -> dict[str, float]:
    bundle = metric_bundle(y_true, y_pred, label_names)
    result = dict(bundle["summary"])
    for row in bundle["per_class"]:
        slug = row["label"].lower().replace(" ", "_")
        result[f"recall_{slug}"] = row["recall"]
        result[f"f1_{slug}"] = row["f1"]
    return result


def config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_manifest(split: TaskSplit, seed: int, validation_fraction: float, config_hash: str) -> pd.DataFrame:
    rows = []
    for split_name, frame in (("train", split.train), ("validation", split.validation)):
        for value in frame["id"].astype(str):
            rows.append(
                {
                    "task": split.task,
                    "id": value,
                    "id_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
                    "split": split_name,
                    "seed": seed,
                    "validation_fraction": validation_fraction,
                    "config_sha256": config_hash,
                }
            )
    return pd.DataFrame(rows)
