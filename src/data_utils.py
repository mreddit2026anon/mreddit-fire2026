"""Data contracts and text preparation for the Phase A CPU baseline."""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 2026
REGRET_LABELS = ("Action", "Inaction", "No Regret")
DOMAIN_LABELS = (
    "Education",
    "Health",
    "Career and Finance",
    "Romance friends and parents",
    "Other domains",
)
LANGUAGE_ORDER = ("English", "Kannada", "Tulu")

FILE_SPECS = {
    ("English", "train"): ("data/raw/training_data/english_train_data.csv", ("id", "title", "text", "Regret", "Domain")),
    ("English", "test"): ("data/raw/test_data/english_test_data.csv", ("id", "title", "text")),
    ("Tulu", "train"): ("data/raw/training_data/tulu_train_data.csv", ("id", "title", "text", "Regret", "Domain")),
    ("Tulu", "test"): ("data/raw/test_data/tulu_test_data.csv", ("id", "title", "text")),
    ("Kannada", "train"): ("data/raw/training_data/kannada_train_data.csv", ("id", "Text", "Regret")),
    ("Kannada", "validation"): ("data/raw/training_data/kannada_validation_data.csv", ("id", "Text", "Regret")),
    ("Kannada", "test"): ("data/raw/test_data/kannada_test_data.csv", ("id", "Text")),
}

TASKS = {
    "english_regret": {"language": "English", "target": "Regret", "labels": REGRET_LABELS},
    "tulu_regret": {"language": "Tulu", "target": "Regret", "labels": REGRET_LABELS},
    "kannada_regret": {"language": "Kannada", "target": "Regret", "labels": REGRET_LABELS},
    "english_domain": {"language": "English", "target": "Domain", "labels": DOMAIN_LABELS},
    "tulu_domain": {"language": "Tulu", "target": "Domain", "labels": DOMAIN_LABELS},
}


class DataContractError(RuntimeError):
    """Raised when an official input violates the expected contract."""


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value)).strip()
    return " ".join(value.split())


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_frame(root: Path, language: str, split: str) -> pd.DataFrame:
    relative, expected_columns = FILE_SPECS[(language, split)]
    path = root / relative
    if not path.is_file():
        raise DataContractError(f"Missing input file: {relative}")
    frame = pd.read_csv(path, dtype={"id": "string"})
    if tuple(frame.columns) != expected_columns:
        raise DataContractError(f"{relative}: expected {expected_columns}, found {tuple(frame.columns)}")
    if frame["id"].isna().any() or frame["id"].str.strip().eq("").any():
        raise DataContractError(f"{relative}: blank ID")
    if frame["id"].duplicated().any():
        raise DataContractError(f"{relative}: duplicate ID")
    if split != "test":
        invalid = sorted(set(frame["Regret"].dropna()) - set(REGRET_LABELS))
        if frame["Regret"].isna().any() or invalid:
            raise DataContractError(f"{relative}: invalid Regret labels {invalid}")
    if language in {"English", "Tulu"} and split == "train":
        invalid = sorted(set(frame["Domain"].dropna()) - set(DOMAIN_LABELS))
        if frame["Domain"].isna().any() or invalid:
            raise DataContractError(f"{relative}: invalid Domain labels {invalid}")
    if language == "Kannada" and "Domain" in frame.columns:
        raise DataContractError(f"{relative}: Kannada must not contain Domain")
    return frame


def build_text(frame: pd.DataFrame, language: str) -> pd.Series:
    if language == "Kannada":
        return frame["Text"].map(normalize_text)
    title = frame["title"].map(normalize_text)
    body = frame["text"].map(normalize_text)
    return "[TITLE] " + title + "\n[TEXT] " + body


def load_task_data(root: Path, task_name: str, include_kannada_validation: bool = False) -> tuple[pd.Series, pd.Series, pd.Series]:
    spec = TASKS[task_name]
    language = spec["language"]
    frame = load_frame(root, language, "train")
    if language == "Kannada" and include_kannada_validation:
        validation = load_frame(root, language, "validation")
        frame = pd.concat([frame, validation], ignore_index=True)
    return build_text(frame, language), frame[spec["target"]].astype(str), frame["id"].astype(str)


def fixed_stratified_split(y: pd.Series) -> tuple[list[int], list[int]]:
    indices = list(range(len(y)))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )
    return list(train_indices), list(validation_indices)


def validate_all_inputs(root: Path) -> dict[str, object]:
    frames = {key: load_frame(root, *key) for key in FILE_SPECS}
    all_ids = pd.concat([frame["id"].astype(str) for frame in frames.values()], ignore_index=True)
    if all_ids.duplicated().any():
        raise DataContractError("IDs are not globally unique across official data")

    template_path = root / "official/baseline_predictions.csv"
    template = pd.read_csv(template_path, dtype={"id": "string"}, keep_default_na=False)
    if tuple(template.columns) != ("id", "Regret", "Domain"):
        raise DataContractError("Official template columns must be id,Regret,Domain")
    expected_order: list[str] = []
    for language in LANGUAGE_ORDER:
        expected_order.extend(frames[(language, "test")]["id"].astype(str).tolist())
    if template["id"].astype(str).tolist() != expected_order:
        raise DataContractError("Official template ID order does not match English, Kannada, Tulu test order")
    if len(template) != 2569 or template["id"].duplicated().any():
        raise DataContractError("Official template row count or ID uniqueness is invalid")

    return {
        "files": {f"{language}_{split}": len(frames[(language, split)]) for language, split in FILE_SPECS},
        "template_rows": len(template),
        "template_language_order": list(LANGUAGE_ORDER),
        "seed": SEED,
    }
