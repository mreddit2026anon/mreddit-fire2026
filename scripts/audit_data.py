"""Reproducible, read-only audit and EDA for the FIRE 2026 mReDDIT data."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED = 2026
LANGUAGES = ("English", "Tulu", "Kannada")
SPLITS = ("train", "validation", "test")
REGRET_LABELS = ("Action", "Inaction", "No Regret")
DOMAIN_LABELS = (
    "Education",
    "Health",
    "Career and Finance",
    "Romance friends and parents",
    "Other domains",
)


@dataclass(frozen=True)
class FileSpec:
    language: str
    split: str
    path: Path
    columns: tuple[str, ...]


FILE_SPECS = (
    FileSpec("English", "train", Path("data/raw/training_data/english_train_data.csv"),
             ("id", "title", "text", "Regret", "Domain")),
    FileSpec("English", "test", Path("data/raw/test_data/english_test_data.csv"),
             ("id", "title", "text")),
    FileSpec("Tulu", "train", Path("data/raw/training_data/tulu_train_data.csv"),
             ("id", "title", "text", "Regret", "Domain")),
    FileSpec("Tulu", "test", Path("data/raw/test_data/tulu_test_data.csv"),
             ("id", "title", "text")),
    FileSpec("Kannada", "train", Path("data/raw/training_data/kannada_train_data.csv"),
             ("id", "Text", "Regret")),
    FileSpec("Kannada", "validation", Path("data/raw/training_data/kannada_validation_data.csv"),
             ("id", "Text", "Regret")),
    FileSpec("Kannada", "test", Path("data/raw/test_data/kannada_test_data.csv"),
             ("id", "Text")),
)
BASELINE_PATH = Path("official/baseline_predictions.csv")
TABLE_DIR = Path("reports/tables")
FIGURE_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/data_audit.md")


class AuditError(RuntimeError):
    """Raised when an input violates the official data contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_missing(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype("string").str.strip().eq("")


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKC", str(value)).strip()
    return " ".join(normalized.split())


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def combined_text(frame: pd.DataFrame, language: str) -> pd.Series:
    if language == "Kannada":
        return frame["Text"].map(normalize_text)
    title = frame["title"].map(normalize_text)
    body = frame["text"].map(normalize_text)
    return (title + " " + body).str.strip()


def validate_frame(frame: pd.DataFrame, spec: FileSpec) -> None:
    actual = tuple(frame.columns)
    if actual != spec.columns:
        raise AuditError(f"{spec.path}: expected columns {spec.columns}, found {actual}")
    if spec.split == "test" and ({"Regret", "Domain"} & set(actual)):
        raise AuditError(f"{spec.path}: test data unexpectedly contains label columns")
    if spec.split != "test":
        missing = int(is_missing(frame["Regret"]).sum())
        invalid = sorted(set(frame.loc[~is_missing(frame["Regret"]), "Regret"]) - set(REGRET_LABELS))
        if missing or invalid:
            raise AuditError(f"{spec.path}: Regret missing={missing}, invalid={invalid}")
    if spec.language in {"English", "Tulu"} and spec.split == "train":
        missing = int(is_missing(frame["Domain"]).sum())
        invalid = sorted(set(frame.loc[~is_missing(frame["Domain"]), "Domain"]) - set(DOMAIN_LABELS))
        if missing or invalid:
            raise AuditError(f"{spec.path}: Domain missing={missing}, invalid={invalid}")
    if spec.language == "Kannada" and "Domain" in actual:
        raise AuditError(f"{spec.path}: Kannada must not contain a Domain column")


def load_inputs(root: Path, smoke: bool) -> tuple[dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    nrows = 25 if smoke else None
    for spec in FILE_SPECS:
        path = root / spec.path
        if not path.is_file():
            raise AuditError(f"Missing input file: {spec.path}")
        try:
            frame = pd.read_csv(path, nrows=nrows)
        except Exception as exc:
            raise AuditError(f"Could not read {spec.path}: {exc}") from exc
        validate_frame(frame, spec)
        frames[(spec.language, spec.split)] = frame
    baseline_path = root / BASELINE_PATH
    if not baseline_path.is_file():
        raise AuditError(f"Missing input file: {BASELINE_PATH}")
    try:
        baseline = pd.read_csv(baseline_path, nrows=nrows, keep_default_na=False)
    except Exception as exc:
        raise AuditError(f"Could not read {BASELINE_PATH}: {exc}") from exc
    if tuple(baseline.columns) != ("id", "Regret", "Domain"):
        raise AuditError(f"{BASELINE_PATH}: expected columns ('id', 'Regret', 'Domain'), found {tuple(baseline.columns)}")
    return frames, baseline


def dataset_sizes(root: Path, frames: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    records = []
    for spec in FILE_SPECS:
        frame = frames[(spec.language, spec.split)]
        records.append({
            "file": spec.path.name,
            "language": spec.language,
            "split": spec.split,
            "rows": len(frame),
            "columns": "|".join(frame.columns),
            "sha256": sha256_file(root / spec.path),
        })
    return pd.DataFrame(records, columns=["file", "language", "split", "rows", "columns", "sha256"])


def missing_values(frames: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    records = []
    for spec in FILE_SPECS:
        frame = frames[(spec.language, spec.split)]
        for field in frame.columns:
            count = int(is_missing(frame[field]).sum())
            records.append({"file": spec.path.name, "language": spec.language, "split": spec.split,
                            "field": field, "missing": count, "missing_pct": count / len(frame) if len(frame) else 0.0})
    return pd.DataFrame(records, columns=["file", "language", "split", "field", "missing", "missing_pct"])


def id_integrity(frames: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    records = []
    all_ids: list[str] = []
    for spec in FILE_SPECS:
        ids = frames[(spec.language, spec.split)]["id"].astype("string")
        all_ids.extend(ids.tolist())
        records.append({"scope": spec.path.name, "language": spec.language, "split": spec.split,
                        "rows": len(ids), "blank_ids": int(is_missing(ids).sum()),
                        "duplicate_ids": int(ids.duplicated(keep=False).sum()), "unique_ids": int(ids.nunique(dropna=False))})
    all_series = pd.Series(all_ids, dtype="string")
    records.append({"scope": "all_official_data", "language": "All", "split": "all", "rows": len(all_series),
                    "blank_ids": int(is_missing(all_series).sum()),
                    "duplicate_ids": int(all_series.duplicated(keep=False).sum()),
                    "unique_ids": int(all_series.nunique(dropna=False))})
    return pd.DataFrame(records, columns=["scope", "language", "split", "rows", "blank_ids", "duplicate_ids", "unique_ids"])


def distribution_table(frames: dict[tuple[str, str], pd.DataFrame], label: str,
                       groups: Sequence[tuple[str, str]], labels: Sequence[str]) -> pd.DataFrame:
    records = []
    for language, split in groups:
        counts = frames[(language, split)][label].value_counts()
        total = int(counts.sum())
        for value in labels:
            count = int(counts.get(value, 0))
            records.append({"language": language, "split": split, "label": value,
                            "count": count, "proportion": count / total if total else 0.0})
    return pd.DataFrame(records, columns=["language", "split", "label", "count", "proportion"])


def imbalance_table(regret: pd.DataFrame, domain: pd.DataFrame) -> pd.DataFrame:
    records = []
    for task, table in (("Regret", regret), ("Domain", domain)):
        for (language, split), group in table.groupby(["language", "split"], sort=False):
            counts = group["count"].to_numpy(dtype=float)
            positive = counts[counts > 0]
            total = counts.sum()
            probabilities = counts / total if total else counts
            entropy = -float(sum(p * math.log(p) for p in probabilities if p > 0))
            normalized = entropy / math.log(len(counts)) if len(counts) > 1 and total else 0.0
            records.append({"task": task, "language": language, "split": split,
                            "max_to_min_ratio": float(positive.max() / positive.min()) if len(positive) else np.nan,
                            "minority_proportion": float(positive.min() / total) if len(positive) and total else np.nan,
                            "normalized_entropy": normalized})
    return pd.DataFrame(records, columns=["task", "language", "split", "max_to_min_ratio",
                                          "minority_proportion", "normalized_entropy"])


def describe_lengths(values: pd.Series, missing: int, language: str, split: str,
                     field: str, unit: str) -> dict[str, object]:
    numeric = values.astype(float)
    quantiles = numeric.quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99]) if len(numeric) else pd.Series(dtype=float)
    return {
        "language": language, "split": split, "field": field, "unit": unit,
        "count": int(len(numeric)), "missing": int(missing),
        "mean": numeric.mean(), "std": numeric.std(ddof=1), "min": numeric.min(),
        "p25": quantiles.get(0.25, np.nan), "median": quantiles.get(0.50, np.nan),
        "p75": quantiles.get(0.75, np.nan), "p90": quantiles.get(0.90, np.nan),
        "p95": quantiles.get(0.95, np.nan), "p99": quantiles.get(0.99, np.nan), "max": numeric.max(),
    }


def text_length_summary(frames: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    records = []
    for spec in FILE_SPECS:
        frame = frames[(spec.language, spec.split)]
        fields = ("Text",) if spec.language == "Kannada" else ("title", "text", "title_plus_text")
        for field in fields:
            if field == "title_plus_text":
                raw = frame["title"].fillna("").astype(str).str.strip() + " " + frame["text"].fillna("").astype(str).str.strip()
                raw = raw.str.strip()
                missing_mask = is_missing(frame["title"]) & is_missing(frame["text"])
            else:
                raw = frame[field]
                missing_mask = is_missing(raw)
            normalized = raw.loc[~missing_mask].map(normalize_text)
            char_lengths = normalized.str.len()
            token_lengths = normalized.map(lambda text: len(text.split()) if text else 0)
            records.append(describe_lengths(char_lengths, int(missing_mask.sum()), spec.language, spec.split, field, "unicode_characters"))
            records.append(describe_lengths(token_lengths, int(missing_mask.sum()), spec.language, spec.split, field, "whitespace_tokens"))
    columns = ["language", "split", "field", "unit", "count", "missing", "mean", "std", "min",
               "p25", "median", "p75", "p90", "p95", "p99", "max"]
    return pd.DataFrame(records, columns=columns)


def duplicate_audit(frames: dict[tuple[str, str], pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prepared: dict[tuple[str, str], pd.DataFrame] = {}
    for spec in FILE_SPECS:
        frame = frames[(spec.language, spec.split)].copy()
        frame["_normalized"] = combined_text(frame, spec.language)
        frame["_hash"] = frame["_normalized"].map(text_hash)
        prepared[(spec.language, spec.split)] = frame

    duplicate_records = []
    kannada_records = []
    for language in LANGUAGES:
        available = [split for split in SPLITS if (language, split) in prepared]
        for split in available:
            hashes = prepared[(language, split)]["_hash"]
            counts = hashes.value_counts()
            repeated = counts[counts > 1]
            record = {"language": language, "scope": "within_split", "split_a": split, "split_b": "",
                      "duplicate_groups": int(len(repeated)), "duplicate_rows": int(repeated.sum()),
                      "shared_unique_texts": int(len(repeated))}
            duplicate_records.append(record)
            if language == "Kannada":
                kannada_records.append(record.copy())
        for index, split_a in enumerate(available):
            for split_b in available[index + 1:]:
                counts_a = prepared[(language, split_a)]["_hash"].value_counts()
                counts_b = prepared[(language, split_b)]["_hash"].value_counts()
                shared = sorted(set(counts_a.index) & set(counts_b.index))
                record = {"language": language, "scope": "cross_split", "split_a": split_a, "split_b": split_b,
                          "duplicate_groups": len(shared),
                          "duplicate_rows": int(sum(counts_a[h] + counts_b[h] for h in shared)),
                          "shared_unique_texts": len(shared)}
                duplicate_records.append(record)
                if language == "Kannada":
                    kannada_records.append(record.copy())

    conflict_records = []
    for language in LANGUAGES:
        labeled_splits = [split for split in SPLITS if (language, split) in prepared and "Regret" in prepared[(language, split)]]
        if not labeled_splits:
            continue
        labeled = pd.concat([
            prepared[(language, split)].assign(_split=split) for split in labeled_splits
        ], ignore_index=True)
        for label_type in ("Regret", "Domain"):
            if label_type not in labeled.columns:
                continue
            valid = labeled.loc[~is_missing(labeled[label_type])]
            for hash_value, group in valid.groupby("_hash", sort=True):
                labels = sorted(group[label_type].astype(str).unique())
                if len(labels) > 1:
                    conflict_records.append({"text_sha256": hash_value, "language": language,
                                             "label_type": label_type,
                                             "split_scope": "|".join(sorted(group["_split"].unique())),
                                             "distinct_label_count": len(labels),
                                             "occurrence_count": len(group)})
    duplicate_columns = ["language", "scope", "split_a", "split_b", "duplicate_groups", "duplicate_rows", "shared_unique_texts"]
    conflict_columns = ["text_sha256", "language", "label_type", "split_scope", "distinct_label_count", "occurrence_count"]
    return (pd.DataFrame(duplicate_records, columns=duplicate_columns),
            pd.DataFrame(conflict_records, columns=conflict_columns),
            pd.DataFrame(kannada_records, columns=duplicate_columns))


def submission_audit(frames: dict[tuple[str, str], pd.DataFrame], baseline: pd.DataFrame) -> pd.DataFrame:
    official_order = []
    # The template's official language block order is English, Kannada, Tulu.
    for language in ("English", "Kannada", "Tulu"):
        official_order.extend(frames[(language, "test")]["id"].astype(str).tolist())
    baseline_ids = baseline["id"].astype(str).tolist()
    expected_set = set(official_order)
    actual_set = set(baseline_ids)
    kannada_ids = set(frames[("Kannada", "test")]["id"].astype(str))
    kannada_rows = baseline[baseline["id"].astype(str).isin(kannada_ids)]
    regret_invalid = int((~baseline["Regret"].isin(REGRET_LABELS)).sum())
    non_kannada = baseline[~baseline["id"].astype(str).isin(kannada_ids)]
    domain_invalid = int((~non_kannada["Domain"].isin(DOMAIN_LABELS)).sum())
    checks = [
        ("columns", "|".join(baseline.columns), "id|Regret|Domain", tuple(baseline.columns) == ("id", "Regret", "Domain")),
        ("row_count", len(baseline), len(official_order), len(baseline) == len(official_order)),
        ("duplicate_ids", int(baseline["id"].duplicated(keep=False).sum()), 0, not baseline["id"].duplicated().any()),
        ("missing_test_ids", len(expected_set - actual_set), 0, expected_set <= actual_set),
        ("extra_template_ids", len(actual_set - expected_set), 0, actual_set <= expected_set),
        ("exact_id_order", baseline_ids == official_order, True, baseline_ids == official_order),
        ("kannada_domain_nonempty", int((kannada_rows["Domain"].astype(str).str.strip() != "").sum()), 0,
         bool((kannada_rows["Domain"].astype(str).str.strip() == "").all())),
        ("invalid_regret_labels", regret_invalid, 0, regret_invalid == 0),
        ("invalid_english_tulu_domain_labels", domain_invalid, 0, domain_invalid == 0),
    ]
    result = pd.DataFrame(checks, columns=["check", "observed", "expected", "passed"])
    failed = result.loc[~result["passed"].astype(bool), "check"].tolist()
    if failed:
        raise AuditError(f"Submission template checks failed: {failed}")
    return result


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.6f")


def plot_regret(table: pd.DataFrame, path: Path) -> None:
    groups = [("English", "train"), ("Tulu", "train"), ("Kannada", "train"), ("Kannada", "validation")]
    x = np.arange(len(REGRET_LABELS)); width = 0.18
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for index, (language, split) in enumerate(groups):
        values = table[(table.language == language) & (table.split == split)].set_index("label").loc[list(REGRET_LABELS), "proportion"]
        ax.bar(x + (index - 1.5) * width, values, width, label=f"{language} {split}")
    ax.set(title="Regret Label Distribution", xlabel="Regret label", ylabel="Proportion", xticks=x, xticklabels=REGRET_LABELS)
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def plot_domain(table: pd.DataFrame, path: Path) -> None:
    x = np.arange(len(DOMAIN_LABELS)); width = 0.36
    fig, ax = plt.subplots(figsize=(11, 6))
    for index, language in enumerate(("English", "Tulu")):
        values = table[table.language == language].set_index("label").loc[list(DOMAIN_LABELS), "proportion"]
        ax.bar(x + (index - 0.5) * width, values, width, label=language)
    ax.set(title="Domain Label Distribution", xlabel="Domain label", ylabel="Proportion", xticks=x, xticklabels=DOMAIN_LABELS)
    ax.tick_params(axis="x", rotation=18); ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def plot_text_lengths(frames: dict[tuple[str, str], pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for language in LANGUAGES:
        values: list[int] = []
        for split in SPLITS:
            if (language, split) in frames:
                values.extend(combined_text(frames[(language, split)], language).str.len().tolist())
        array = np.asarray(values, dtype=float)
        cap = np.quantile(array, 0.99)
        clipped = np.minimum(array, cap)
        ordered = np.sort(clipped)
        y = np.arange(1, len(ordered) + 1) / len(ordered)
        ax.plot(ordered, y, label=f"{language} (P99={cap:.0f})")
    ax.set(title="Text Length Distribution (ECDF, Clipped at Language P99)",
           xlabel="Unicode characters", ylabel="Cumulative proportion")
    ax.legend(frameon=False); ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def plot_dataset_sizes(size_table: pd.DataFrame, path: Path) -> None:
    x = np.arange(len(LANGUAGES)); width = 0.24
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for index, split in enumerate(SPLITS):
        values = []
        for language in LANGUAGES:
            match = size_table[(size_table.language == language) & (size_table.split == split)]
            values.append(int(match.rows.iloc[0]) if len(match) else 0)
        ax.bar(x + (index - 1) * width, values, width, label=split.title())
    ax.set(title="Dataset Size by Language and Split", xlabel="Language", ylabel="Number of samples",
           xticks=x, xticklabels=LANGUAGES)
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: Sequence[str] | None = None, max_rows: int | None = None) -> str:
    shown = frame.loc[:, list(columns)] if columns else frame.copy()
    if max_rows is not None:
        shown = shown.head(max_rows)
    shown = shown.fillna("")
    headers = [str(column) for column in shown.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in shown.itertuples(index=False, name=None):
        cells = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(tables: dict[str, pd.DataFrame], root: Path) -> None:
    sizes = tables["dataset_sizes"]
    regret = tables["regret_distribution"].copy(); regret["proportion"] = regret["proportion"].map(lambda x: f"{x:.2%}")
    domain = tables["domain_distribution"].copy(); domain["proportion"] = domain["proportion"].map(lambda x: f"{x:.2%}")
    missing = tables["missing_values"]
    ids = tables["id_integrity"]
    lengths = tables["text_length_summary"]
    imbalance = tables["class_imbalance"].copy()
    for column in ("max_to_min_ratio", "minority_proportion", "normalized_entropy"):
        imbalance[column] = imbalance[column].map(lambda x: f"{x:.4f}")
    duplicates = tables["duplicate_summary"]
    conflicts = tables["label_conflicts"]
    kannada = tables["kannada_duplicate_audit"]
    submission = tables["submission_template_audit"]
    conflicts_by_type = (conflicts.groupby(["language", "label_type"], sort=False).size().reset_index(name="conflicting_text_hashes")
                         if len(conflicts) else pd.DataFrame(columns=["language", "label_type", "conflicting_text_hashes"]))

    report = f"""# FIRE 2026 mReDDIT Data Audit and EDA

Generated reproducibly by `scripts/audit_data.py` with seed `{SEED}`. Input files were read only. No model was trained and test data were not used for optimization.

## 1. Task and data overview

**Fact.** Regret detection covers English, Tulu, and Kannada. Domain identification covers English and Tulu only. Kannada Domain is not applicable and must remain empty. Macro F1 is the primary metric for later model selection; Accuracy and Weighted F1 are secondary.

## 2. Dataset files and schema

{markdown_table(sizes, ["file", "language", "split", "rows", "columns", "sha256"])}

**Fact.** SHA-256 values identify the exact raw inputs used for this audit. No raw file was rewritten.

## 3. Dataset sizes

English and Tulu have no official validation file; this is represented as absent rather than fabricated. Kannada has an official validation set.

![Dataset sizes](figures/dataset_sizes.png)

## 4. Label distributions

### Regret

{markdown_table(regret)}

![Regret distribution](figures/regret_distribution.png)

### Domain

{markdown_table(domain)}

![Domain distribution](figures/domain_distribution.png)

## 5. Missing values and ID integrity

**Fact.** Field-level counts are stored in `tables/missing_values.csv`. Fields with at least one missing value are shown below.

{markdown_table(missing[missing.missing > 0]) if bool((missing.missing > 0).any()) else "No missing or blank values were found in the audited fields."}

{markdown_table(ids)}

## 6. Text length statistics

The complete character and whitespace-token statistics are in `tables/text_length_summary.csv`.

{markdown_table(lengths[(lengths.unit == "unicode_characters") & (lengths.field.isin(["title_plus_text", "Text"]))], ["language", "split", "field", "count", "missing", "mean", "std", "median", "p90", "p95", "p99", "max"])}

![Text length distribution](figures/text_length_distribution.png)

**Limitation.** Whitespace token counts are coarse descriptive statistics. For Tulu and Kannada in particular, whitespace boundaries are not equivalent to linguistic word segmentation or Transformer tokens.

## 7. Class imbalance

{markdown_table(imbalance)}

**Recommendation.** Use Macro F1 as the primary validation and model-selection metric so minority classes contribute equally at the class level.

## 8. Duplicate and overlap audit

Texts were normalized with Unicode NFKC, trimmed, and had consecutive whitespace collapsed. Only aggregate counts and SHA-256 hashes are emitted; full post text is not exported.

{markdown_table(duplicates)}

Label conflicts aggregate:

{markdown_table(conflicts_by_type) if len(conflicts_by_type) else "No conflicting labels were found among labeled splits after normalization."}

**Interpretation.** Duplicate or overlapping text is a data property and is not, by itself, evidence of a rule violation. Cross-split overlap can nevertheless inflate random-split estimates and should be considered when designing later validation.

## 9. Kannada-specific findings

{markdown_table(kannada)}

**Leakage safeguard.** Kannada test overlap will not be used for manual label lookup, direct label copying, a text-to-label mapping, feature selection, or any prediction decision. Test-side overlap is reported only as an aggregate audit fact.

## 10. Submission template audit

{markdown_table(submission)}

**Fact.** Template IDs are checked against the official test IDs as both a set and an exact ordered sequence. Kannada Domain cells are required to be empty.

## 11. Risks and limitations

- Exact normalized duplicates are detected, but semantic paraphrases and near duplicates are outside this audit.
- Whitespace tokenization is not a language-aware tokenizer.
- English and Tulu lack official validation files; later local validation must use training data only.
- Dataset statistics describe the supplied files and do not establish representativeness of the underlying populations.

## 12. Recommendations for CPU baseline

- Use normalized `title + text` for English/Tulu and `Text` for Kannada while preserving the original raw files unchanged.
- Fit every vectorizer and preprocessing estimator on training folds only.
- Train Regret for all languages; train Domain only for English and Tulu.
- Use fixed-seed stratified validation and report Macro F1 first.
- Run a group-aware sensitivity analysis when exact duplicates span candidate training/validation folds.
- Keep official test IDs and template ordering immutable; leave Kannada Domain empty.

## 13. Statistics suitable for the paper

The sample counts, label proportions, character-length summaries, coarse whitespace-token summaries, class-imbalance measures, missingness, ID integrity, and aggregate duplicate/overlap rates can support a Dataset Statistics section. Report the whitespace-token limitation and describe duplicate findings neutrally. Do not publish raw post text or use test overlap to infer labels.
"""
    (root / REPORT_PATH).write_text(report, encoding="utf-8", newline="\n")


def run(root: Path, smoke: bool) -> None:
    np.random.seed(SEED)
    frames, baseline = load_inputs(root, smoke)
    regret_groups = (("English", "train"), ("Tulu", "train"), ("Kannada", "train"), ("Kannada", "validation"))
    domain_groups = (("English", "train"), ("Tulu", "train"))
    regret = distribution_table(frames, "Regret", regret_groups, REGRET_LABELS)
    domain = distribution_table(frames, "Domain", domain_groups, DOMAIN_LABELS)
    # Exercise core transformations during smoke mode without creating outputs.
    _ = missing_values(frames)
    _ = id_integrity(frames)
    _ = text_length_summary(frames)
    _ = imbalance_table(regret, domain)
    _ = duplicate_audit(frames)
    if smoke:
        print("SMOKE TEST PASSED: schemas, labels, text handling, and core audit computations succeeded on up to 25 rows per file.")
        return

    tables = {
        "dataset_sizes": dataset_sizes(root, frames),
        "regret_distribution": regret,
        "domain_distribution": domain,
        "missing_values": missing_values(frames),
        "id_integrity": id_integrity(frames),
        "text_length_summary": text_length_summary(frames),
        "class_imbalance": imbalance_table(regret, domain),
    }
    duplicate_summary, label_conflicts, kannada_audit = duplicate_audit(frames)
    tables["duplicate_summary"] = duplicate_summary
    tables["label_conflicts"] = label_conflicts
    tables["kannada_duplicate_audit"] = kannada_audit
    tables["submission_template_audit"] = submission_audit(frames, baseline)

    (root / TABLE_DIR).mkdir(parents=True, exist_ok=True)
    (root / FIGURE_DIR).mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        save_csv(table, root / TABLE_DIR / f"{name}.csv")
    plot_regret(regret, root / FIGURE_DIR / "regret_distribution.png")
    plot_domain(domain, root / FIGURE_DIR / "domain_distribution.png")
    plot_text_lengths(frames, root / FIGURE_DIR / "text_length_distribution.png")
    plot_dataset_sizes(tables["dataset_sizes"], root / FIGURE_DIR / "dataset_sizes.png")
    write_report(tables, root)
    print(f"FULL AUDIT PASSED: wrote {len(tables)} tables, 4 figures, and {REPORT_PATH}.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Read up to 25 rows per file and validate the audit flow without writing outputs.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path.cwd()
    if not (root / "AGENTS.md").is_file() or not (root / ".git").exists():
        print("ERROR: run this script from the repository root.", file=sys.stderr)
        return 2
    try:
        run(root, args.smoke)
    except (AuditError, OSError, ValueError, KeyError) as exc:
        print(f"AUDIT FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
