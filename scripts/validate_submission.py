"""Strict validation for the Codabench submission CSV and ZIP."""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils import DOMAIN_LABELS, LANGUAGE_ORDER, REGRET_LABELS, load_frame


class SubmissionError(RuntimeError):
    pass


def _official_context(root: Path) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    template = pd.read_csv(root / "official/baseline_predictions.csv", dtype={"id": "string"}, keep_default_na=False)
    language_ids = {
        language: set(load_frame(root, language, "test")["id"].astype(str)) for language in LANGUAGE_ORDER
    }
    return template, language_ids


def _validate_frame(frame: pd.DataFrame, root: Path) -> dict[str, object]:
    template, language_ids = _official_context(root)
    errors = []
    if tuple(frame.columns) != ("id", "Regret", "Domain"):
        errors.append(f"columns={tuple(frame.columns)}")
    if len(frame) != 2569:
        errors.append(f"row_count={len(frame)}")
    if "id" not in frame.columns:
        raise SubmissionError("Missing id column; " + "; ".join(errors))
    ids = frame["id"].astype(str).tolist()
    template_ids = template["id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        errors.append("duplicate IDs")
    if set(ids) != set(template_ids):
        errors.append("ID set mismatch")
    if ids != template_ids:
        errors.append("ID order mismatch")
    if "Regret" in frame.columns:
        invalid = sorted(set(frame["Regret"]) - set(REGRET_LABELS))
        if invalid:
            errors.append(f"invalid Regret labels={invalid}")
    if "Domain" in frame.columns:
        kannada_mask = frame["id"].astype(str).isin(language_ids["Kannada"])
        if not frame.loc[kannada_mask, "Domain"].eq("").all():
            errors.append("Kannada Domain must be empty")
        non_kannada = frame.loc[~kannada_mask, "Domain"]
        if non_kannada.eq("").any():
            errors.append("English/Tulu Domain contains empty values")
        invalid = sorted(set(non_kannada) - set(DOMAIN_LABELS))
        if invalid:
            errors.append(f"invalid English/Tulu Domain labels={invalid}")
    if errors:
        raise SubmissionError("; ".join(errors))

    warnings = []
    distributions: dict[str, dict[str, dict[str, int]]] = {}
    for language in LANGUAGE_ORDER:
        block = frame[frame["id"].astype(str).isin(language_ids[language])]
        distributions[language] = {"Regret": block["Regret"].value_counts().sort_index().to_dict()}
        if block["Regret"].nunique() == 1:
            warnings.append(f"{language} Regret predictions contain one class")
        if language != "Kannada":
            distributions[language]["Domain"] = block["Domain"].value_counts().sort_index().to_dict()
            if block["Domain"].nunique() == 1:
                warnings.append(f"{language} Domain predictions contain one class")
    return {"passed": True, "rows": len(frame), "warnings": warnings, "distributions": distributions}


def validate_submission_bytes(content: bytes, root: Path = ROOT) -> dict[str, object]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SubmissionError(f"CSV is not valid UTF-8: {exc}") from exc
    frame = pd.read_csv(io.StringIO(text), dtype={"id": "string"}, keep_default_na=False)
    result = _validate_frame(frame, root)
    roundtrip = frame.to_csv(index=False, lineterminator="\n").encode("utf-8").decode("utf-8")
    if not roundtrip:
        raise SubmissionError("UTF-8 roundtrip failed")
    return result


def validate_submission_csv(path: Path, root: Path = ROOT) -> dict[str, object]:
    if not path.is_file():
        raise SubmissionError(f"Missing CSV: {path}")
    return validate_submission_bytes(path.read_bytes(), root)


def validate_submission_zip(path: Path, root: Path = ROOT) -> dict[str, object]:
    if not path.is_file():
        raise SubmissionError(f"Missing ZIP: {path}")
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if names != ["submission.csv"]:
            raise SubmissionError(f"ZIP must contain exactly root submission.csv; found {names}")
        member = PurePosixPath(names[0])
        if member.is_absolute() or len(member.parts) != 1 or ".." in member.parts:
            raise SubmissionError("ZIP contains an unsafe or nested path")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise SubmissionError(f"ZIP CRC failed for {bad_member}")
        content = archive.read("submission.csv")
    result = validate_submission_bytes(content, root)
    result["zip_members"] = ["submission.csv"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--zip", dest="zip_path", type=Path)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root containing organizer-provided official/ and data/raw/ files",
    )
    args = parser.parse_args()
    if not args.csv and not args.zip_path:
        parser.error("provide --csv and/or --zip")
    try:
        output = {}
        if args.csv:
            output["csv"] = validate_submission_csv(args.csv, args.root)
        if args.zip_path:
            output["zip"] = validate_submission_zip(args.zip_path, args.root)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except (SubmissionError, OSError, zipfile.BadZipFile, pd.errors.ParserError) as exc:
        print(f"SUBMISSION VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
