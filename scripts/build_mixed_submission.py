"""Build or validate an isolated Stage 4 mixed submission candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transformer_submission import (
    Stage4Error,
    build_candidate_frame,
    validate_candidate_package,
    validate_stage4_config,
    write_candidate_package,
)


DEFAULT_CONFIG = ROOT / "configs/transformer_submission.json"


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_stage4_config(config)
    return config


def dry_run(config: dict[str, Any], candidate: str) -> dict[str, Any]:
    if candidate not in config["candidates"]:
        raise Stage4Error(f"Unknown candidate: {candidate}")
    item = config["candidates"][candidate]
    return {
        "passed": True,
        "mode": "dry_run",
        "candidate": candidate,
        "sources": item["sources"],
        "output_dir": item["output_dir"],
        "test_text_read": False,
        "prediction_files_read": False,
        "model_loaded": False,
        "submission_generated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace this candidate's Stage 4 artifacts")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    if args.dry_run:
        result = dry_run(config, args.candidate)
    elif args.validate_only:
        if args.candidate not in config["candidates"]:
            raise Stage4Error(f"Unknown candidate: {args.candidate}")
        result = validate_candidate_package(ROOT, config, args.candidate)
    else:
        frame, metadata = build_candidate_frame(ROOT, config, args.candidate)
        result = write_candidate_package(
            ROOT,
            config,
            args.candidate,
            frame,
            metadata,
            overwrite=args.overwrite,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
