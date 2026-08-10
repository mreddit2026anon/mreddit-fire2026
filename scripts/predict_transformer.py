"""Run isolated Stage 4 test inference from a task's saved best model."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils import TASKS, build_text, load_frame
from src.transformer_data import fixed_label_mapping
from src.transformer_submission import (
    TRANSFORMER_TASKS,
    Stage4Error,
    canonical_model_path,
    expected_task_ids,
    load_template,
    resolve_path,
    validate_prediction_frame,
    validate_stage4_config,
)


DEFAULT_CONFIG = ROOT / "configs/transformer_submission.json"


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_stage4_config(config)
    return config


def task_contract(config: dict[str, Any], task: str) -> dict[str, Any]:
    if task not in TRANSFORMER_TASKS:
        raise Stage4Error(f"Transformer test inference is not allowed for {task}")
    item = config["transformer_tasks"][task]
    return {
        "task": task,
        "language": TASKS[task]["language"],
        "target": TASKS[task]["target"],
        "labels": list(TASKS[task]["labels"]),
        "best_model_path": item["model_dir"],
        "prediction_path": item["prediction_file"],
    }


def validate_saved_model(model_dir: Path) -> list[dict[str, Any]]:
    required = ("config.json", "model.safetensors")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Best model is incomplete; missing {missing}: {model_dir}")
    inventory = []
    for path in sorted(model_dir.iterdir()):
        if path.is_file():
            size = path.stat().st_size
            if size <= 0:
                raise Stage4Error(f"Best model contains an empty file: {path}")
            inventory.append({"name": path.name, "bytes": size})
    return inventory


def validate_model_labels(model: Any, task: str) -> None:
    expected_label2id, expected_id2label = fixed_label_mapping(task)
    actual_label2id = {str(key): int(value) for key, value in model.config.label2id.items()}
    actual_id2label = {int(key): str(value) for key, value in model.config.id2label.items()}
    if actual_label2id != expected_label2id or actual_id2label != expected_id2label:
        raise Stage4Error(f"{task}: saved model label mapping differs from the fixed repository mapping")


def import_inference_runtime() -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

    return {
        "torch": torch,
        "AutoModel": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "DataCollator": DataCollatorWithPadding,
    }


def predict_batches(
    runtime: dict[str, Any],
    model: Any,
    tokenizer: Any,
    texts: list[str],
    batch_size: int,
    max_length: int,
    use_bf16: bool,
) -> list[int]:
    torch = runtime["torch"]
    collator = runtime["DataCollator"](tokenizer=tokenizer, padding=True, return_tensors="pt")
    predictions: list[int] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        encoded = [
            tokenizer(text, truncation=True, max_length=max_length, padding=False)
            for text in texts[start : start + batch_size]
        ]
        batch = {key: value.to("cuda") for key, value in collator(encoded).items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(**batch).logits
        predictions.extend(logits.argmax(dim=-1).detach().cpu().tolist())
    return predictions


def run_inference(config: dict[str, Any], task: str, overwrite: bool = False) -> dict[str, Any]:
    contract = task_contract(config, task)
    model_dir = resolve_path(ROOT, contract["best_model_path"])
    prediction_path = resolve_path(ROOT, contract["prediction_path"])
    if prediction_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing Transformer prediction: {prediction_path}")
    inventory = validate_saved_model(model_dir)
    runtime = import_inference_runtime()
    torch = runtime["torch"]
    inference = config["inference"]
    if inference.get("cuda_required", True) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal Transformer test inference")
    use_bf16 = bool(inference.get("bf16", False))
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Inference configuration requires bf16, but the runtime does not support it")

    tokenizer = runtime["AutoTokenizer"].from_pretrained(model_dir, local_files_only=True)
    model = runtime["AutoModel"].from_pretrained(model_dir, local_files_only=True).to("cuda")
    validate_model_labels(model, task)

    language = contract["language"]
    test_frame = load_frame(ROOT, language, "test")
    template = load_template(ROOT, config)
    official_ids = expected_task_ids(template, config, task)
    if test_frame["id"].astype(str).tolist() != official_ids:
        raise Stage4Error(f"{task}: raw test IDs do not match the official template task range")
    texts = build_text(test_frame, language).astype(str).tolist()
    predicted_ids = predict_batches(
        runtime,
        model,
        tokenizer,
        texts,
        batch_size=int(inference["batch_size"]),
        max_length=int(inference["max_length"]),
        use_bf16=use_bf16,
    )
    _, id2label = fixed_label_mapping(task)
    frame = pd.DataFrame(
        {
            "id": test_frame["id"].astype(str),
            "predicted_label": [id2label[int(index)] for index in predicted_ids],
            "model_task": task,
            "best_model_path": canonical_model_path(task),
        }
    )
    validate_prediction_frame(
        frame,
        task,
        "transformer",
        test_frame["id"].astype(str).tolist(),
        canonical_model_path(task),
    )
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = prediction_path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    temporary.replace(prediction_path)

    reloaded = pd.read_csv(prediction_path, dtype={"id": "string"}, keep_default_na=False)
    validate_prediction_frame(
        reloaded,
        task,
        "transformer",
        official_ids,
        canonical_model_path(task),
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "rows": len(frame),
        "best_model_path": canonical_model_path(task),
        "best_model_inventory": inventory,
        "prediction_path": prediction_path.relative_to(ROOT).as_posix(),
        "label_counts": frame["predicted_label"].value_counts().reindex(TASKS[task]["labels"], fill_value=0).to_dict(),
        "device": torch.cuda.get_device_name(0),
        "bf16": use_bf16,
        "test_text_read": True,
        "submission_generated": False,
    }
    summary_path = prediction_path.with_name(f"{task}_test_inference.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def validate_only(config: dict[str, Any], task: str) -> dict[str, Any]:
    contract = task_contract(config, task)
    prediction_path = resolve_path(ROOT, contract["prediction_path"])
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Missing Transformer prediction: {prediction_path}")
    template = load_template(ROOT, config)
    frame = pd.read_csv(prediction_path, dtype={"id": "string"}, keep_default_na=False)
    validate_prediction_frame(
        frame,
        task,
        "transformer",
        expected_task_ids(template, config, task),
        canonical_model_path(task),
    )
    return {
        "passed": True,
        "task": task,
        "rows": len(frame),
        "prediction_path": contract["prediction_path"],
        "test_text_read": False,
        "model_loaded": False,
        "submission_generated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TRANSFORMER_TASKS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace this task's Stage 4 prediction")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    if args.dry_run:
        result = {
            **task_contract(config, args.task),
            "passed": True,
            "mode": "dry_run",
            "test_text_read": False,
            "model_loaded": False,
            "submission_generated": False,
        }
    elif args.validate_only:
        result = validate_only(config, args.task)
    else:
        result = run_inference(config, args.task, overwrite=args.overwrite)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
