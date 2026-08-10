"""Train and evaluate the independent Phase 04 XLM-R baseline.

This entry point has no prediction or submission action and loads labeled
training data only. A successful GPU smoke run is required before full training.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transformer_data import (
    DOMAIN_LABEL2ID,
    REGRET_LABEL2ID,
    TaskSplit,
    class_weight_record,
    compute_class_weights,
    config_sha256,
    fixed_label_mapping,
    get_task_spec,
    load_training_task,
    make_smoke_split,
    make_stratified_split,
    metric_bundle,
    split_manifest,
    trainer_metric_scalars,
)


DEFAULT_CONFIG = ROOT / "configs/transformer_baseline.json"


def configured_early_stopping_patience(config: dict[str, Any]) -> int | None:
    """Return a positive patience, or None when early stopping is disabled."""
    value = config["training"].get("early_stopping_patience")
    if value is None:
        return None
    patience = int(value)
    if patience < 1:
        raise ValueError("early_stopping_patience must be null or a positive integer")
    return patience


def trainer_callbacks(runtime: dict[str, Any], config: dict[str, Any], smoke: bool) -> list[Any]:
    """Build callbacks without registering early stopping when patience is null."""
    if smoke:
        return []
    patience = configured_early_stopping_patience(config)
    if patience is None:
        return []
    return [runtime["EarlyStoppingCallback"](early_stopping_patience=patience)]


def continuation_contract(config: dict[str, Any], task: str) -> dict[str, Any] | None:
    contract = config.get("continuation")
    if contract is None:
        return None
    if contract.get("task") != task:
        raise ValueError(f"Continuation contract does not apply to task: {task}")
    return contract


def training_model_source(config: dict[str, Any], task: str) -> str | Path:
    contract = continuation_contract(config, task)
    if contract is None:
        return config["model_name"]
    return resolve_path(str(contract["model_source_path"]))


def enforce_baseline_metric(observed: float, expected: float, tolerance: float) -> float:
    if not math.isfinite(observed) or tolerance < 0:
        raise ValueError("Baseline metric and tolerance must be finite/non-negative")
    delta = abs(observed - expected)
    if delta > tolerance:
        raise RuntimeError(
            f"Baseline reproduction gate failed: observed={observed}, expected={expected}, "
            f"absolute_error={delta}, tolerance={tolerance}"
        )
    return delta


def baseline_reference_metadata(contract: dict[str, Any]) -> dict[str, Any]:
    expected = float(contract["expected_macro_f1"])
    training_time = float(contract["training_time_macro_f1"])
    independent_reload = float(contract["independent_reload_macro_f1"])
    one_prediction = contract.get("difference_due_to_one_inaction_prediction")
    if expected != independent_reload:
        raise ValueError("Continuation gate reference must equal the independent reload Macro F1")
    if one_prediction is not True:
        raise ValueError("Continuation audit must record the one-Inaction-prediction difference")
    return {
        "expected_macro_f1": expected,
        "training_time_macro_f1": training_time,
        "independent_reload_macro_f1": independent_reload,
        "difference_due_to_one_inaction_prediction": True,
        "reference_kind": "stable_independent_baseline_reload",
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("execution_enabled", True) is not True:
        raise ValueError(f"Experiment configuration is suspended: {config.get('suspension_reason', path)}")
    required_mapping = {"Regret": REGRET_LABEL2ID, "Domain": DOMAIN_LABEL2ID}
    if config.get("labels") != required_mapping:
        raise ValueError("Config label mappings differ from the fixed repository mapping")
    allowed_tasks = config.get("allowed_tasks", [])
    if not allowed_tasks:
        raise ValueError("Transformer configuration must allow at least one task")
    if config.get("first_task") not in allowed_tasks:
        raise ValueError("The first Transformer task must belong to allowed_tasks")
    if config["training"]["metric_for_best_model"] != "macro_f1":
        raise ValueError("Macro F1 must be the checkpoint-selection metric")
    configured_early_stopping_patience(config)
    if not config["text"].get("dynamic_padding"):
        raise ValueError("Phase 04 requires dynamic padding")
    task_configs = config.get("tasks", {})
    for task in allowed_tasks:
        spec = get_task_spec(task)
        task_config = task_configs.get(task)
        if not task_config:
            raise ValueError(f"Missing Transformer task configuration: {task}")
        if task_config.get("language") != spec["language"] or task_config.get("target") != spec["target"]:
            raise ValueError(f"Transformer task configuration disagrees with repository contract: {task}")
        if task_config.get("validation") != "stratified_80_20":
            raise ValueError(f"Unsupported validation protocol for {task}")
    contract = config.get("continuation")
    if contract is not None:
        if allowed_tasks != ["tulu_regret"] or contract.get("task") != "tulu_regret":
            raise ValueError("Continuation configuration is restricted to tulu_regret")
        required = {
            "initialized_from_baseline_best": True,
            "weights_only": True,
            "optimizer_state_restored": False,
            "scheduler_state_restored": False,
            "trainer_state_restored": False,
            "resume_from_checkpoint": False,
            "require_baseline_evaluation_gate": True,
        }
        for key, expected in required.items():
            if contract.get(key) is not expected:
                raise ValueError(f"Continuation contract requires {key}={expected}")
        if contract.get("model_source_path") != "models/transformer_baseline/tulu_regret/best":
            raise ValueError("Continuation must initialize from the canonical Tulu Regret baseline best model")
        if int(config["training"]["num_train_epochs"]) > 3:
            raise ValueError("Tulu Regret continuation is limited to three epochs")
        if float(config["training"]["learning_rate"]) != 5e-6:
            raise ValueError("Tulu Regret continuation learning rate must be 5e-6")
        if float(config["training"]["warmup_ratio"]) != 0.0:
            raise ValueError("Tulu Regret continuation uses zero warmup")
        if float(contract.get("baseline_macro_f1_absolute_tolerance", -1)) < 0:
            raise ValueError("Continuation baseline tolerance must be non-negative")
        baseline_reference_metadata(contract)
    return config


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def output_paths(config: dict[str, Any], task: str, smoke: bool = False) -> dict[str, Path]:
    outputs = config["outputs"]
    scope = Path("smoke") / task if smoke else Path(task)
    return {
        "output": resolve_path(outputs["root"]) / scope,
        "checkpoint": resolve_path(outputs["checkpoints"]) / scope,
        "model": resolve_path(outputs["models"]) / scope / "best",
        "log": resolve_path(outputs["logs"]) / scope,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    temporary.replace(path)


def setup_logging(config: dict[str, Any], task: str, mode: str) -> Path:
    paths = output_paths(config, task, smoke=mode == "smoke")
    paths["log"].mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = paths["log"] / f"{stamp}_{mode}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return path


def prepare_split(config: dict[str, Any], task: str) -> TaskSplit:
    frame = load_training_task(ROOT, task)
    return make_stratified_split(
        frame,
        task,
        validation_fraction=float(config["validation_fraction"]),
        seed=int(config["seed"]),
    )


def check_inputs(config: dict[str, Any], config_path: Path, task: str) -> dict[str, Any]:
    split = prepare_split(config, task)
    paths = output_paths(config, task)
    config_hash = config_sha256(config)
    manifest = split_manifest(
        split,
        seed=int(config["seed"]),
        validation_fraction=float(config["validation_fraction"]),
        config_hash=config_hash,
    )
    manifest_path = paths["output"] / "split_manifest.csv"
    atomic_write_csv(manifest_path, manifest)
    weights = class_weight_record(split)
    payload = {
        "created_utc": utc_now(),
        "phase": config["phase"],
        "task": task,
        "language": get_task_spec(task)["language"],
        "target": get_task_spec(task)["target"],
        "config_path": str(config_path.resolve()),
        "config_sha256": config_hash,
        "model_name": config["model_name"],
        "train_rows": len(split.train),
        "validation_rows": len(split.validation),
        "train_labels": split.train["label"].value_counts().reindex(split.label_names).astype(int).to_dict(),
        "validation_labels": split.validation["label"].value_counts().reindex(split.label_names).astype(int).to_dict(),
        "train_validation_id_overlap": 0,
        "class_weights": weights,
        "split_manifest": str(manifest_path.relative_to(ROOT)),
        "test_files_read": False,
        "submission_generated": False,
    }
    atomic_write_json(paths["output"] / "input_check.json", payload)
    atomic_write_json(paths["output"] / "config_snapshot.json", config)
    return payload


def import_training_runtime() -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    import transformers
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    return {
        "torch": torch,
        "functional": functional,
        "transformers": transformers,
        "Dataset": Dataset,
        "AutoModel": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "DataCollator": DataCollatorWithPadding,
        "EarlyStoppingCallback": EarlyStoppingCallback,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def configure_reproducibility(runtime: dict[str, Any], seed: int) -> None:
    torch = runtime["torch"]
    random.seed(seed)
    np.random.seed(seed)
    runtime["set_seed"](seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def require_gpu(runtime: dict[str, Any], bf16_required: bool) -> dict[str, Any]:
    torch = runtime["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Transformer smoke/training")
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    if bf16_required and not bf16_supported:
        raise RuntimeError("Configuration requires bf16 but this GPU/runtime does not support it")
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": runtime["transformers"].__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "bf16_supported": bf16_supported,
    }


def dataset_class(runtime: dict[str, Any]) -> type:
    base = runtime["Dataset"]

    class EncodedTextDataset(base):
        def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int) -> None:
            self.encodings = tokenizer(
                frame["text"].astype(str).tolist(),
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            self.labels = frame["label_id"].astype(int).tolist()

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item = {key: values[index] for key, values in self.encodings.items()}
            item["labels"] = self.labels[index]
            return item

    return EncodedTextDataset


def weighted_trainer_class(runtime: dict[str, Any]) -> type:
    base = runtime["Trainer"]
    functional = runtime["functional"]

    class WeightedCrossEntropyTrainer(base):
        def __init__(self, *args: Any, class_weights: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weights = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            loss = functional.cross_entropy(logits, labels, weight=weights)
            return (loss, outputs) if return_outputs else loss

    return WeightedCrossEntropyTrainer


def training_arguments(
    runtime: dict[str, Any],
    config: dict[str, Any],
    task: str,
    output_dir: Path,
    smoke: bool,
) -> Any:
    train = config["training"]
    common = {
        "output_dir": str(output_dir),
        "seed": int(config["seed"]),
        "data_seed": int(config["seed"]),
        "learning_rate": float(train["learning_rate"]),
        "weight_decay": float(train["weight_decay"]),
        "warmup_ratio": float(train["warmup_ratio"]),
        "max_grad_norm": float(train["max_grad_norm"]),
        "bf16": bool(train["bf16"]),
        "fp16": bool(train["fp16"]),
        "dataloader_num_workers": int(train["dataloader_num_workers"]),
        "report_to": [],
        "remove_unused_columns": True,
        "save_safetensors": True,
        "run_name": f"transformer_baseline_{task}_{'smoke' if smoke else 'train'}",
    }
    if smoke:
        smoke_config = config["smoke"]
        return runtime["TrainingArguments"](
            **common,
            max_steps=int(smoke_config["max_steps"]),
            per_device_train_batch_size=int(smoke_config["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(smoke_config["per_device_eval_batch_size"]),
            gradient_accumulation_steps=1,
            eval_strategy="no",
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=1,
        )
    return runtime["TrainingArguments"](
        **common,
        num_train_epochs=float(train["num_train_epochs"]),
        per_device_train_batch_size=int(train["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train["gradient_accumulation_steps"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=int(train["logging_steps"]),
        load_best_model_at_end=True,
        metric_for_best_model=str(train["metric_for_best_model"]),
        greater_is_better=bool(train["greater_is_better"]),
        save_total_limit=int(train["save_total_limit"]),
    )


def build_trainer(
    runtime: dict[str, Any],
    config: dict[str, Any],
    task: str,
    split: TaskSplit,
    model_source: str | Path,
    checkpoint_dir: Path,
    smoke: bool,
) -> tuple[Any, Any, Any, np.ndarray]:
    label2id, id2label = fixed_label_mapping(task)
    tokenizer = runtime["AutoTokenizer"].from_pretrained(str(model_source))
    model = runtime["AutoModel"].from_pretrained(
        str(model_source),
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )
    dataset_type = dataset_class(runtime)
    max_length = int(config["text"]["max_length"])
    train_dataset = dataset_type(split.train, tokenizer, max_length)
    validation_dataset = dataset_type(split.validation, tokenizer, max_length)
    collator = runtime["DataCollator"](tokenizer=tokenizer, padding=True)
    weights = compute_class_weights(split.train["label_id"], len(split.label_names))
    torch_weights = runtime["torch"].tensor(weights, dtype=runtime["torch"].float32)

    def compute_metrics(prediction: Any) -> dict[str, float]:
        predicted = np.argmax(prediction.predictions, axis=-1)
        return trainer_metric_scalars(prediction.label_ids, predicted, split.label_names)

    callbacks = trainer_callbacks(runtime, config, smoke)
    trainer_type = weighted_trainer_class(runtime)
    trainer = trainer_type(
        model=model,
        args=training_arguments(runtime, config, task, checkpoint_dir, smoke),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        class_weights=torch_weights,
    )
    return trainer, tokenizer, validation_dataset, weights


def prediction_details(prediction: Any, split: TaskSplit) -> dict[str, Any]:
    predicted = np.argmax(prediction.predictions, axis=-1)
    return metric_bundle(prediction.label_ids, predicted, split.label_names)


def save_metric_artifacts(output_dir: Path, details: dict[str, Any], prefix: str) -> None:
    atomic_write_json(output_dir / f"{prefix}_metrics.json", details)
    atomic_write_csv(output_dir / f"{prefix}_per_class.csv", pd.DataFrame(details["per_class"]))
    atomic_write_csv(output_dir / f"{prefix}_confusion.csv", pd.DataFrame(details["confusion"]))


def epoch_metric_rows(log_history: list[dict[str, Any]], total_epochs: int) -> list[dict[str, Any]]:
    """Extract one stable validation record per completed epoch."""
    if total_epochs < 1:
        raise ValueError("total_epochs must be positive")
    rows = []
    for item in log_history:
        if "eval_macro_f1" not in item or "epoch" not in item:
            continue
        epoch = int(round(float(item["epoch"])))
        rows.append(
            {
                "epoch": epoch,
                "progress_percent": round(epoch * 100.0 / total_epochs, 6),
                "global_step": int(item.get("step", 0)),
                "macro_f1": float(item["eval_macro_f1"]),
                "accuracy": float(item["eval_accuracy"]),
                "weighted_f1": float(item["eval_weighted_f1"]),
                "eval_loss": float(item["eval_loss"]),
            }
        )
    deduplicated = {row["epoch"]: row for row in rows}
    return [deduplicated[epoch] for epoch in sorted(deduplicated)]


def model_inventory(model_dir: Path) -> list[dict[str, Any]]:
    required = ("config.json", "model.safetensors")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Best model is incomplete; missing {missing}")
    rows = []
    for path in sorted(model_dir.iterdir()):
        if path.is_file():
            if path.stat().st_size <= 0:
                raise RuntimeError(f"Best model contains an empty file: {path.name}")
            rows.append({"name": path.name, "bytes": path.stat().st_size})
    return rows


def verify_formal_prerequisites(config: dict[str, Any], task: str) -> None:
    paths = output_paths(config, task)
    input_path = paths["output"] / "input_check.json"
    smoke_path = output_paths(config, task, smoke=True)["output"] / "smoke_result.json"
    if not input_path.is_file():
        raise RuntimeError(f"Run --check-inputs for {task} before full training")
    input_record = json.loads(input_path.read_text(encoding="utf-8"))
    if input_record.get("config_sha256") != config_sha256(config):
        raise RuntimeError("Input-check configuration hash is stale")
    if not smoke_path.is_file():
        raise RuntimeError(f"Run --smoke for {task} before full training")
    smoke_record = json.loads(smoke_path.read_text(encoding="utf-8"))
    if not smoke_record.get("passed") or smoke_record.get("config_sha256") != config_sha256(config):
        raise RuntimeError("Smoke result is missing, failed, or uses a stale configuration")
    if continuation_contract(config, task) is not None:
        baseline_path = paths["output"] / "baseline_evaluation.json"
        if not baseline_path.is_file():
            raise RuntimeError(f"Run --check-baseline for {task} before continuation training")
        baseline_record = json.loads(baseline_path.read_text(encoding="utf-8"))
        if not baseline_record.get("passed") or baseline_record.get("config_sha256") != config_sha256(config):
            raise RuntimeError("Baseline evaluation gate is missing, failed, or uses a stale configuration")


def evaluate_continuation_baseline(
    config: dict[str, Any],
    task: str,
    runtime: dict[str, Any],
    split: TaskSplit,
) -> dict[str, Any]:
    contract = continuation_contract(config, task)
    if contract is None:
        raise ValueError("--check-baseline requires a continuation configuration")
    paths = output_paths(config, task)
    source = training_model_source(config, task)
    if not isinstance(source, Path):
        raise RuntimeError("Continuation model source must be a local baseline-best directory")
    inventory = model_inventory(source)
    trainer, _, validation_dataset, _ = build_trainer(
        runtime,
        config,
        task,
        split,
        source,
        paths["checkpoint"] / "baseline_evaluation",
        smoke=True,
    )
    prediction = trainer.predict(validation_dataset)
    details = prediction_details(prediction, split)
    observed = float(details["summary"]["macro_f1"])
    reference = baseline_reference_metadata(contract)
    expected = float(reference["expected_macro_f1"])
    tolerance = float(contract["baseline_macro_f1_absolute_tolerance"])
    passed = abs(observed - expected) <= tolerance
    payload = {
        "created_utc": utc_now(),
        "passed": passed,
        "task": task,
        "config_sha256": config_sha256(config),
        "model_source_path": str(source.relative_to(ROOT)),
        "baseline_model_inventory": inventory,
        "observed_macro_f1": observed,
        "absolute_error": abs(observed - expected),
        "absolute_tolerance": tolerance,
        **reference,
        "metrics": details["summary"],
        "initialized_from_baseline_best": True,
        "optimizer_state_restored": False,
        "scheduler_state_restored": False,
        "trainer_state_restored": False,
        "resume_from_checkpoint": False,
        "test_files_read": False,
        "submission_generated": False,
    }
    atomic_write_json(paths["output"] / "baseline_evaluation.json", payload)
    save_metric_artifacts(paths["output"], details, "continuation_baseline")
    enforce_baseline_metric(observed, expected, tolerance)
    return payload


def run_baseline_check(config: dict[str, Any], task: str) -> dict[str, Any]:
    runtime = import_training_runtime()
    require_gpu(runtime, bool(config["training"]["bf16"]))
    configure_reproducibility(runtime, int(config["seed"]))
    split = prepare_split(config, task)
    paths = output_paths(config, task)
    atomic_write_json(paths["output"] / "config_snapshot.json", config)
    return evaluate_continuation_baseline(config, task, runtime, split)


def run_smoke(config: dict[str, Any], task: str) -> dict[str, Any]:
    runtime = import_training_runtime()
    gpu = require_gpu(runtime, bool(config["training"]["bf16"]))
    configure_reproducibility(runtime, int(config["seed"]))
    frame = load_training_task(ROOT, task)
    smoke_config = config["smoke"]
    split = make_smoke_split(
        frame,
        task,
        int(smoke_config["train_samples_per_class"]),
        int(smoke_config["eval_samples_per_class"]),
        int(config["seed"]),
    )
    paths = output_paths(config, task, smoke=True)
    atomic_write_json(paths["output"] / "config_snapshot.json", config)
    source = training_model_source(config, task)
    if isinstance(source, Path):
        model_inventory(source)
    trainer, tokenizer, validation_dataset, weights = build_trainer(
        runtime,
        config,
        task,
        split,
        source,
        paths["checkpoint"],
        smoke=True,
    )
    train_result = trainer.train()
    before = trainer.predict(validation_dataset)
    details = prediction_details(before, split)
    if not math.isfinite(float(train_result.training_loss)):
        raise RuntimeError("Smoke weighted training loss is not finite")
    smoke_model = paths["model"]
    smoke_model.mkdir(parents=True, exist_ok=True)
    trainer.save_model(smoke_model)
    tokenizer.save_pretrained(smoke_model)

    reloaded_trainer, _, reloaded_validation, _ = build_trainer(
        runtime,
        config,
        task,
        split,
        smoke_model,
        paths["checkpoint"] / "reload_check",
        smoke=True,
    )
    after = reloaded_trainer.predict(reloaded_validation)
    before_labels = np.argmax(before.predictions, axis=-1)
    after_labels = np.argmax(after.predictions, axis=-1)
    if not np.array_equal(before_labels, after_labels):
        raise RuntimeError("Smoke predictions changed after save/load")
    payload = {
        "created_utc": utc_now(),
        "passed": True,
        "task": task,
        "config_sha256": config_sha256(config),
        "gpu": gpu,
        "train_rows": len(split.train),
        "validation_rows": len(split.validation),
        "max_steps": int(smoke_config["max_steps"]),
        "training_loss": float(train_result.training_loss),
        "weighted_loss_finite": True,
        "class_weights_from_smoke_train_only": {
            label: float(weights[index]) for index, label in enumerate(split.label_names)
        },
        "save_load_predictions_match": True,
        "metrics": details,
        "formal_model_generated": False,
        "test_files_read": False,
        "submission_generated": False,
    }
    contract = continuation_contract(config, task)
    if contract is not None:
        payload.update(
            {
                "model_source_path": contract["model_source_path"],
                "initialized_from_baseline_best": True,
                "optimizer_state_restored": False,
                "scheduler_state_restored": False,
                "resume_from_checkpoint": False,
            }
        )
    atomic_write_json(paths["output"] / "smoke_result.json", payload)
    return payload


def run_training(config: dict[str, Any], task: str) -> dict[str, Any]:
    verify_formal_prerequisites(config, task)
    runtime = import_training_runtime()
    gpu = require_gpu(runtime, bool(config["training"]["bf16"]))
    configure_reproducibility(runtime, int(config["seed"]))
    split = prepare_split(config, task)
    paths = output_paths(config, task)
    atomic_write_json(paths["output"] / "config_snapshot.json", config)
    expected_manifest = split_manifest(
        split,
        int(config["seed"]),
        float(config["validation_fraction"]),
        config_sha256(config),
    )
    saved_manifest = pd.read_csv(paths["output"] / "split_manifest.csv", dtype={"id": "string"})
    if not expected_manifest.astype(str).equals(saved_manifest.astype(str)):
        raise RuntimeError("Saved split manifest does not match the reproducible split")
    contract = continuation_contract(config, task)
    baseline_evaluation = None
    if contract is not None:
        baseline_evaluation = evaluate_continuation_baseline(config, task, runtime, split)
        configure_reproducibility(runtime, int(config["seed"]))
    source = training_model_source(config, task)
    trainer, tokenizer, validation_dataset, weights = build_trainer(
        runtime,
        config,
        task,
        split,
        source,
        paths["checkpoint"],
        smoke=False,
    )
    train_result = trainer.train()
    prediction = trainer.predict(validation_dataset)
    details = prediction_details(prediction, split)
    paths["model"].mkdir(parents=True, exist_ok=True)
    trainer.save_model(paths["model"])
    tokenizer.save_pretrained(paths["model"])
    save_metric_artifacts(paths["output"], details, "validation")
    state = trainer.state
    state.save_to_json(str(paths["output"] / "trainer_state.json"))
    maximum_epochs = int(config["training"]["num_train_epochs"])
    epochs = epoch_metric_rows(state.log_history, maximum_epochs)
    if not epochs or len(epochs) > maximum_epochs:
        raise RuntimeError(f"Invalid per-epoch validation history: found {len(epochs)} records")
    atomic_write_csv(paths["output"] / "epoch_metrics.csv", pd.DataFrame(epochs))
    inventory = model_inventory(paths["model"])
    best_checkpoint = Path(state.best_model_checkpoint) if state.best_model_checkpoint else None
    if best_checkpoint is None or not best_checkpoint.is_dir():
        raise RuntimeError("Trainer did not preserve a valid best checkpoint directory")
    checkpoint_inventory = model_inventory(best_checkpoint)
    payload = {
        "created_utc": utc_now(),
        "task": task,
        "config_sha256": config_sha256(config),
        "gpu": gpu,
        "train_rows": len(split.train),
        "validation_rows": len(split.validation),
        "class_weights": {label: float(weights[index]) for index, label in enumerate(split.label_names)},
        "training_loss": float(train_result.training_loss),
        "global_step": int(state.global_step),
        "best_checkpoint": state.best_model_checkpoint,
        "best_checkpoint_inventory": checkpoint_inventory,
        "best_metric": None if state.best_metric is None else float(state.best_metric),
        "resolved_training_arguments": {
            "optimizer": str(trainer.args.optim),
            "lr_scheduler_type": str(trainer.args.lr_scheduler_type),
            "max_steps": int(state.max_steps),
            "warmup_ratio": float(trainer.args.warmup_ratio),
            "warmup_steps": int(trainer.args.get_warmup_steps(int(state.max_steps))),
            "per_device_train_batch_size": int(trainer.args.per_device_train_batch_size),
            "gradient_accumulation_steps": int(trainer.args.gradient_accumulation_steps),
            "learning_rate": float(trainer.args.learning_rate),
            "bf16": bool(trainer.args.bf16),
        },
        "initialization_contract": {
            "model_source": str(source.relative_to(ROOT)) if isinstance(source, Path) else str(source),
            "seed": int(config["seed"]),
            "reproducibility_configured_before_split_and_model_load": True,
            "resume_from_checkpoint": False,
        },
        "model_path": str(paths["model"].relative_to(ROOT)),
        "best_model_inventory": inventory,
        "epoch_metrics": epochs,
        "metrics": details["summary"],
        "domain_minority": details["domain_minority"],
        "test_files_read": False,
        "submission_generated": False,
    }
    if contract is not None:
        payload.update(
            {
                "model_source_path": contract["model_source_path"],
                "initialized_from_baseline_best": True,
                "optimizer_state_restored": False,
                "scheduler_state_restored": False,
                "trainer_state_restored": False,
                "resume_from_checkpoint": False,
                "learning_rate": float(config["training"]["learning_rate"]),
                "max_epochs": int(config["training"]["num_train_epochs"]),
                "continuation_baseline_evaluation": baseline_evaluation,
            }
        )
    atomic_write_json(paths["output"] / "training_summary.json", payload)
    return payload


def evaluate_only(config: dict[str, Any], task: str) -> dict[str, Any]:
    runtime = import_training_runtime()
    require_gpu(runtime, bool(config["training"]["bf16"]))
    configure_reproducibility(runtime, int(config["seed"]))
    split = prepare_split(config, task)
    paths = output_paths(config, task)
    atomic_write_json(paths["output"] / "config_snapshot.json", config)
    if not (paths["model"] / "config.json").is_file():
        raise FileNotFoundError(f"Missing trained model: {paths['model']}")
    trainer, _, validation_dataset, _ = build_trainer(
        runtime,
        config,
        task,
        split,
        paths["model"],
        paths["checkpoint"] / "evaluate_only",
        smoke=True,
    )
    prediction = trainer.predict(validation_dataset)
    details = prediction_details(prediction, split)
    save_metric_artifacts(paths["output"], details, "evaluate_only")
    return {
        "task": task,
        "model_path": str(paths["model"].relative_to(ROOT)),
        "metrics": details["summary"],
        "domain_minority": details["domain_minority"],
        "test_files_read": False,
        "submission_generated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-inputs", action="store_true")
    actions.add_argument("--check-baseline", action="store_true")
    actions.add_argument("--smoke", action="store_true")
    actions.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--task", default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    task = args.task or config["first_task"]
    if task not in config["allowed_tasks"]:
        raise ValueError(f"Task is not allowed by Transformer config: {task}")
    get_task_spec(task)
    mode = (
        "check_inputs"
        if args.check_inputs
        else "check_baseline"
        if args.check_baseline
        else "smoke"
        if args.smoke
        else "evaluate_only"
        if args.evaluate_only
        else "train"
    )
    log_path = setup_logging(config, task, mode)
    logging.info("mode=%s task=%s config=%s log=%s", mode, task, config_path, log_path)
    if args.check_inputs:
        result = check_inputs(config, config_path, task)
    elif args.check_baseline:
        result = run_baseline_check(config, task)
    elif args.smoke:
        result = run_smoke(config, task)
    elif args.evaluate_only:
        result = evaluate_only(config, task)
    else:
        result = run_training(config, task)
    logging.info("result=%s", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
