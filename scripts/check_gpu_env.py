"""Read-only AutoDL environment check for the Phase 04 Transformer baseline."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/transformer_baseline.json"


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-model", action="store_true")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow --check-model to contact the Hub; default is local cache only.",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = config["environment"]
    errors: list[str] = []
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ("torch", "transformers", "accelerate", "sentencepiece", "safetensors")
        },
    }

    try:
        import torch

        result["cuda_available"] = bool(torch.cuda.is_available())
        result["torch_cuda"] = torch.version.cuda
        result["cudnn"] = torch.backends.cudnn.version()
        result["bf16_supported"] = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            result["gpu"] = properties.name
            result["gpu_total_memory_gib"] = round(properties.total_memory / (1024**3), 3)
    except Exception as error:
        errors.append(f"torch import/check failed: {type(error).__name__}: {error}")
        result["cuda_available"] = False
        result["bf16_supported"] = False

    expected_packages = {
        "torch": expected["torch"],
        "transformers": expected["transformers"],
        "accelerate": expected["accelerate"],
        "sentencepiece": expected["sentencepiece"],
        "safetensors": expected["safetensors"],
    }
    for name, wanted in expected_packages.items():
        observed = result["packages"].get(name)
        if observed != wanted:
            errors.append(f"{name}: expected {wanted}, found {observed}")
    if platform.python_version() != expected["python"]:
        errors.append(f"python: expected {expected['python']}, found {platform.python_version()}")
    if expected.get("cuda_required") and not result.get("cuda_available"):
        errors.append("CUDA is required but unavailable")
    if expected.get("bf16_required") and not result.get("bf16_supported"):
        errors.append("bf16 is required but unsupported")

    if args.check_model:
        try:
            from transformers import AutoConfig, AutoTokenizer

            local_only = not args.allow_download
            model_name = config["model_name"]
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
            model_config = AutoConfig.from_pretrained(model_name, local_files_only=local_only)
            result["model_check"] = {
                "model_name": model_name,
                "local_files_only": local_only,
                "tokenizer_class": type(tokenizer).__name__,
                "model_type": model_config.model_type,
                "max_position_embeddings": model_config.max_position_embeddings,
            }
        except Exception as error:
            errors.append(f"model tokenizer/config check failed: {type(error).__name__}: {error}")

    result["passed"] = not errors
    result["errors"] = errors
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
