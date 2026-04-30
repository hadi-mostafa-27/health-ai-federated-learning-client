from __future__ import annotations

import json
import os
import platform
import random
import sys
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch


def create_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def collect_environment_info() -> dict[str, Any]:
    packages = {}
    for package in ["torch", "torchvision", "numpy", "pandas", "scikit-learn", "PySide6"]:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "torch_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "packages": packages,
        "cwd": os.getcwd(),
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


def export_experiment_config(
    config: dict[str, Any],
    output_dir: str | Path,
    run_id: str | None = None,
) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or create_run_id()
    path = output_dir / f"{run_id}_config.json"
    payload = {
        "run_id": run_id,
        "config": config,
        "environment": collect_environment_info(),
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(path)
