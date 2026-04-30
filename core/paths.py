from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


APP_NAME = "HospitalFLSystem"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def project_root() -> Path:
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent.parent


def resource_path(relative_path: str | os.PathLike) -> Path:
    """Return a bundled resource path in PyInstaller, or repo path in development."""
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return project_root() / path


def app_data_dir() -> Path:
    if not is_frozen():
        return project_root()
    base = os.environ.get("APPDATA")
    root = Path(base) if base else Path.home() / "AppData" / "Roaming"
    path = root / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def writable_path(relative_path: str | os.PathLike) -> Path:
    """Return a user-writable path for runtime files such as DBs and reports."""
    path = Path(relative_path)
    if path.is_absolute():
        return path
    resolved = app_data_dir() / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_writable_copy(relative_path: str | os.PathLike, default_content: str = "{}") -> Path:
    """Copy a bundled template to AppData on first run, or create a default file."""
    destination = writable_path(relative_path)
    if destination.exists():
        return destination

    source = resource_path(relative_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.exists() and source.is_file():
        shutil.copy2(source, destination)
    else:
        destination.write_text(default_content, encoding="utf-8")
    return destination
