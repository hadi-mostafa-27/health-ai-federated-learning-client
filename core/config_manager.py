from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.paths import ensure_writable_copy, is_frozen, resource_path, writable_path


class ConfigManager:
    def __init__(self, config_path: str = "config/app_config.json") -> None:
        if Path(config_path).is_absolute():
            self.config_path = Path(config_path)
        else:
            self.config_path = ensure_writable_copy(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.config_path.exists():
            self.config_path.write_text("{}", encoding="utf-8")
        self._data = json.loads(self.config_path.read_text(encoding="utf-8") or "{}")

    @property
    def database_path(self) -> str:
        return self.get("database_path", "database/hospital_client.db")

    def get(self, key: str, default: Any = None) -> Any:
        value = self._data.get(key, default)
        if is_frozen() and isinstance(value, str):
            if key == "database_path":
                return str(writable_path(value))
            if key in {"models_dir", "reports_dir", "logs_dir", "predictions_dir", "visualizations_dir"}:
                path = writable_path(value)
                path.mkdir(parents=True, exist_ok=True)
                return str(path)
            if key == "dataset_dir":
                writable = writable_path(value)
                bundled = resource_path(value)
                if writable.exists() and any(writable.iterdir()):
                    return str(writable)
                if bundled.exists():
                    return str(bundled)
                writable.mkdir(parents=True, exist_ok=True)
                return str(writable)
        return value

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def update(self, mapping: dict[str, Any]) -> None:
        self._data.update(mapping)
        self.save()

    def save(self) -> None:
        self.config_path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
