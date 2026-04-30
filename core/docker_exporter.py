from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.db import DatabaseManager
from core.paths import writable_path


class DockerExportError(RuntimeError):
    pass


def _row_to_dict(row) -> dict[str, Any]:
    return dict(row) if row is not None else {}


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "hospital"


class DockerPackageExporter:
    """Create prototype Docker deployment folders for completed FL projects."""

    def __init__(self, db: DatabaseManager, export_root: str | Path | None = None) -> None:
        self.db = db
        self.export_root = Path(export_root) if export_root else writable_path("exports/docker")
        self.export_root.mkdir(parents=True, exist_ok=True)

    def export_for_hospital(
        self,
        *,
        project_id: int,
        hospital_id: str,
        requester_role: str = "hospital",
    ) -> dict[str, str]:
        project = self.db.get_fl_project(project_id)
        if not project:
            raise DockerExportError("Project was not found.")

        status = str(project["status"] or "").lower()
        if status != "completed":
            raise DockerExportError("Docker export is available only after the FL project is completed.")

        membership = self.db.get_project_membership(project_id, hospital_id)
        joined = membership is not None and "joined" in str(membership["status"]).lower()
        if requester_role != "admin" and not joined:
            raise DockerExportError("This hospital can export only projects it joined.")
        if requester_role == "admin" and not joined:
            raise DockerExportError("Admin can export packages only for joined project hospitals.")

        hospital = self.db.get_hospital(hospital_id)
        hospital_name = (
            str(membership["hospital_name"])
            if membership and membership["hospital_name"]
            else str(hospital["hospital_name"] if hospital else hospital_id)
        )

        final_metrics = self._project_metrics(project_id, project)
        project_metadata = self._project_metadata(project, hospital_id, hospital_name)
        settings = self._fl_settings(project)

        folder = self.export_root / str(project_id) / _safe_name(hospital_name)
        model_dir = folder / "model"
        if folder.exists():
            shutil.rmtree(folder)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_artifact = self._copy_or_create_model_artifact(project, model_dir, final_metrics)
        self._write_json(folder / "project_metadata.json", project_metadata)
        self._write_json(folder / "fl_settings.json", settings)
        self._write_json(folder / "final_metrics.json", final_metrics)
        self._write_json(
            folder / "hospital_metadata.json",
            {
                "hospital_id": hospital_id,
                "hospital_name": hospital_name,
                "membership_status": str(membership["status"]) if membership else "unknown",
                "current_node_status": str(hospital["node_status"]) if hospital else "unknown",
            },
        )
        self._write_text(folder / "Dockerfile", self._dockerfile())
        self._write_text(folder / "README_DEPLOY.md", self._readme(project_metadata, model_artifact))
        self._write_text(folder / "run_container.bat", self._run_script(project_id, hospital_id))
        self._write_text(folder / "serve_model.py", self._serve_script())

        zip_path = folder.parent / f"{_safe_name(hospital_name)}_docker_package.zip"
        if zip_path.exists():
            zip_path.unlink()
        self._zip_folder(folder, zip_path)

        metadata = {
            "project_id": project_id,
            "hospital_id": hospital_id,
            "export_folder": str(folder),
            "zip_path": str(zip_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_artifact": model_artifact,
        }
        self.db.record_docker_export(
            project_id=project_id,
            hospital_id=hospital_id,
            hospital_name=hospital_name,
            export_folder=str(folder),
            zip_path=str(zip_path),
            metadata=metadata,
        )
        self.db.log(
            "docker_export",
            f"Docker package exported for Project #{project_id} / {hospital_name}",
            "success",
        )
        return metadata

    def _project_metrics(self, project_id: int, project) -> dict[str, Any]:
        if "final_metrics_json" in project.keys() and project["final_metrics_json"]:
            try:
                return json.loads(project["final_metrics_json"])
            except json.JSONDecodeError:
                pass
        return self.db.get_latest_project_metrics(project_id)

    def _project_metadata(self, project, hospital_id: str, hospital_name: str) -> dict[str, Any]:
        details = {}
        if "details_json" in project.keys() and project["details_json"]:
            try:
                details = json.loads(project["details_json"])
            except json.JSONDecodeError:
                details = {}
        return {
            "project_id": int(project["id"]),
            "project_name": project["project_name"],
            "status": project["status"],
            "completed_at": project["completed_at"] if "completed_at" in project.keys() else None,
            "participating_hospital_id": hospital_id,
            "participating_hospital_name": hospital_name,
            "created_by": details.get("created_by_display_name", "Administration Department"),
            "prototype_notice": "Academic prototype package. Not for clinical or production use.",
        }

    def _fl_settings(self, project) -> dict[str, Any]:
        details = {}
        if "details_json" in project.keys() and project["details_json"]:
            try:
                details = json.loads(project["details_json"])
            except json.JSONDecodeError:
                details = {}
        return {
            "disease_target": project["disease_target"],
            "model_backbone": project["model_backbone"],
            "fl_algorithm": project["fl_algorithm"],
            "total_rounds": project["total_rounds"],
            "local_epochs": project["local_epochs"],
            "batch_size": project["batch_size"],
            "learning_rate": project["learning_rate"],
            "participation_fraction": project["participation_fraction"],
            "stop_accuracy": project["stop_accuracy"],
            "non_iid_strategy": details.get("non_iid_strategy"),
            "imbalance_severity": details.get("imbalance_severity"),
        }

    def _copy_or_create_model_artifact(self, project, model_dir: Path, metrics: dict[str, Any]) -> str:
        final_model_path = project["final_model_path"] if "final_model_path" in project.keys() else None
        if final_model_path and Path(final_model_path).exists():
            destination = model_dir / Path(final_model_path).name
            shutil.copy2(final_model_path, destination)
            return str(destination.name)

        placeholder = model_dir / "simulated_model_artifact.json"
        self._write_json(
            placeholder,
            {
                "artifact_type": "simulated_model_placeholder",
                "reason": "No trained model file was available when the package was exported.",
                "project_metrics": metrics,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return placeholder.name

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def _write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _zip_folder(self, folder: Path, zip_path: Path) -> None:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in folder.rglob("*"):
                if file_path.is_file():
                    archive.write(file_path, file_path.relative_to(folder.parent))

    def _dockerfile(self) -> str:
        return """FROM python:3.11-slim
WORKDIR /app
COPY . /app
CMD ["python", "serve_model.py"]
"""

    def _readme(self, metadata: dict[str, Any], model_artifact: str) -> str:
        return f"""# Hospital FL Docker Deployment Package

Project: {metadata.get("project_name")}
Hospital: {metadata.get("participating_hospital_name")}

This is an academic prototype deployment package. It is not a clinical decision
system and is not production-secure.

## Contents

- `model/{model_artifact}`: final trained model file, or a simulated placeholder
- `project_metadata.json`: project and hospital metadata
- `fl_settings.json`: selected federated learning settings
- `final_metrics.json`: final available results
- `Dockerfile`: basic prototype container definition
- `serve_model.py`: minimal container entry point

## Build

```bat
docker build -t hospital-fl-project-{metadata.get("project_id")} .
```

## Run

```bat
docker run --rm hospital-fl-project-{metadata.get("project_id")}
```
"""

    def _run_script(self, project_id: int, hospital_id: str) -> str:
        tag = f"hospital-fl-project-{project_id}-{_safe_name(hospital_id).lower()}"
        return f"""@echo off
docker build -t {tag} .
docker run --rm {tag}
"""

    def _serve_script(self) -> str:
        return """import json
from pathlib import Path


def load_json(name):
    path = Path(name)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


metadata = load_json("project_metadata.json")
metrics = load_json("final_metrics.json")

print("HospitalFLSystem prototype deployment package")
print(f"Project: {metadata.get('project_name')}")
print(f"Hospital: {metadata.get('participating_hospital_name')}")
print("This container is a prototype wrapper, not a clinical service.")
print("Final metrics:")
print(json.dumps(metrics, indent=2))
"""
