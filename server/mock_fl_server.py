from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from torchvision import models

from core.secure_aggregation import (
    SECURE_AGG_SIM_DISCLAIMER,
    aggregate_masked_weighted_state_dicts,
    communication_size_bytes,
)

ROOT = Path(__file__).resolve().parent
STORE = ROOT / "storage"
MODELS = STORE / "models"
UPDATES = STORE / "updates"
MODELS.mkdir(parents=True, exist_ok=True)
UPDATES.mkdir(parents=True, exist_ok=True)

DEFAULT_SECURITY_MODE = "none"
DEFAULT_MIN_CLIENTS = 2
DEFAULT_ALGORITHM = "FedAvg"

app = FastAPI(title="Federated Aggregation Prototype Server")

SERVER_STATE: dict[str, Any] = {
    "current_round": 1,
    "global_model_path": None,
    "clients": {},
    "updates": {},
    "aggregation_history": [],
    "security_mode": DEFAULT_SECURITY_MODE,
    "min_clients": DEFAULT_MIN_CLIENTS,
    "aggregation_algorithm": DEFAULT_ALGORITHM,
    "mask_std": 0.01,
}


class ClientRegistration(BaseModel):
    hospital_id: str
    hospital_name: str | None = None
    samples: int | None = None
    security_mode: str = Field(default=DEFAULT_SECURITY_MODE)


class AggregateRequest(BaseModel):
    round_number: int | None = None
    min_clients: int | None = None
    security_mode: str | None = None
    aggregation_algorithm: str | None = None


def _build_densenet_binary() -> nn.Module:
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model.cpu()


def _checkpoint_payload(model: nn.Module, round_number: int, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "threshold": 0.5,
        "img_size": 224,
        "model_name": "torchvision.densenet121",
        "architecture": "torchvision.densenet121_binary",
        "class_names": ["NORMAL", "PNEUMONIA"],
        "round_number": round_number,
        "metadata": metadata or {},
    }


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if all(isinstance(k, str) for k in checkpoint):
            return checkpoint
    raise HTTPException(status_code=400, detail="Unsupported checkpoint/update format.")


def _latest_model_file() -> Path | None:
    files = sorted(MODELS.glob("global_round_*.pt"))
    if files:
        return files[-1]
    files = sorted(MODELS.glob("*.pt")) + sorted(MODELS.glob("*.pth"))
    return files[-1] if files else None


def _ensure_global_model() -> Path:
    configured = SERVER_STATE.get("global_model_path")
    if configured and Path(configured).exists():
        return Path(configured)

    latest = _latest_model_file()
    if latest:
        try:
            torch.load(latest, map_location="cpu", weights_only=False)
            SERVER_STATE["global_model_path"] = str(latest)
            return latest
        except Exception:
            pass

    path = MODELS / "global_round_001.pt"
    torch.save(_checkpoint_payload(_build_densenet_binary(), 1, {"source": "server_initialized"}), path)
    SERVER_STATE["global_model_path"] = str(path)
    return path


def _load_global_state() -> dict[str, torch.Tensor]:
    path = _ensure_global_model()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return _extract_state_dict(checkpoint)


def _validate_update_shapes(update_state: dict[str, torch.Tensor]) -> None:
    expected = _load_global_state()
    expected_keys = set(expected)
    update_keys = set(update_state)
    if expected_keys != update_keys:
        missing = sorted(expected_keys - update_keys)[:5]
        unexpected = sorted(update_keys - expected_keys)[:5]
        raise HTTPException(
            status_code=400,
            detail={"message": "Update parameter names do not match global model.", "missing": missing, "unexpected": unexpected},
        )
    for key, expected_tensor in expected.items():
        if tuple(update_state[key].shape) != tuple(expected_tensor.shape):
            raise HTTPException(
                status_code=400,
                detail=f"Shape mismatch for {key}: expected {tuple(expected_tensor.shape)}, got {tuple(update_state[key].shape)}",
            )


def _round_updates(round_number: int) -> dict[str, dict[str, Any]]:
    return SERVER_STATE["updates"].setdefault(str(round_number), {})


def _registered_client_ids() -> list[str]:
    return sorted(SERVER_STATE["clients"].keys())


def _aggregate_plain(updates: dict[str, dict[str, Any]]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    total_samples = sum(int(record["num_samples"]) for record in updates.values())
    if total_samples <= 0:
        raise HTTPException(status_code=400, detail="Cannot aggregate updates with zero total samples.")

    states = {}
    for hospital_id, record in updates.items():
        checkpoint = torch.load(record["path"], map_location="cpu", weights_only=False)
        states[hospital_id] = _extract_state_dict(checkpoint)

    first_id = sorted(states)[0]
    aggregated: dict[str, torch.Tensor] = {}
    for key, first_tensor in states[first_id].items():
        if torch.is_floating_point(first_tensor):
            value = torch.zeros_like(first_tensor.detach().cpu().float())
            for hospital_id, state in states.items():
                weight = int(updates[hospital_id]["num_samples"]) / total_samples
                value = value + state[key].detach().cpu().float() * weight
            aggregated[key] = value.to(first_tensor.dtype)
        else:
            aggregated[key] = first_tensor.detach().cpu().clone()

    metadata = {
        "security_mode": "none",
        "total_samples": total_samples,
        "completed_clients": sorted(updates),
    }
    return aggregated, metadata


def _aggregate_secure(round_number: int, updates: dict[str, dict[str, Any]]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    masked_updates = {}
    sample_counts = {}
    for hospital_id, record in updates.items():
        checkpoint = torch.load(record["path"], map_location="cpu", weights_only=False)
        masked_updates[hospital_id] = _extract_state_dict(checkpoint)
        sample_counts[hospital_id] = int(record["num_samples"])
    client_cohorts = {
        hospital_id: record.get("metadata", {}).get("cohort_ids", _registered_client_ids())
        for hospital_id, record in updates.items()
    }
    return aggregate_masked_weighted_state_dicts(
        masked_updates=masked_updates,
        sample_counts=sample_counts,
        cohort_ids=_registered_client_ids(),
        round_number=round_number,
        mask_std=float(SERVER_STATE.get("mask_std", 0.01)),
        client_cohorts=client_cohorts,
    )


@app.get("/health")
def health():
    return {"status": "ok", "server": "Federated Aggregation Prototype Server"}


@app.post("/fl/register-client")
def register_client(payload: ClientRegistration):
    if payload.security_mode not in {"none", "secure_agg_sim", "he_demo"}:
        raise HTTPException(status_code=400, detail="security_mode must be none, secure_agg_sim, or he_demo.")
    SERVER_STATE["clients"][payload.hospital_id] = {
        "hospital_id": payload.hospital_id,
        "hospital_name": payload.hospital_name or payload.hospital_id,
        "samples": payload.samples,
        "security_mode": payload.security_mode,
        "registered_at": time.time(),
        "status": "registered",
    }
    if payload.security_mode != "none":
        SERVER_STATE["security_mode"] = payload.security_mode
    _ensure_global_model()
    return {
        "message": "client registered",
        "hospital_id": payload.hospital_id,
        "round_number": SERVER_STATE["current_round"],
        "security_mode": SERVER_STATE["security_mode"],
        "aggregation_algorithm": SERVER_STATE["aggregation_algorithm"],
        "registered_clients": _registered_client_ids(),
        "min_clients": SERVER_STATE["min_clients"],
        "secure_agg_disclaimer": SECURE_AGG_SIM_DISCLAIMER if SERVER_STATE["security_mode"] == "secure_agg_sim" else None,
    }


@app.get("/fl/global-model")
def global_model():
    path = _ensure_global_model()
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


@app.post("/fl/upload-update")
async def upload_update(
    hospital_id: str = Form(...),
    round_number: int = Form(...),
    num_samples: int = Form(...),
    metadata_json: str = Form("{}"),
    file: UploadFile = File(...),
):
    if hospital_id not in SERVER_STATE["clients"]:
        raise HTTPException(status_code=403, detail=f"Client {hospital_id} is not registered.")
    if round_number != int(SERVER_STATE["current_round"]):
        raise HTTPException(status_code=409, detail=f"Server is on round {SERVER_STATE['current_round']}, got update for round {round_number}.")
    if num_samples <= 0:
        raise HTTPException(status_code=400, detail="num_samples must be positive.")

    metadata = json.loads(metadata_json or "{}")
    security_mode = metadata.get("security_mode", SERVER_STATE["security_mode"])
    if SERVER_STATE["security_mode"] != "none" and security_mode != SERVER_STATE["security_mode"]:
        raise HTTPException(status_code=400, detail=f"Client security_mode {security_mode} does not match server mode {SERVER_STATE['security_mode']}.")
    if SERVER_STATE["security_mode"] == "none" and security_mode not in {"none", "he_demo"}:
        raise HTTPException(status_code=400, detail=f"Server is in none mode but received {security_mode}. Register clients with that mode first.")

    data = await file.read()
    round_dir = UPDATES / f"round_{round_number:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    save_path = round_dir / f"{hospital_id}_{file.filename}"
    save_path.write_bytes(data)

    try:
        checkpoint = torch.load(save_path, map_location="cpu", weights_only=False)
        update_state = _extract_state_dict(checkpoint)
        _validate_update_shapes(update_state)
    except HTTPException:
        save_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid update checkpoint: {exc}") from exc

    _round_updates(round_number)[hospital_id] = {
        "hospital_id": hospital_id,
        "round_number": round_number,
        "num_samples": int(num_samples),
        "metadata": metadata,
        "path": str(save_path),
        "bytes": len(data),
        "uploaded_at": time.time(),
        "security_mode": security_mode,
    }
    return {
        "message": "update accepted",
        "hospital_id": hospital_id,
        "round_number": round_number,
        "num_samples": num_samples,
        "completed_clients": sorted(_round_updates(round_number)),
        "min_clients": SERVER_STATE["min_clients"],
        "can_aggregate": len(_round_updates(round_number)) >= int(SERVER_STATE["min_clients"]),
        "communication_size_bytes": len(data),
    }


@app.get("/fl/project-status")
def project_status():
    round_number = int(SERVER_STATE["current_round"])
    updates = _round_updates(round_number)
    return {
        "current_round": round_number,
        "aggregation_algorithm": SERVER_STATE["aggregation_algorithm"],
        "security_mode": SERVER_STATE["security_mode"],
        "min_clients": SERVER_STATE["min_clients"],
        "registered_clients": list(SERVER_STATE["clients"].values()),
        "registered_client_ids": _registered_client_ids(),
        "completed_clients": sorted(updates),
        "dropped_clients": sorted(set(_registered_client_ids()) - set(updates)),
        "can_aggregate": len(updates) >= int(SERVER_STATE["min_clients"]),
        "aggregation_history": SERVER_STATE["aggregation_history"][-10:],
        "disclaimer": "Prototype coordinator; not production-secure and not a clinical system.",
    }


@app.post("/fl/aggregate-round")
def aggregate_round(payload: AggregateRequest | None = None):
    payload = payload or AggregateRequest()
    round_number = int(payload.round_number or SERVER_STATE["current_round"])
    if round_number != int(SERVER_STATE["current_round"]):
        raise HTTPException(status_code=409, detail=f"Only current round {SERVER_STATE['current_round']} can be aggregated.")

    min_clients = int(payload.min_clients or SERVER_STATE["min_clients"])
    security_mode = payload.security_mode or SERVER_STATE["security_mode"]
    aggregation_algorithm = payload.aggregation_algorithm or SERVER_STATE["aggregation_algorithm"]
    updates = _round_updates(round_number)
    if len(updates) < min_clients:
        raise HTTPException(status_code=409, detail=f"Need at least {min_clients} updates; received {len(updates)}.")

    start = time.perf_counter()
    if security_mode == "secure_agg_sim":
        aggregated_state, agg_metadata = _aggregate_secure(round_number, updates)
    else:
        aggregated_state, agg_metadata = _aggregate_plain(updates)
        if security_mode == "he_demo":
            agg_metadata["he_demo_note"] = "Full model weights were aggregated normally; HE demo is toy-only."
    aggregation_time = time.perf_counter() - start

    model = _build_densenet_binary()
    model.load_state_dict(aggregated_state, strict=True)
    next_round = round_number + 1
    output_path = MODELS / f"global_round_{next_round:03d}.pt"
    record = {
        "round_number": round_number,
        "next_round": next_round,
        "aggregation_algorithm": aggregation_algorithm,
        "security_mode": security_mode,
        "number_of_clients": len(_registered_client_ids()),
        "number_of_completed_clients": len(updates),
        "completed_clients": sorted(updates),
        "dropped_clients": sorted(set(_registered_client_ids()) - set(updates)),
        "aggregation_time_seconds": aggregation_time,
        "communication_size_bytes": sum(int(record["bytes"]) for record in updates.values()),
        "state_dict_size_bytes": communication_size_bytes(aggregated_state),
        "metadata": agg_metadata,
    }
    torch.save(_checkpoint_payload(model, next_round, record), output_path)
    SERVER_STATE["global_model_path"] = str(output_path)
    SERVER_STATE["aggregation_history"].append(record)
    SERVER_STATE["current_round"] = next_round
    SERVER_STATE["updates"].setdefault(str(next_round), {})
    return {"message": "round aggregated", "global_model": str(output_path), **record}


@app.get("/fl/current-round")
def current_round():
    status = project_status()
    return {
        "round": status["current_round"],
        "version": Path(_ensure_global_model()).stem,
        "status": "in_progress",
        "aggregation_algorithm": status["aggregation_algorithm"],
        "security_mode": status["security_mode"],
        "registered_clients": status["registered_client_ids"],
        "completed_clients": status["completed_clients"],
        "note": "Prototype coordinator; secure_agg_sim is a simulation and formal privacy is not implemented.",
    }


@app.get("/models/latest")
def latest_model():
    latest = _ensure_global_model()
    return {
        "version": latest.stem,
        "filename": latest.name,
        "download_url": f"/files/{latest.name}",
    }


@app.get("/files/{filename}")
def download_file(filename: str):
    file_path = MODELS / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found.")
    return FileResponse(file_path)


@app.post("/fl/send-update")
async def send_update(hospital_id: str = Form(...), metadata_json: str = Form("{}"), file: UploadFile = File(...)):
    metadata = json.loads(metadata_json or "{}")
    data = await file.read()
    save_path = UPDATES / f"{hospital_id}_{file.filename}"
    save_path.write_bytes(data)
    return {
        "message": "legacy update received",
        "hospital_id": hospital_id,
        "saved_to": str(save_path),
        "num_samples": metadata.get("num_samples"),
        "metadata": metadata,
        "privacy_note": "Legacy endpoint stores updates only; use /fl/upload-update for LAN aggregation.",
    }
