from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from core.db import DatabaseManager


class FLClient:
    def __init__(self, server_url: str, api_token: str, db: DatabaseManager) -> None:
        self.server_url = server_url.rstrip("/")
        self.api_token = api_token
        self.db = db

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_token}"}

    def ping(self) -> dict[str, Any]:
        try:
            r = requests.get(f"{self.server_url}/health", timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            self.db.log("fl_server", f"Server unavailable: {exc}", "error")
            raise RuntimeError(f"Federated server is unavailable at {self.server_url}") from exc

    def current_round(self) -> dict[str, Any]:
        try:
            r = requests.get(f"{self.server_url}/fl/current-round", headers=self.headers, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            self.db.log("fl_round", f"Could not fetch current round: {exc}", "error")
            raise RuntimeError("Could not fetch the current FL round from the server.") from exc

    def register_client(
        self,
        hospital_id: str,
        hospital_name: str | None = None,
        samples: int | None = None,
        security_mode: str = "none",
    ) -> dict[str, Any]:
        payload = {
            "hospital_id": hospital_id,
            "hospital_name": hospital_name or hospital_id,
            "samples": samples,
            "security_mode": security_mode,
        }
        try:
            r = requests.post(f"{self.server_url}/fl/register-client", headers=self.headers, json=payload, timeout=15)
            r.raise_for_status()
            result = r.json()
            self.db.log("fl_register", f"Registered client {hospital_id} with server", "success")
            return result
        except requests.RequestException as exc:
            self.db.log("fl_register", f"Could not register client: {exc}", "error")
            raise RuntimeError("Could not register this hospital client with the FL server.") from exc

    def project_status(self) -> dict[str, Any]:
        try:
            r = requests.get(f"{self.server_url}/fl/project-status", headers=self.headers, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            self.db.log("fl_status", f"Could not fetch project status: {exc}", "error")
            raise RuntimeError("Could not fetch FL project status from the server.") from exc

    def fetch_global_model(self, destination_dir: str | Path, filename: str | None = None) -> dict[str, Any]:
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        try:
            r = requests.get(f"{self.server_url}/fl/global-model", headers=self.headers, timeout=120)
            r.raise_for_status()
        except requests.RequestException as exc:
            self.db.log("fl_fetch", f"Could not fetch LAN global model: {exc}", "error")
            raise RuntimeError("Could not download the LAN global model from the server.") from exc
        model_name = filename or self._filename_from_response(r, "global_model.pt")
        model_path = destination_dir / model_name
        model_path.write_bytes(r.content)
        self.db.log("fl_fetch", f"Fetched LAN global model {model_name}", "success")
        return {"filename": model_name, "saved_to": str(model_path), "bytes": len(r.content)}

    def _filename_from_response(self, response, fallback: str) -> str:
        content_disposition = response.headers.get("content-disposition", "")
        marker = "filename="
        if marker in content_disposition:
            return content_disposition.split(marker, 1)[1].strip('"')
        return fallback

    def upload_update(
        self,
        hospital_id: str,
        checkpoint_path: str | Path,
        round_number: int,
        num_samples: int,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing model update checkpoint: {checkpoint_path}")
        with checkpoint_path.open("rb") as f:
            files = {"file": (checkpoint_path.name, f, "application/octet-stream")}
            data = {
                "hospital_id": hospital_id,
                "round_number": str(round_number),
                "num_samples": str(num_samples),
                "metadata_json": json.dumps(metadata or {}),
            }
            try:
                r = requests.post(f"{self.server_url}/fl/upload-update", headers=self.headers, files=files, data=data, timeout=180)
                r.raise_for_status()
            except requests.RequestException as exc:
                self.db.log("fl_upload", f"Could not upload model update: {exc}", "error")
                raise RuntimeError("Could not upload the local model update to the FL server.") from exc
        payload = r.json()
        self.db.log("fl_upload", f"Uploaded round {round_number} update for {hospital_id}", "success")
        return payload

    def aggregate_round(
        self,
        round_number: int | None = None,
        min_clients: int | None = None,
        security_mode: str | None = None,
        aggregation_algorithm: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "round_number": round_number,
            "min_clients": min_clients,
            "security_mode": security_mode,
            "aggregation_algorithm": aggregation_algorithm,
        }
        payload = {key: value for key, value in payload.items() if value is not None}
        try:
            r = requests.post(f"{self.server_url}/fl/aggregate-round", headers=self.headers, json=payload, timeout=300)
            r.raise_for_status()
            result = r.json()
            self.db.log("fl_aggregate", f"Requested aggregation for round {result.get('round_number')}", "success")
            return result
        except requests.RequestException as exc:
            self.db.log("fl_aggregate", f"Could not aggregate round: {exc}", "error")
            raise RuntimeError("Could not aggregate the FL round on the server.") from exc

    def fetch_latest_model(self, destination_dir: str | Path) -> dict[str, Any]:
        try:
            info = requests.get(f"{self.server_url}/models/latest", headers=self.headers, timeout=15)
            info.raise_for_status()
            payload = info.json()
            destination_dir = Path(destination_dir)
            destination_dir.mkdir(parents=True, exist_ok=True)
            model_name = payload.get("filename", "global_model.pt")
            model_path = destination_dir / model_name
            content = requests.get(f"{self.server_url}{payload['download_url']}", headers=self.headers, timeout=60)
            content.raise_for_status()
            model_path.write_bytes(content.content)
            self.db.log("fl_fetch", f"Fetched global model {model_name}", "success")
            return {**payload, "saved_to": str(model_path)}
        except requests.RequestException as exc:
            self.db.log("fl_fetch", f"Could not fetch global model: {exc}", "error")
            raise RuntimeError("Could not download the latest global model from the server.") from exc

    def send_model_update(self, hospital_id: str, checkpoint_path: str | Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing model update checkpoint: {checkpoint_path}")
        with checkpoint_path.open("rb") as f:
            files = {"file": (checkpoint_path.name, f, "application/octet-stream")}
            data = {"hospital_id": hospital_id, "metadata_json": json.dumps(metadata or {})}
            try:
                r = requests.post(f"{self.server_url}/fl/send-update", headers=self.headers, files=files, data=data, timeout=120)
                r.raise_for_status()
            except requests.RequestException as exc:
                self.db.log("fl_send", f"Could not send local update: {exc}", "error")
                raise RuntimeError("Could not send the local model update to the server.") from exc
        payload = r.json()
        self.db.log("fl_send", f"Sent local update {checkpoint_path.name}", "success")
        return payload
