from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
import json

import torch

from core.db import DatabaseManager
from core.model_loader import LoadedModel, NotebookAwareModelLoader
from core.paths import writable_path


@dataclass
class PredictionResult:
    predicted_label: str
    confidence: float
    elapsed_seconds: float
    details: dict[str, Any]


class InferenceEngine:
    def __init__(self, loader: NotebookAwareModelLoader, db: DatabaseManager) -> None:
        self.loader = loader
        self.db = db

    def predict(self, image_path: str, loaded: LoadedModel) -> PredictionResult:
        x = self.loader.prepare_image(image_path, loaded)

        start = perf_counter()
        with torch.no_grad():
            logits = loaded.model(x)
            prob = torch.sigmoid(logits).view(-1)[0].item()

        elapsed = perf_counter() - start

        threshold = float(getattr(loaded, "threshold", 0.5))
        pred_idx = 1 if prob >= threshold else 0

        class_names = loaded.class_names if loaded.class_names else ["negative", "positive"]
        predicted_label = class_names[pred_idx]

        return PredictionResult(
            predicted_label=predicted_label,
            confidence=float(prob),
            elapsed_seconds=elapsed,
            details={
                "predicted_index": pred_idx,
                "probability_positive": float(prob),
                "threshold": threshold,
                "checkpoint_path": loaded.checkpoint_path,
                "notebook_profile": loaded.notebook_profile,
                "metadata": loaded.metadata,
            },
        )

    def persist_prediction(self, image_path: str, result: PredictionResult, model_version: str) -> int:
        query = """
            INSERT INTO predictions (image_path, true_label, predicted_label, confidence, model_version, inference_time, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        import json
        from datetime import datetime
        from pathlib import Path

        details_json = json.dumps(result.details)

        # IMPORTANT: db.execute returns int (row_id)
        row_id = self.db.execute(
            query,
            (
                image_path,
                None,
                result.predicted_label,
                result.confidence,
                model_version,
                result.elapsed_seconds,
                details_json,
            ),
        )

        # Save JSON file
        pred_dir = writable_path("data/predictions")
        pred_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = pred_dir / f"prediction_{timestamp}.json"

        payload = {
            "image_path": image_path,
            "predicted_label": result.predicted_label,
            "confidence": result.confidence,
            "model_version": model_version,
            "elapsed_seconds": result.elapsed_seconds,
            "details": result.details,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return int(row_id)
