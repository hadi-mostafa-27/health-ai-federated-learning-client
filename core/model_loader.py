from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

from core.paths import resource_path


@dataclass
class LoadedModel:
    model: nn.Module
    checkpoint_path: str
    num_classes: int
    class_names: list[str]
    notebook_profile: str
    metadata: dict[str, Any]
    threshold: float
    checkpoint: dict[str, Any] | None = None


class NotebookAwareModelLoader:
    def __init__(self, models_dir: str | None = None, device: str = "cpu"):
        if models_dir:
            self.models_dir = Path(models_dir)
            if not self.models_dir.is_absolute():
                self.models_dir = resource_path(self.models_dir)
        else:
            self.models_dir = resource_path("models")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def build_model(self, pretrained: bool = False) -> nn.Module:
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 1)
        return model.to(self.device)

    def _extract_state_dict(self, checkpoint: Any) -> dict[str, Any]:
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                return checkpoint["state_dict"]
            if "model_state_dict" in checkpoint:
                return checkpoint["model_state_dict"]
            if all(isinstance(k, str) for k in checkpoint.keys()):
                return checkpoint
        raise RuntimeError("Unsupported checkpoint format.")

    def _normalize_state_dict_keys(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        if state_dict and all(str(k).startswith("module.") for k in state_dict.keys()):
            return {str(k)[7:]: v for k, v in state_dict.items()}
        return state_dict

    def _guess_default_model_path(self) -> Path:
        model_dirs = [self.models_dir]
        bundled_models_dir = resource_path("models")
        if bundled_models_dir not in model_dirs:
            model_dirs.append(bundled_models_dir)

        for model_dir in model_dirs:
            candidates = [
                model_dir / "fedprox_best.pt",
                model_dir / "fedavg_best.pt",
                model_dir / "centralized_best.pt",
                model_dir / "global" / "global_model.pt",
                model_dir / "global_model.pt",
            ]

            for path in candidates:
                if path.exists():
                    return path

            if model_dir.exists():
                found = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pth"))
                if found:
                    return found[0]

        raise FileNotFoundError(
            f"No default model file found in {self.models_dir} or bundled models. "
            f"Expected one of: fedprox_best.pt, fedavg_best.pt, centralized_best.pt"
        )

    def _infer_profile(self, model_path: Path) -> str:
        name = model_path.name.lower()
        if "fedprox" in name:
            return "fedprox"
        if "fedavg" in name:
            return "fedavg"
        if "central" in name:
            return "centralized"
        return "generic"

    def prepare_image(self, image_path: str | Path, loaded_model: LoadedModel | None = None) -> torch.Tensor:
        image = Image.open(image_path).convert("L")
        if loaded_model is not None:
            img_size = int(loaded_model.metadata.get("img_size", 224))
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            transform = self.transform
        tensor = transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def load_model(self, model_path: str | Path) -> LoadedModel:
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        try:
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Could not load checkpoint '{model_path}'. The file may be corrupted.") from exc

        state_dict = self._normalize_state_dict_keys(self._extract_state_dict(checkpoint))

        model = self.build_model()
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint architecture is incompatible with DenseNet121 binary head. "
                "Expected torchvision.densenet121 with classifier output size 1."
            ) from exc
        model.eval()

        notebook_profile = self._infer_profile(model_path)

        class_names = checkpoint.get("class_names", ["negative", "positive"]) if isinstance(checkpoint, dict) else ["negative", "positive"]
        threshold = float(checkpoint.get("threshold", 0.5)) if isinstance(checkpoint, dict) else 0.5

        metadata = {
            "checkpoint_path": str(model_path),
            "notebook_profile": notebook_profile,
            "model_name": checkpoint.get("model_name", "torchvision.densenet121") if isinstance(checkpoint, dict) else "torchvision.densenet121",
            "architecture": checkpoint.get("architecture", "torchvision.densenet121_binary") if isinstance(checkpoint, dict) else "torchvision.densenet121_binary",
            "class_names": class_names,
            "threshold": threshold,
            "img_size": checkpoint.get("img_size", 224) if isinstance(checkpoint, dict) else 224,
            "target_label": checkpoint.get("target_label", "positive") if isinstance(checkpoint, dict) else "positive",
            "mask_mode": checkpoint.get("mask_mode") if isinstance(checkpoint, dict) else None,
            "loaded_at": datetime.now(timezone.utc).isoformat(),
            "metrics": checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {},
            "training_config": checkpoint.get("training_config", {}) if isinstance(checkpoint, dict) else {},
        }

        return LoadedModel(
            model=model,
            checkpoint_path=str(model_path),
            num_classes=1,
            class_names=class_names,
            notebook_profile=notebook_profile,
            metadata=metadata,
            threshold=threshold,
            checkpoint=checkpoint if isinstance(checkpoint, dict) else None,
        )

    def load_default(self) -> LoadedModel:
        default_path = self._guess_default_model_path()
        return self.load_model(default_path)
