from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from core.model_loader import LoadedModel


class GradCAMEngine:
    DISCLAIMER = (
        "Grad-CAM is an explanatory visualization aid only. It is not clinical proof, "
        "a localization guarantee, or a substitute for radiologist review."
    )

    def generate_overlay(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        loaded_model: LoadedModel | None = None,
        target_class: int | None = None,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = Path(image_path)

        if loaded_model is None:
            return self._prototype_intensity_overlay(image_path, output_dir)

        model = loaded_model.model
        model.eval()
        device = next(model.parameters()).device
        img_size = int(loaded_model.metadata.get("img_size", 224))

        original = np.array(Image.open(image_path).convert("RGB"))
        image_tensor = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])(Image.fromarray(original)).unsqueeze(0).to(device)

        target_layer, target_layer_name = self._target_layer(model)
        activations: list[torch.Tensor] = []

        def forward_hook(_module, _inputs, output):
            # Torchvision DenseNet applies a functional in-place ReLU after
            # features.norm5. Returning a clone prevents autograd hook views
            # from being modified in-place during the classifier forward pass.
            cloned = output.clone()
            cloned.retain_grad()
            activations.append(cloned)
            return cloned

        handle_fwd = target_layer.register_forward_hook(forward_hook)
        try:
            model.zero_grad(set_to_none=True)
            logits = model(image_tensor).view(-1)
            prob = torch.sigmoid(logits)[0].item()
            threshold = float(loaded_model.threshold)
            predicted_class = 1 if prob >= threshold else 0
            class_to_explain = predicted_class if target_class is None else int(target_class)
            score = logits[0] if class_to_explain == 1 else -logits[0]
            score.backward()

            if not activations or activations[-1].grad is None:
                raise RuntimeError("Grad-CAM hook did not capture activations/gradients.")

            act = activations[-1].detach()[0]
            grad = activations[-1].grad.detach()[0]
            weights = grad.mean(dim=(1, 2), keepdim=True)
            cam = torch.relu((weights * act).sum(dim=0))
            cam = cam - cam.min()
            cam = cam / torch.clamp(cam.max(), min=1e-8)
            heat = cam.detach().cpu().numpy()
        finally:
            handle_fwd.remove()

        heat_resized = cv2.resize(heat, (original.shape[1], original.shape[0]))
        heat_uint8 = np.uint8(255 * heat_resized)
        heatmap_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        overlay_bgr = cv2.addWeighted(original_bgr, 0.58, heatmap_bgr, 0.42, 0)

        original_path = output_dir / f"{image_path.stem}_original.png"
        heat_path = output_dir / f"{image_path.stem}_heatmap.png"
        overlay_path = output_dir / f"{image_path.stem}_overlay.png"
        comparison_path = output_dir / f"{image_path.stem}_comparison.png"
        cv2.imwrite(str(original_path), original_bgr)
        cv2.imwrite(str(heat_path), heatmap_bgr)
        cv2.imwrite(str(overlay_path), overlay_bgr)
        comparison = np.concatenate([original_bgr, overlay_bgr], axis=1)
        cv2.imwrite(str(comparison_path), comparison)

        return {
            "original_path": str(original_path),
            "heatmap_path": str(heat_path),
            "overlay_path": str(overlay_path),
            "comparison_path": str(comparison_path),
            "target_layer": target_layer_name,
            "predicted_class": predicted_class,
            "target_class": class_to_explain,
            "probability_positive": float(prob),
            "threshold": threshold,
            "disclaimer": self.DISCLAIMER,
        }

    def _target_layer(self, model) -> tuple[Any, str]:
        if hasattr(model, "features") and hasattr(model.features, "norm5"):
            return model.features.norm5, "features.norm5"
        if hasattr(model, "features"):
            children = list(model.features.children())
            if children:
                return children[-1], "features[-1]"
        raise RuntimeError("Could not identify a Grad-CAM target layer for this model.")

    def _prototype_intensity_overlay(self, image_path: Path, output_dir: Path) -> dict[str, Any]:
        img = np.array(Image.open(image_path).convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        heat = cv2.GaussianBlur(gray, (0, 0), sigmaX=9)
        heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)

        original_path = output_dir / f"{image_path.stem}_original.png"
        heat_path = output_dir / f"{image_path.stem}_heatmap.png"
        overlay_path = output_dir / f"{image_path.stem}_overlay.png"
        comparison_path = output_dir / f"{image_path.stem}_comparison.png"
        cv2.imwrite(str(original_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(heat_path), heatmap)
        cv2.imwrite(str(overlay_path), overlay)
        cv2.imwrite(str(comparison_path), np.concatenate([cv2.cvtColor(img, cv2.COLOR_RGB2BGR), overlay], axis=1))
        return {
            "original_path": str(original_path),
            "heatmap_path": str(heat_path),
            "overlay_path": str(overlay_path),
            "comparison_path": str(comparison_path),
            "target_layer": "prototype_intensity_overlay",
            "disclaimer": self.DISCLAIMER,
            "warning": "No loaded PyTorch model was supplied; produced a non-Grad-CAM intensity overlay.",
        }
