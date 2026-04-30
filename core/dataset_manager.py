from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from core.db import DatabaseManager

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
EXPECTED_CLASSES = ("NORMAL", "PNEUMONIA")


@dataclass
class DatasetSummary:
    dataset_name: str
    dataset_path: str
    label_csv_path: str | None
    num_samples: int
    num_classes: int
    train_count: int
    val_count: int
    test_count: int
    class_distribution: dict[str, int] | None = None
    split_distribution: dict[str, dict[str, int]] | None = None
    imbalance_ratio: float | None = None
    warnings: list[str] | None = None
    invalid_images: list[str] | None = None
    random_seed: int = 42


class DatasetManager:
    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def scan_folder(self, dataset_path: str | Path) -> pd.DataFrame:
        df, _, _ = self._scan_folder_with_quality(dataset_path)
        return df

    def _scan_folder_with_quality(
        self,
        dataset_path: str | Path,
        validate_images: bool = True,
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset folder does not exist: {dataset_path}")

        rows = []
        warnings = []
        invalid_images: list[str] = []
        class_dirs = [p for p in dataset_path.iterdir() if p.is_dir()]
        if class_dirs:
            present = {p.name.upper(): p for p in class_dirs}
            missing = [cls for cls in EXPECTED_CLASSES if cls not in present]
            if missing:
                warnings.append(
                    "Missing expected class folder(s): "
                    + ", ".join(missing)
                    + ". Expected NORMAL/ and PNEUMONIA/."
                )
            for class_dir in class_dirs:
                for img in class_dir.rglob("*"):
                    if img.suffix.lower() not in IMAGE_EXTS:
                        continue
                    if validate_images and not self._is_valid_image(img):
                        invalid_images.append(str(img))
                        continue
                    rows.append({"file_path": str(img), "label": class_dir.name.upper()})
        else:
            for img in dataset_path.rglob("*"):
                if img.suffix.lower() not in IMAGE_EXTS:
                    continue
                if validate_images and not self._is_valid_image(img):
                    invalid_images.append(str(img))
                    continue
                rows.append({"file_path": str(img), "label": "UNLABELED"})
            warnings.append("No class folders found. Expected NORMAL/ and PNEUMONIA/.")
        return pd.DataFrame(rows), warnings, invalid_images

    def _is_valid_image(self, image_path: Path) -> bool:
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def load_csv(self, csv_path: str | Path) -> pd.DataFrame:
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        return df

    def register_dataset(self, dataset_name: str, dataset_path: str, label_csv_path: str | None = None,
                         train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15,
                         random_seed: int = 42, validate_images: bool = True) -> DatasetSummary:
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1")

        warnings: list[str] = []
        invalid_images: list[str] = []
        if label_csv_path:
            df = self.load_csv(label_csv_path)
            if "file_path" not in df.columns:
                if {"path", "label"}.issubset(df.columns):
                    df = df.rename(columns={"path": "file_path"})
                else:
                    raise ValueError("CSV must contain file_path and label columns")
            if "label" not in df.columns:
                raise ValueError("CSV must contain file_path and label columns")
            df["label"] = df["label"].astype(str).str.upper()
            dataset_root = Path(dataset_path)
            df["file_path"] = df["file_path"].apply(
                lambda p: str((dataset_root / p).resolve()) if not Path(str(p)).is_absolute() else str(p)
            )
            if validate_images:
                valid_rows = []
                for _, row in df.iterrows():
                    if self._is_valid_image(Path(row["file_path"])):
                        valid_rows.append(row)
                    else:
                        invalid_images.append(str(row["file_path"]))
                df = pd.DataFrame(valid_rows)
        else:
            df, warnings, invalid_images = self._scan_folder_with_quality(dataset_path, validate_images)

        if df.empty:
            raise ValueError("No images found")

        df["label"] = df["label"].astype(str).str.upper()
        class_distribution = {str(k): int(v) for k, v in Counter(df["label"]).items()}
        missing_classes = [cls for cls in EXPECTED_CLASSES if class_distribution.get(cls, 0) == 0]
        if missing_classes:
            warnings.append("Missing expected class label(s): " + ", ".join(missing_classes))

        num_samples = len(df)
        if num_samples < 100:
            warnings.append(
                f"Dataset has only {num_samples} valid images. This is small for medical AI evaluation."
            )

        nonzero_counts = [count for count in class_distribution.values() if count > 0]
        imbalance_ratio = (max(nonzero_counts) / max(min(nonzero_counts), 1)) if nonzero_counts else None
        if imbalance_ratio and imbalance_ratio >= 3.0:
            warnings.append(
                f"Severe class imbalance detected (majority/minority ratio {imbalance_ratio:.2f})."
            )
        if invalid_images:
            warnings.append(f"Skipped {len(invalid_images)} invalid image file(s).")

        labels = df["label"].astype(str)
        train_df, val_df, test_df = self._split_dataframe(
            df,
            labels=labels,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
        )

        split_distribution = {
            split_name: {str(k): int(v) for k, v in Counter(split_df["label"]).items()}
            for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]
        }

        summary = DatasetSummary(
            dataset_name=dataset_name,
            dataset_path=str(dataset_path),
            label_csv_path=label_csv_path,
            num_samples=len(df),
            num_classes=int(df["label"].nunique()),
            train_count=len(train_df),
            val_count=len(val_df),
            test_count=len(test_df),
            class_distribution=class_distribution,
            split_distribution=split_distribution,
            imbalance_ratio=imbalance_ratio,
            warnings=warnings,
            invalid_images=invalid_images,
            random_seed=random_seed,
        )

        dataset_id = self.db.execute(
            "INSERT INTO datasets (dataset_name, dataset_path, label_csv_path, num_samples, num_classes, train_count, val_count, test_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (summary.dataset_name, summary.dataset_path, summary.label_csv_path, summary.num_samples, summary.num_classes, summary.train_count, summary.val_count, summary.test_count),
        )
        self.db.execute(
            """
            UPDATE datasets
            SET class_distribution_json = ?, split_distribution_json = ?,
                imbalance_ratio = ?, invalid_image_count = ?, warnings_json = ?,
                random_seed = ?, details_json = ?
            WHERE id = ?
            """,
            (
                json.dumps(class_distribution),
                json.dumps(split_distribution),
                imbalance_ratio,
                len(invalid_images),
                json.dumps(warnings),
                random_seed,
                json.dumps({"invalid_images": invalid_images}),
                dataset_id,
            ),
        )
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            for _, row in split_df.iterrows():
                self.db.execute(
                    "INSERT INTO images (dataset_id, file_path, label, split) VALUES (?, ?, ?, ?)",
                    (dataset_id, str(row["file_path"]), str(row["label"]), split_name),
                )
            counts = Counter(split_df["label"])
            total = int(sum(counts.values()))
            nonzero = [int(v) for v in counts.values() if int(v) > 0]
            split_ratio = (max(nonzero) / max(min(nonzero), 1)) if nonzero else None
            self.db.save_dataset_distribution(
                dataset_id=dataset_id,
                run_id=None,
                split_strategy="registered_split",
                hospital_id=None,
                split=split_name,
                total_count=total,
                normal_count=int(counts.get("NORMAL", 0)),
                pneumonia_count=int(counts.get("PNEUMONIA", 0)),
                imbalance_ratio=split_ratio,
                details={"class_counts": {str(k): int(v) for k, v in counts.items()}},
            )
        self.db.log("dataset", f"Registered dataset {dataset_name} with {len(df)} images", "success")
        return summary

    def _split_dataframe(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        random_seed: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        stratify_labels = labels if labels.nunique() > 1 and labels.value_counts().min() >= 2 else None
        try:
            train_df, temp_df = train_test_split(
                df,
                test_size=(1 - train_ratio),
                stratify=stratify_labels,
                random_state=random_seed,
            )
            relative_test = test_ratio / (val_ratio + test_ratio)
            temp_labels = temp_df["label"].astype(str)
            temp_stratify = temp_labels if temp_labels.nunique() > 1 and temp_labels.value_counts().min() >= 2 else None
            val_df, test_df = train_test_split(
                temp_df,
                test_size=relative_test,
                stratify=temp_stratify,
                random_state=random_seed,
            )
            return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
        except ValueError:
            shuffled = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
            n = len(shuffled)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train_df = shuffled.iloc[:n_train]
            val_df = shuffled.iloc[n_train:n_train + n_val]
            test_df = shuffled.iloc[n_train + n_val:]
            return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def latest_dataset(self):
        return self.db.fetchone("SELECT * FROM datasets ORDER BY id DESC LIMIT 1")

    def images_for_split(self, dataset_id: int, split: str) -> list[dict]:
        rows = self.db.fetchall("SELECT * FROM images WHERE dataset_id = ? AND split = ?", (dataset_id, split))
        return [dict(r) for r in rows]
