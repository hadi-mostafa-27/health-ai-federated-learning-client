from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
from typing import Any

import pandas as pd

from core.db import DatabaseManager
from core.paths import writable_path


class ReportGenerator:
    def __init__(self, db: DatabaseManager, reports_dir: str = "reports"):
        self.db = db
        self.reports_dir = writable_path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def save_prediction_report(self, payload: dict[str, Any]) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.reports_dir / f"prediction_report_{timestamp}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return str(path)

    def save_results_report(self, payload: dict[str, Any]) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.reports_dir / f"results_report_{timestamp}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return str(path)

    def save_experiment_report(self, run_id: str, payload: dict[str, Any]) -> dict[str, str]:
        out_dir = self.reports_dir / "experiments" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "report.json"
        json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

        paths = {"json": str(json_path)}
        rows = payload.get("summary_rows", [])
        if rows:
            csv_path = out_dir / "metrics_summary.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            paths["csv"] = str(csv_path)

        per_round = payload.get("per_round", [])
        if per_round:
            round_csv = out_dir / "per_round_metrics.csv"
            pd.DataFrame(per_round).to_csv(round_csv, index=False)
            paths["per_round_csv"] = str(round_csv)
            plot_path = self._save_convergence_plot(per_round, out_dir)
            if plot_path:
                paths["convergence_plot"] = plot_path

        client_rows = payload.get("client_level", [])
        if client_rows:
            client_csv = out_dir / "client_level_metrics.csv"
            pd.DataFrame(client_rows).to_csv(client_csv, index=False)
            paths["client_csv"] = str(client_csv)

        return paths

    def _save_convergence_plot(self, per_round: list[dict[str, Any]], out_dir: Path) -> str | None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return None

        rounds = [row.get("round_number") for row in per_round]
        accuracy = [row.get("accuracy") for row in per_round]
        f1 = [row.get("f1_score") for row in per_round]
        sensitivity = [row.get("sensitivity") for row in per_round]

        plt.figure(figsize=(8, 5))
        plt.plot(rounds, accuracy, marker="o", label="Accuracy")
        plt.plot(rounds, f1, marker="o", label="F1")
        plt.plot(rounds, sensitivity, marker="o", label="Sensitivity")
        plt.xlabel("FL Round")
        plt.ylabel("Metric")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        path = out_dir / "convergence.png"
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return str(path)
