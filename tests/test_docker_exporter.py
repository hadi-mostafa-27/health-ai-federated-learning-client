import json
import tempfile
import unittest
from pathlib import Path

from core.db import DatabaseManager
from core.docker_exporter import DockerExportError, DockerPackageExporter


class DockerPackageExporterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = DatabaseManager(str(self.root / "test.db"))
        self.db.initialize()
        self.db.add_hospital("HOSP_1", "Test Hospital", "Beirut", "active")
        self.project_id = self.db.create_fl_project(
            project_name="Pneumonia FL",
            disease_target="Pneumonia",
            model_backbone="DenseNet121",
            fl_algorithm="FedAvg",
            total_rounds=1,
            local_epochs=1,
            batch_size=2,
            learning_rate=0.001,
            participation_fraction=1.0,
            stop_accuracy=0.9,
            details_json=json.dumps({"created_by_display_name": "Admin"}),
        )
        self.db.add_project_membership(self.project_id, "HOSP_1", "Test Hospital", "joined")

    def tearDown(self):
        self.tmp.cleanup()

    def test_rejects_incomplete_project(self):
        exporter = DockerPackageExporter(self.db, self.root / "exports")
        with self.assertRaises(DockerExportError):
            exporter.export_for_hospital(
                project_id=self.project_id,
                hospital_id="HOSP_1",
                requester_role="hospital",
            )

    def test_exports_completed_project_package(self):
        self.db.complete_fl_project(self.project_id, final_metrics={"accuracy": 0.8, "loss": 0.4})
        exporter = DockerPackageExporter(self.db, self.root / "exports")

        result = exporter.export_for_hospital(
            project_id=self.project_id,
            hospital_id="HOSP_1",
            requester_role="hospital",
        )

        folder = Path(result["export_folder"])
        self.assertTrue(folder.exists())
        self.assertTrue((folder / "Dockerfile").exists())
        self.assertTrue((folder / "README_DEPLOY.md").exists())
        self.assertTrue((folder / "model" / "simulated_model_artifact.json").exists())
        self.assertTrue(Path(result["zip_path"]).exists())


if __name__ == "__main__":
    unittest.main()
