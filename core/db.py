from __future__ import annotations

import json
import sqlite3
from pathlib import Path


SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS hospital_profile (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hospital_id TEXT UNIQUE,
        hospital_name TEXT,
        country TEXT,
        city TEXT,
        node_status TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_name TEXT,
        dataset_path TEXT,
        label_csv_path TEXT,
        num_samples INTEGER,
        num_classes INTEGER,
        train_count INTEGER,
        val_count INTEGER,
        test_count INTEGER,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER,
        file_path TEXT,
        label TEXT,
        split TEXT,
        imported_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT,
        model_type TEXT,
        version TEXT,
        source TEXT,
        file_path TEXT,
        round_number INTEGER,
        accuracy REAL,
        loss REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        is_active INTEGER DEFAULT 0,
        metadata_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS training_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id INTEGER,
        dataset_id INTEGER,
        start_time TEXT DEFAULT CURRENT_TIMESTAMP,
        end_time TEXT,
        epochs INTEGER,
        batch_size INTEGER,
        learning_rate REAL,
        optimizer TEXT,
        fedprox_mu REAL,
        status TEXT,
        best_accuracy REAL,
        best_loss REAL,
        history_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT,
        true_label TEXT,
        predicted_label TEXT,
        confidence REAL,
        model_version TEXT,
        inference_time REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        details_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS gradcam_outputs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_id INTEGER,
        heatmap_path TEXT,
        overlay_path TEXT,
        target_layer TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fl_rounds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        round_number INTEGER,
        global_model_version TEXT,
        local_model_version TEXT,
        update_sent_at TEXT,
        global_received_at TEXT,
        status TEXT,
        local_accuracy REAL,
        global_accuracy REAL,
        details_json TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS activity_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT,
        description TEXT,
        status TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
"""
CREATE TABLE IF NOT EXISTS fl_projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_name TEXT,
    disease_target TEXT,
    model_backbone TEXT,
    fl_algorithm TEXT,
    total_rounds INTEGER,
    local_epochs INTEGER,
    batch_size INTEGER,
    learning_rate REAL,
    participation_fraction REAL,
    stop_accuracy REAL,
    status TEXT DEFAULT 'created',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""",
"""
CREATE TABLE IF NOT EXISTS round_participants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER,
    round_number INTEGER,
    hospital_id TEXT,
    status TEXT,
    local_accuracy REAL,
    update_sent_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""",
"""
CREATE TABLE IF NOT EXISTS fl_projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_name TEXT,
    disease_target TEXT,
    model_backbone TEXT,
    fl_algorithm TEXT,
    total_rounds INTEGER,
    local_epochs INTEGER,
    batch_size INTEGER,
    learning_rate REAL,
    participation_fraction REAL,
    stop_accuracy REAL,
    status TEXT DEFAULT 'created',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""",
"""
CREATE TABLE IF NOT EXISTS round_participants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER,
    round_number INTEGER,
    hospital_id TEXT,
    status TEXT,
    local_accuracy REAL,
    update_sent_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""",
"""
CREATE TABLE IF NOT EXISTS project_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hospital_name TEXT,
    project_name TEXT,
    disease_target TEXT,
    dataset_type TEXT,
    suggested_backbone TEXT,
    fl_algorithm TEXT,
    reason TEXT,
    status TEXT DEFAULT 'pending',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TEXT,
    details_json TEXT
)
""",
"""
CREATE TABLE IF NOT EXISTS project_memberships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER,
    hospital_id TEXT,
    hospital_name TEXT,
    status TEXT DEFAULT 'invited',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""",
"""
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT,
    architecture TEXT,
    version TEXT,
    file_path TEXT,
    source TEXT,
    aggregation_algorithm TEXT,
    threshold REAL,
    metrics_json TEXT,
    training_config_json TEXT,
    metadata_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""",
"""
CREATE TABLE IF NOT EXISTS federated_rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_run_id TEXT,
    project_id INTEGER,
    round_number INTEGER,
    aggregation_algorithm TEXT,
    participation_fraction REAL,
    participating_clients_json TEXT,
    client_sample_counts_json TEXT,
    global_metrics_json TEXT,
    status TEXT,
    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT
)
""",
"""
CREATE TABLE IF NOT EXISTS client_updates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_run_id TEXT,
    project_id INTEGER,
    round_number INTEGER,
    hospital_id TEXT,
    num_samples INTEGER,
    local_loss REAL,
    local_accuracy REAL,
    local_metrics_json TEXT,
    update_path TEXT,
    status TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""",
"""
CREATE TABLE IF NOT EXISTS experiment_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT UNIQUE,
    run_name TEXT,
    experiment_type TEXT,
    aggregation_algorithm TEXT,
    dataset_id INTEGER,
    seed INTEGER,
    config_json TEXT,
    environment_json TEXT,
    status TEXT,
    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    summary_json TEXT
)
""",
"""
CREATE TABLE IF NOT EXISTS evaluation_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    model_version_id INTEGER,
    scope TEXT,
    round_number INTEGER,
    hospital_id TEXT,
    split TEXT,
    threshold REAL,
    accuracy REAL,
    precision REAL,
    recall REAL,
    f1_score REAL,
    roc_auc REAL,
    sensitivity REAL,
    specificity REAL,
    false_negatives INTEGER,
    false_positives INTEGER,
    metrics_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""",
"""
CREATE TABLE IF NOT EXISTS confusion_matrices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    model_version_id INTEGER,
    scope TEXT,
    round_number INTEGER,
    hospital_id TEXT,
    split TEXT,
    true_negative INTEGER,
    false_positive INTEGER,
    false_negative INTEGER,
    true_positive INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""",
"""
CREATE TABLE IF NOT EXISTS dataset_distributions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER,
    run_id TEXT,
    split_strategy TEXT,
    hospital_id TEXT,
    split TEXT,
    total_count INTEGER,
    normal_count INTEGER,
    pneumonia_count INTEGER,
    imbalance_ratio REAL,
    details_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""",
]


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        conn = self._connect()
        cur = conn.cursor()
        for stmt in SCHEMA:
            cur.execute(stmt)
        self._run_idempotent_migrations(cur)
        conn.commit()
        conn.close()
        self.seed_hospitals()

    def _column_exists(self, cur, table: str, column: str) -> bool:
        cur.execute(f"PRAGMA table_info({table})")
        return any(row[1] == column for row in cur.fetchall())

    def _add_column_if_missing(self, cur, table: str, column: str, definition: str) -> None:
        if not self._column_exists(cur, table, column):
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _run_idempotent_migrations(self, cur) -> None:
        migrations = [
            ("project_requests", "details_json", "TEXT"),
            ("fl_projects", "details_json", "TEXT"),
            ("fl_projects", "aggregation_algorithm", "TEXT"),
            ("fl_projects", "threshold_strategy", "TEXT"),
            ("fl_projects", "non_iid_strategy", "TEXT"),
            ("fl_projects", "imbalance_severity", "REAL"),
            ("fl_projects", "completed_at", "TEXT"),
            ("fl_projects", "final_model_path", "TEXT"),
            ("fl_projects", "final_metrics_json", "TEXT"),
            ("fl_projects", "docker_export_path", "TEXT"),
            ("fl_projects", "docker_export_created_at", "TEXT"),
            ("datasets", "class_distribution_json", "TEXT"),
            ("datasets", "split_distribution_json", "TEXT"),
            ("datasets", "imbalance_ratio", "REAL"),
            ("datasets", "invalid_image_count", "INTEGER DEFAULT 0"),
            ("datasets", "warnings_json", "TEXT"),
            ("datasets", "random_seed", "INTEGER DEFAULT 42"),
            ("datasets", "details_json", "TEXT"),
            ("models", "threshold", "REAL"),
            ("models", "architecture", "TEXT"),
            ("models", "metrics_json", "TEXT"),
            ("models", "training_config_json", "TEXT"),
            ("fl_rounds", "aggregation_algorithm", "TEXT"),
            ("fl_rounds", "participating_clients_json", "TEXT"),
            ("fl_rounds", "client_sample_counts_json", "TEXT"),
            ("fl_rounds", "global_metrics_json", "TEXT"),
            ("round_participants", "sample_count", "INTEGER"),
            ("round_participants", "local_loss", "REAL"),
            ("round_participants", "local_metrics_json", "TEXT"),
        ]
        for table, column, definition in migrations:
            try:
                self._add_column_if_missing(cur, table, column, definition)
            except sqlite3.OperationalError:
                pass
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS docker_exports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                hospital_id TEXT,
                hospital_name TEXT,
                export_folder TEXT,
                zip_path TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                metadata_json TEXT
            )
            """
        )

    def seed_hospitals(self):
        demo_hospitals = [
                ("AUBMC", "American University of Beirut Medical Center", "Beirut"),
                ("RHUH", "Rafik Hariri University Hospital", "Beirut"),
                ("HDF", "Hotel Dieu de France", "Beirut"),
                ("CMC", "Clemenceau Medical Center", "Beirut"),
                ("BMC", "Bellevue Medical Center", "Mansourieh"),
                ("MLH", "Mount Lebanon Hospital", "Hazmieh"),
                ("LAUMC_RIZK", "LAU Medical Center - Rizk Hospital", "Beirut"),
                ("SGHUMC", "Saint George Hospital University Medical Center", "Beirut"),
                ("GEITAOUI", "Lebanese Hospital Geitaoui", "Beirut"),
                ("HAMMOUD", "Hammoud Hospital University Medical Center", "Saida"),
                ("NINI", "Nini Hospital", "Tripoli"),
                ("HAYKAL", "Haykal Hospital", "Tripoli"),
                ("DAR_AL_AMAL", "Dar Al Amal University Hospital", "Baalbek"),
                ("NABATIEH_GOV", "Nabatieh Governmental Hospital", "Nabatieh"),
                ("TEBNINE_GOV", "Tebnine Governmental Hospital", "Tebnine"),
                ("SAIDA_GOV", "Saida Governmental Hospital", "Saida"),
                ("TRIPOLI_GOV", "Tripoli Governmental Hospital", "Tripoli"),
                ("BAABDA_GOV", "Baabda Governmental Hospital", "Baabda"),
                ("ZAHLE_GOV", "Zahle Governmental Hospital", "Zahle"),
                ("KMC", "Keserwan Medical Center", "Ghazir"),
                ("NDSUH", "Notre Dame de Secours University Hospital", "Jbeil"),
                ("MEIH", "Middle East Institute of Health", "Bsalim"),
                ("MAKASSED", "Makassed General Hospital", "Beirut"),
                ("ISLAMIC_TRIPOLI", "Islamic Hospital Tripoli", "Tripoli"),
                ("SAHEL", "Sahel General Hospital", "Beirut"),
                ("ZAHRAA", "Al Zahraa Hospital University Medical Center", "Beirut"),
                ("BAHMAN", "Bahman Hospital", "Beirut"),
                ("RASSOUL", "Al Rassoul Al Aazam Hospital", "Beirut"),
                ("LABIB", "Labib Medical Center", "Saida"),
                ("HAMMANA", "Hammana Hospital", "Hammana"),
                ("DALLAA", "Dalla'a General Hospital", "Saida"),
                ("MONLA", "Monla Hospital", "Tripoli"),
                ("MAZLOUM", "Mazloum Hospital", "Tripoli"),
                ("NDM", "Hopital Notre Dame Maritime", "Jbeil"),
                ("TAANAYEL", "Taanayel General Hospital", "Taanayel"),
                ("RAYAK", "Rayak Hospital", "Rayak"),
                ("ELIAS_HRAWI", "Elias Hrawi Governmental Hospital", "Zahle"),
                ("MARJAYOUN_GOV", "Marjayoun Governmental Hospital", "Marjayoun"),
                ("HASBAYA_GOV", "Hasbaya Governmental Hospital", "Hasbaya"),
                ("JEZZINE_GOV", "Jezzine Governmental Hospital", "Jezzine"),
                ("BENT_JBEIL_GOV", "Bent Jbeil Governmental Hospital", "Bent Jbeil"),
                ("HERMEL_GOV", "Hermel Governmental Hospital", "Hermel")
        ]
        for hid, name, city in demo_hospitals:
            self.seed_hospital(hid, name, city)

    def seed_hospital(self, hospital_id: str, hospital_name: str, city: str = ""):
        """Create demo hospital rows without overwriting user/admin status changes."""
        query = """
            INSERT INTO hospital_profile (hospital_id, hospital_name, city, node_status)
            VALUES (?, ?, ?, 'active')
            ON CONFLICT(hospital_id) DO UPDATE SET
                hospital_name=excluded.hospital_name,
                city=excluded.city
        """
        return self.execute(query, (hospital_id, hospital_name, city))

    def execute(self, query: str, params=()) -> int:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()
        lastrowid = cur.lastrowid
        conn.close()
        return lastrowid

    def fetch_all(self, query: str, params=()):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        conn.close()
        return rows

    def fetch_one(self, query: str, params=()):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(query, params)
        row = cur.fetchone()
        conn.close()
        return row

    def fetchall(self, query: str, params=()):
        return self.fetch_all(query, params)

    def fetchone(self, query: str, params=()):
        return self.fetch_one(query, params)

    def log(self, event_type: str, description: str, status: str = "info") -> int:
        query = """
            INSERT INTO activity_logs (event_type, description, status)
            VALUES (?, ?, ?)
        """
        return self.execute(query, (event_type, description, status))

    def save_model_version(
        self,
        *,
        model_name: str,
        architecture: str,
        version: str,
        file_path: str,
        source: str,
        aggregation_algorithm: str | None = None,
        threshold: float | None = None,
        metrics: dict | None = None,
        training_config: dict | None = None,
        metadata: dict | None = None,
    ) -> int:
        return self.execute(
            """
            INSERT INTO model_versions (
                model_name, architecture, version, file_path, source,
                aggregation_algorithm, threshold, metrics_json,
                training_config_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_name,
                architecture,
                version,
                file_path,
                source,
                aggregation_algorithm,
                threshold,
                json.dumps(metrics or {}, default=str),
                json.dumps(training_config or {}, default=str),
                json.dumps(metadata or {}, default=str),
            ),
        )

    def create_experiment_run(
        self,
        *,
        run_id: str,
        run_name: str,
        experiment_type: str,
        aggregation_algorithm: str | None,
        dataset_id: int | None,
        seed: int,
        config: dict,
        environment: dict,
        status: str = "running",
    ) -> int:
        return self.execute(
            """
            INSERT OR REPLACE INTO experiment_runs (
                run_id, run_name, experiment_type, aggregation_algorithm,
                dataset_id, seed, config_json, environment_json, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                run_name,
                experiment_type,
                aggregation_algorithm,
                dataset_id,
                seed,
                json.dumps(config, default=str),
                json.dumps(environment, default=str),
                status,
            ),
        )

    def finish_experiment_run(self, run_id: str, status: str, summary: dict | None = None) -> int:
        return self.execute(
            """
            UPDATE experiment_runs
            SET status = ?, completed_at = CURRENT_TIMESTAMP, summary_json = ?
            WHERE run_id = ?
            """,
            (status, json.dumps(summary or {}, default=str), run_id),
        )

    def save_evaluation_metrics(
        self,
        *,
        run_id: str | None,
        model_version_id: int | None,
        scope: str,
        metrics: dict,
        round_number: int | None = None,
        hospital_id: str | None = None,
        split: str | None = None,
    ) -> int:
        return self.execute(
            """
            INSERT INTO evaluation_metrics (
                run_id, model_version_id, scope, round_number, hospital_id, split,
                threshold, accuracy, precision, recall, f1_score, roc_auc,
                sensitivity, specificity, false_negatives, false_positives,
                metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                model_version_id,
                scope,
                round_number,
                hospital_id,
                split,
                metrics.get("threshold"),
                metrics.get("accuracy"),
                metrics.get("precision"),
                metrics.get("recall"),
                metrics.get("f1_score"),
                metrics.get("roc_auc"),
                metrics.get("sensitivity"),
                metrics.get("specificity"),
                metrics.get("false_negatives"),
                metrics.get("false_positives"),
                json.dumps(metrics, default=str),
            ),
        )

    def save_confusion_matrix(
        self,
        *,
        run_id: str | None,
        model_version_id: int | None,
        scope: str,
        metrics: dict,
        round_number: int | None = None,
        hospital_id: str | None = None,
        split: str | None = None,
    ) -> int:
        cm = metrics.get("confusion_matrix") or [[0, 0], [0, 0]]
        return self.execute(
            """
            INSERT INTO confusion_matrices (
                run_id, model_version_id, scope, round_number, hospital_id, split,
                true_negative, false_positive, false_negative, true_positive
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                model_version_id,
                scope,
                round_number,
                hospital_id,
                split,
                int(cm[0][0]),
                int(cm[0][1]),
                int(cm[1][0]),
                int(cm[1][1]),
            ),
        )

    def save_federated_round(
        self,
        *,
        round_number: int,
        aggregation_algorithm: str,
        participating_clients: list[str],
        client_sample_counts: dict,
        global_metrics: dict,
        status: str,
        experiment_run_id: str | None = None,
        project_id: int | None = None,
        participation_fraction: float | None = None,
    ) -> int:
        return self.execute(
            """
            INSERT INTO federated_rounds (
                experiment_run_id, project_id, round_number, aggregation_algorithm,
                participation_fraction, participating_clients_json,
                client_sample_counts_json, global_metrics_json, status,
                completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                experiment_run_id,
                project_id,
                round_number,
                aggregation_algorithm,
                participation_fraction,
                json.dumps(participating_clients, default=str),
                json.dumps(client_sample_counts, default=str),
                json.dumps(global_metrics, default=str),
                status,
            ),
        )

    def save_client_update(
        self,
        *,
        hospital_id: str,
        num_samples: int,
        local_loss: float | None,
        local_accuracy: float | None,
        local_metrics: dict | None,
        status: str,
        round_number: int | None = None,
        experiment_run_id: str | None = None,
        project_id: int | None = None,
        update_path: str | None = None,
    ) -> int:
        return self.execute(
            """
            INSERT INTO client_updates (
                experiment_run_id, project_id, round_number, hospital_id,
                num_samples, local_loss, local_accuracy, local_metrics_json,
                update_path, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_run_id,
                project_id,
                round_number,
                hospital_id,
                num_samples,
                local_loss,
                local_accuracy,
                json.dumps(local_metrics or {}, default=str),
                update_path,
                status,
            ),
        )

    def save_dataset_distribution(
        self,
        *,
        dataset_id: int | None,
        run_id: str | None,
        split_strategy: str,
        hospital_id: str | None,
        split: str,
        total_count: int,
        normal_count: int,
        pneumonia_count: int,
        imbalance_ratio: float | None,
        details: dict | None = None,
    ) -> int:
        return self.execute(
            """
            INSERT INTO dataset_distributions (
                dataset_id, run_id, split_strategy, hospital_id, split,
                total_count, normal_count, pneumonia_count, imbalance_ratio,
                details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                run_id,
                split_strategy,
                hospital_id,
                split,
                total_count,
                normal_count,
                pneumonia_count,
                imbalance_ratio,
                json.dumps(details or {}, default=str),
            ),
        )

    def create_fl_project(
            self,
            project_name: str,
            disease_target: str,
            model_backbone: str,
            fl_algorithm: str,
            total_rounds: int,
            local_epochs: int,
            batch_size: int,
            learning_rate: float,
            participation_fraction: float,
            stop_accuracy: float,
            details_json: str = None
    ) -> int:
        query = """
            INSERT INTO fl_projects (
                project_name, disease_target, model_backbone, fl_algorithm,
                total_rounds, local_epochs, batch_size, learning_rate,
                participation_fraction, stop_accuracy, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        project_id = self.execute(
            query,
            (
                project_name,
                disease_target,
                model_backbone,
                fl_algorithm,
                total_rounds,
                local_epochs,
                batch_size,
                learning_rate,
                participation_fraction,
                stop_accuracy,
                "created",
            ),
        )
        
        if details_json:
            # Dynamically alter table to ensure it exists
            try:
                self.execute("ALTER TABLE fl_projects ADD COLUMN details_json TEXT")
            except sqlite3.OperationalError:
                pass
            self.execute("UPDATE fl_projects SET details_json = ? WHERE id = ?", (details_json, project_id))
            
        return project_id

    def get_latest_fl_project(self):
        return self.fetchone("SELECT * FROM fl_projects ORDER BY id DESC LIMIT 1")

    def get_fl_project(self, project_id: int):
        return self.fetchone("SELECT * FROM fl_projects WHERE id = ?", (project_id,))

    def update_fl_project_status(self, project_id: int, status: str) -> int:
        return self.execute(
            "UPDATE fl_projects SET status = ? WHERE id = ?",
            (status, project_id),
        )

    def complete_fl_project(
        self,
        project_id: int,
        final_model_path: str | None = None,
        final_metrics: dict | None = None,
    ) -> int:
        return self.execute(
            """
            UPDATE fl_projects
            SET status = 'completed',
                completed_at = CURRENT_TIMESTAMP,
                final_model_path = COALESCE(?, final_model_path),
                final_metrics_json = ?
            WHERE id = ?
            """,
            (final_model_path, json.dumps(final_metrics or {}, default=str), project_id),
        )

    def get_latest_project_metrics(self, project_id: int) -> dict:
        row = self.fetchone(
            """
            SELECT global_metrics_json
            FROM federated_rounds
            WHERE project_id = ? AND global_metrics_json IS NOT NULL
            ORDER BY round_number DESC, id DESC
            LIMIT 1
            """,
            (project_id,),
        )
        if not row or not row["global_metrics_json"]:
            project = self.get_fl_project(project_id)
            if project and "final_metrics_json" in project.keys() and project["final_metrics_json"]:
                try:
                    return json.loads(project["final_metrics_json"])
                except json.JSONDecodeError:
                    return {}
            return {}
        try:
            return json.loads(row["global_metrics_json"])
        except json.JSONDecodeError:
            return {}

    def add_round_participant(self, project_id: int, round_number: int, hospital_id: str,
                              status: str = "selected") -> int:
        query = """
            INSERT INTO round_participants (project_id, round_number, hospital_id, status)
            VALUES (?, ?, ?, ?)
        """
        return self.execute(query, (project_id, round_number, hospital_id, status))

    def list_round_participants(self, project_id: int, round_number: int):
        return self.fetchall(
            "SELECT * FROM round_participants WHERE project_id = ? AND round_number = ? ORDER BY id ASC",
            (project_id, round_number),
        )



    def create_project_request(self, hospital_name, project_name, disease_target, dataset_type, suggested_backbone, fl_algorithm, reason, details_json=None):
        query = """
            INSERT INTO project_requests (
                hospital_name, project_name, disease_target, dataset_type,
                suggested_backbone, fl_algorithm, reason, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        return self.execute(query, (hospital_name, project_name, disease_target, dataset_type, suggested_backbone, fl_algorithm, reason, details_json))

    def list_project_requests(self, status=None):
        if status:
            return self.fetchall("SELECT * FROM project_requests WHERE status = ? ORDER BY created_at DESC", (status,))
        return self.fetchall("SELECT * FROM project_requests ORDER BY created_at DESC")

    def update_project_request_status(self, request_id, status):
        query = "UPDATE project_requests SET status = ?, reviewed_at = CURRENT_TIMESTAMP WHERE id = ?"
        return self.execute(query, (status, request_id))

    def approve_project_request(self, request_id):
        self.update_project_request_status(request_id, 'approved')
        return self.create_project_from_request(request_id)

    def reject_project_request(self, request_id):
        return self.update_project_request_status(request_id, 'rejected')

    def create_project_from_request(self, request_id):
        req = self.fetchone("SELECT * FROM project_requests WHERE id = ?", (request_id,))
        if not req:
            return None
            
        import json
        details = {}
        if req['details_json']:
            try:
                details = json.loads(req['details_json'])
            except json.JSONDecodeError:
                pass
                
        # Preserve ownership info inside details
        details['created_by_role'] = 'hospital'
        details['created_by_hospital_id'] = details.get('requesting_hospital_id', 'UNKNOWN')
        details['created_by_display_name'] = req['hospital_name']
        details['approved_by'] = 'Administration Department'
        details['created_from_request_id'] = request_id
                
        project_id = self.create_fl_project(
            project_name=req['project_name'],
            disease_target=req['disease_target'],
            model_backbone=req['suggested_backbone'],
            fl_algorithm=req['fl_algorithm'],
            total_rounds=details.get('total_rounds', 5),
            local_epochs=details.get('local_epochs', 1),
            batch_size=details.get('batch_size', 8),
            learning_rate=details.get('learning_rate', 0.0001),
            participation_fraction=details.get('participation_fraction', 1.0),
            stop_accuracy=details.get('stop_accuracy', 0.92),
            details_json=json.dumps(details)
        )
        
        # Add members
        def is_active_hospital(hospital_id: str) -> bool:
            hospital = self.get_hospital(hospital_id)
            return str(hospital["node_status"] if hospital else "inactive").lower() == "active"

        req_hospital_id = details.get('requesting_hospital_id')
        if req_hospital_id:
            status = "requester_joined" if is_active_hospital(req_hospital_id) else "unavailable"
            self.add_project_membership(project_id, req_hospital_id, req['hospital_name'], status)
            
        invited_hospitals = details.get('requested_hospitals', [])
        for inv_id, inv_name in invited_hospitals:
            if inv_id != req_hospital_id:
                status = "invited" if is_active_hospital(inv_id) else "unavailable"
                self.add_project_membership(project_id, inv_id, inv_name, status)
                
        return project_id

    def add_project_membership(self, project_id, hospital_id, hospital_name, status="invited"):
        query = """
            INSERT INTO project_memberships (project_id, hospital_id, hospital_name, status)
            VALUES (?, ?, ?, ?)
        """
        return self.execute(query, (project_id, hospital_id, hospital_name, status))

    def list_project_memberships(self, project_id):
        return self.fetchall("SELECT * FROM project_memberships WHERE project_id = ? ORDER BY id ASC", (project_id,))

    def get_project_membership(self, project_id: int, hospital_id: str):
        return self.fetchone(
            "SELECT * FROM project_memberships WHERE project_id = ? AND hospital_id = ? LIMIT 1",
            (project_id, hospital_id),
        )

    def list_joined_project_memberships(self, project_id: int):
        return self.fetchall(
            """
            SELECT pm.*, hp.node_status
            FROM project_memberships pm
            LEFT JOIN hospital_profile hp ON hp.hospital_id = pm.hospital_id
            WHERE pm.project_id = ? AND pm.status LIKE '%joined%'
            ORDER BY pm.id ASC
            """,
            (project_id,),
        )

    def update_membership_status(self, project_id, hospital_id, status):
        query = "UPDATE project_memberships SET status = ? WHERE project_id = ? AND hospital_id = ?"
        return self.execute(query, (status, project_id, hospital_id))

    def list_approved_projects(self):
        return self.fetchall("SELECT * FROM fl_projects WHERE status = 'created' ORDER BY id DESC")

    def add_hospital(self, hospital_id: str, hospital_name: str, city: str = "", status: str = "active"):
        query = """
            INSERT INTO hospital_profile (hospital_id, hospital_name, city, node_status)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(hospital_id) DO UPDATE SET
                hospital_name=excluded.hospital_name,
                city=excluded.city,
                node_status=excluded.node_status
        """
        return self.execute(query, (hospital_id, hospital_name, city, status))

    def remove_hospital(self, hospital_id: str):
        return self.execute("DELETE FROM hospital_profile WHERE hospital_id = ?", (hospital_id,))

    def list_hospitals(self, active_only: bool = True):
        if active_only:
            return self.fetchall("SELECT * FROM hospital_profile WHERE node_status = 'active' ORDER BY id ASC")
        return self.fetchall("SELECT * FROM hospital_profile ORDER BY id ASC")

    def update_hospital_status(self, hospital_id: str, status: str):
        return self.execute("UPDATE hospital_profile SET node_status = ? WHERE hospital_id = ?", (status, hospital_id))

    def get_hospital(self, hospital_id: str):
        return self.fetchone("SELECT * FROM hospital_profile WHERE hospital_id = ?", (hospital_id,))

    def record_docker_export(
        self,
        *,
        project_id: int,
        hospital_id: str,
        hospital_name: str,
        export_folder: str,
        zip_path: str,
        metadata: dict | None = None,
    ) -> int:
        export_id = self.execute(
            """
            INSERT INTO docker_exports (
                project_id, hospital_id, hospital_name, export_folder, zip_path, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                hospital_id,
                hospital_name,
                export_folder,
                zip_path,
                json.dumps(metadata or {}, default=str),
            ),
        )
        self.execute(
            """
            UPDATE fl_projects
            SET docker_export_path = ?,
                docker_export_created_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (zip_path, project_id),
        )
        return export_id
