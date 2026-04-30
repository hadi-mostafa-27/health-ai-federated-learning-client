# Health AI Federated Learning Client

Supporting technical documentation for the Health AI Federated Learning Client.

The primary academic report is [ACADEMIC_MASTER_REPORT.md](ACADEMIC_MASTER_REPORT.md). This file gives a shorter implementation-focused overview of the current desktop application.

This project is a PySide6 + PyTorch + SQLite academic prototype for hospital federated learning workflows on chest X-ray pneumonia classification. It is not a clinical device, not a production hospital system, and not a validated diagnostic tool.

## Current Application Scope

The current public version focuses on a desktop simulation workflow:

- admin and hospital login
- admin dashboard
- hospital dashboard
- hospital registry
- hospital active/inactive status
- project request workflow
- project approval workflow
- FL project runner
- simulated FedAvg/FedProx rounds
- dataset validation
- local training
- prediction
- Grad-CAM visualization
- results review
- Docker package export for completed projects
- Windows executable and installer packaging

## Main Components

| Component | Files | Purpose |
| --- | --- | --- |
| App launch | `app.py` | Starts the desktop app. |
| Main window | `ui/main_window.py` | Role-based navigation and page container. |
| UI pages | `ui/pages/*.py` | Dashboard, registry, project requests, runner, results, dataset, training, prediction, Grad-CAM, settings, profile. |
| FL visualizer | `ui/widgets/fl_network_canvas.py` | Draws active/inactive hospitals and simulated FL communication. |
| Dataset manager | `core/dataset_manager.py` | Validates `NORMAL/` and `PNEUMONIA/` folders and creates reproducible splits. |
| Training | `core/trainer.py`, `core/model_loader.py` | DenseNet121 binary classifier training and checkpointing. |
| FL simulation | `core/fl_engine.py`, `core/non_iid.py` | FedAvg, FedProx, partial participation, and non-IID hospital splits. |
| Metrics/reports | `core/metrics.py`, `core/report_generator.py`, `core/experiment_runner.py` | Medical metrics, CSV/JSON outputs, and experiment summaries. |
| Database | `core/db.py` | SQLite persistence for users, hospitals, projects, rounds, metrics, exports, and logs. |
| Docker export | `core/docker_exporter.py` | Prototype Docker deployment package generation after project completion. |
| Packaging | `HospitalFLSystem.spec`, `build_exe.bat`, `installer/HospitalFLSystem.iss` | Windows executable and installer creation. |

## GUI Workflow Summary

### Admin

The admin can:

- view network-level dashboard statistics
- manage registered hospitals
- activate/deactivate hospitals
- review project requests
- approve/reject project requests
- run FL projects with selected active hospitals
- inspect completed project results
- export Docker packages for completed projects

### Hospital

The hospital user can:

- view local dashboard information
- request participation in FL projects
- manage local dataset registration
- run local training
- run prediction
- generate Grad-CAM outputs
- view joined/completed project results
- export Docker package when eligible

## Dataset Format

Datasets must be arranged as:

```text
dataset_root/
  NORMAL/
  PNEUMONIA/
```

The dataset page validates folder structure, counts images, records class distribution, detects invalid images, warns about imbalance, and creates reproducible train/validation/test splits.

## Federated Learning Simulation

The FL project runner simulates hospital participation in federated learning. Active hospitals can participate; inactive hospitals are shown as unavailable and are skipped.

Implemented algorithms:

- weighted FedAvg
- FedProx

Weighted FedAvg:

```text
w_global = sum_k ((n_k / sum_j n_j) * w_k)
```

FedProx local objective:

```text
Loss = BCEWithLogitsLoss + (mu / 2) * ||w_local - w_global||^2
```

## Results And Medical Metrics

The results workflow can show:

- accuracy
- precision
- recall / sensitivity
- specificity
- F1-score
- ROC-AUC
- false negatives
- false positives
- confusion matrix
- per-round performance
- client-level performance

False negatives are highlighted because missing pneumonia is clinically serious.

## Grad-CAM

Grad-CAM is provided as an explanation aid. It can export original images, heatmaps, overlays, and side-by-side comparisons.

Grad-CAM is not clinical proof and must not be used as a replacement for radiologist review.

## Docker Export

After a project is completed, eligible hospitals can export a prototype Docker package under:

```text
exports/docker/<project_id>/<hospital_name>/
```

The package includes metadata, FL settings, metrics, a Dockerfile, deployment README, run script, and model artifact or placeholder artifact.

## Run From Source

```powershell
pip install -r requirements.txt
python app.py
```

## Build Windows Executable

```powershell
.\build_exe.bat
```

Then test:

```powershell
dist\HospitalFLSystem\HospitalFLSystem.exe
```

## Build Installer

Open:

```text
installer/HospitalFLSystem.iss
```

in Inno Setup Compiler and click **Compile**.

## Limitations

- Desktop academic prototype only.
- FL behavior is simulated for demonstration and experimentation.
- No clinical validation.
- No production secure aggregation.
- No formal differential privacy accounting.
- No PACS/EHR integration.
- No regulatory approval.
