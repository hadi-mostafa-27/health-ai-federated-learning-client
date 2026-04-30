# Health AI Federated Learning Client

Comprehensive academic project documentation for the Health AI Federated Learning Client.

The primary thesis/report source is [ACADEMIC_MASTER_REPORT.md](ACADEMIC_MASTER_REPORT.md). This file is kept as a supporting technical reference.

This project is a research and teaching prototype for federated learning in medical imaging. It is not a clinical device, not a production hospital system, and not a validated diagnostic tool.

## Table Of Contents

1. Project summary
2. Objectives and research motivation
3. System architecture
4. Repository structure
5. Installation and setup
6. Application workflow
7. Dataset handling
8. Non-IID federated simulation
9. Model architecture
10. Training pipeline
11. Federated learning methodology
12. Experimental evaluation
13. Medical AI metrics
14. Threshold tuning
15. Grad-CAM explanation module
16. Database design
17. User interface guide
18. Mock FL server
19. Reproducibility
20. Privacy and security limitations
21. Robustness and error handling
22. Reports and artifacts
23. How to run experiments
24. How to reproduce results
25. Troubleshooting
26. Limitations
27. Future work

## 1. Project Summary

The Health AI Federated Learning Client is a PySide6 desktop application backed by PyTorch, SQLite, and a FastAPI mock coordinator. It is designed to simulate federated learning across hospitals for binary chest X-ray pneumonia classification.

The classification task is:

- `NORMAL`
- `PNEUMONIA`

The project supports:

- Local-only training
- Centralized baseline training
- Federated FedAvg
- Federated FedProx
- Non-IID hospital simulation
- Medical AI evaluation metrics
- Threshold tuning
- Grad-CAM visualization
- SQLite experiment tracking
- JSON/CSV reports

The project is intentionally labelled as a prototype and simulation where appropriate.

## 2. Objectives And Research Motivation

Medical AI models often require data from multiple institutions to generalize well. However, patient data sharing is constrained by privacy, governance, legal, and institutional limitations. Federated learning is a method where hospitals train locally and share model updates rather than raw data.

This project studies that idea in a controlled prototype setting.

Primary objectives:

- Build a reproducible binary pneumonia classification pipeline.
- Compare local-only, centralized, FedAvg, and FedProx training.
- Track clinically relevant metrics beyond accuracy.
- Simulate non-IID hospital data distributions.
- Make FL limitations clear rather than overstating privacy guarantees.
- Provide a UI suitable for demonstration and academic supervision.

Research questions this prototype can help explore:

- Does federated training improve over local-only models in a simulated hospital network?
- How does non-IID label or quantity skew affect convergence?
- Does FedProx improve stability under heterogeneous client data?
- How do threshold choices affect false negatives and sensitivity?
- How does client participation fraction affect global performance?

## 3. System Architecture

The system has five main layers:

```text
Desktop UI (PySide6)
  |
  |-- Dataset registration and validation
  |-- Local training controls
  |-- Federated project runner
  |-- Prediction and Grad-CAM pages
  |-- Results and model history pages

Core ML/FL Layer
  |
  |-- Dataset manager
  |-- DenseNet121 model loader
  |-- Local trainer
  |-- Federated engine
  |-- Experiment runner
  |-- Metrics and threshold utilities

Persistence Layer
  |
  |-- SQLite database
  |-- Model version records
  |-- FL round records
  |-- Client update records
  |-- Evaluation metrics
  |-- Dataset distributions

Mock Coordination Layer
  |
  |-- FastAPI mock FL server
  |-- Model download endpoint
  |-- Client update upload endpoint

Filesystem Artifacts
  |
  |-- datasets
  |-- checkpoints
  |-- reports
  |-- Grad-CAM outputs
  |-- prediction JSON files
```

## 4. Repository Structure

```text
hospital_fl_client/
  app.py
  run_experiments.py
  README.md
  requirements.txt

  assets/
    style.qss

  config/
    app_config.json

  core/
    config_manager.py
    data_generator.py
    dataset_manager.py
    db.py
    experiment_runner.py
    fl_client.py
    fl_engine.py
    gradcam_engine.py
    inference.py
    metrics.py
    model_loader.py
    non_iid.py
    notebook_profiles.py
    report_generator.py
    reproducibility.py
    trainer.py

  data/
    raw/
      NORMAL/
      PNEUMONIA/
    predictions/
    visualizations/

  database/
    hospital_client.db

  docs/
    PROJECT_DOCUMENTATION.md

  models/
    local/
    server_downloads/
    fedavg_best.pt
    fedprox_best.pt
    centralized_best.pt

  reports/
    experiments/

  sample_data/

  server/
    mock_fl_server.py
    storage/

  ui/
    main_window.py
    login_window.py
    pages/
    widgets/
```

## 5. Installation And Setup

### Requirements

- Python 3.9 or newer
- PyTorch
- Torchvision
- PySide6
- scikit-learn
- pandas
- Pillow
- OpenCV
- FastAPI
- Uvicorn

Install dependencies:

```powershell
pip install -r requirements.txt
```

If using the included virtual environment on Windows:

```powershell
.\.venv\Scripts\python.exe app.py
```

### Start The Desktop App

```powershell
python app.py
```

### Start The Mock FL Server

```powershell
python -m uvicorn server.mock_fl_server:app --reload
```

The default server URL is stored in:

```text
config/app_config.json
```

Default:

```json
"server_url": "http://127.0.0.1:8000"
```

## 6. Application Workflow

Typical academic workflow:

1. Launch the app.
2. Register or validate a chest X-ray dataset.
3. Inspect class distribution and warnings.
4. Train a local model.
5. Tune threshold on validation data.
6. Run prediction and optional Grad-CAM.
7. Create or request a federated project.
8. Configure FL algorithm, participation fraction, non-IID split, and local epochs.
9. Run FL simulation.
10. Inspect per-client and per-round metrics.
11. Export reports.
12. Run CLI experiments for systematic comparison.

## 7. Dataset Handling

### Expected Folder Format

The default dataset format is:

```text
dataset_root/
  NORMAL/
    image_001.jpeg
    image_002.jpeg
  PNEUMONIA/
    image_003.jpeg
    image_004.jpeg
```

Supported image extensions:

- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.webp`

### Dataset Validation

Implemented in:

```text
core/dataset_manager.py
```

The dataset manager validates:

- Dataset folder exists
- `NORMAL/` folder exists
- `PNEUMONIA/` folder exists
- Images can be opened by Pillow
- Dataset is not empty
- Class distribution is recorded
- Train/validation/test splits are created

### Dataset Summary

The registered dataset summary includes:

- Total image count
- Number of classes
- Class distribution
- Train/validation/test counts
- Split-level class distribution
- Imbalance ratio
- Invalid image count
- Warnings
- Random seed

### Dataset Warnings

The system warns when:

- No images are found
- Expected class folders are missing
- A class has zero samples
- Dataset is very small
- Class imbalance is severe
- Invalid image files are skipped

### Splitting

The split is configurable:

- Train ratio
- Validation ratio
- Test ratio
- Random seed

The default split is:

```text
train = 0.70
validation = 0.15
test = 0.15
seed = 42
```

### Transforms

Training transforms:

- Resize
- Grayscale to 3 channels
- Mild random rotation
- Mild affine transform
- Random horizontal flip
- Tensor conversion
- ImageNet normalization

Validation/test transforms:

- Resize
- Grayscale to 3 channels
- Tensor conversion
- ImageNet normalization

Validation and test transforms intentionally avoid augmentation.

## 8. Non-IID Federated Simulation

Implemented in:

```text
core/non_iid.py
```

Supported strategies:

### Balanced IID Split

Class samples are distributed as evenly as possible across simulated hospitals.

Use when studying an idealized FL baseline.

### Label-Skew Split

Class proportions differ across hospitals.

Example:

- Hospital A sees mostly normal cases.
- Hospital B sees mostly pneumonia cases.
- Hospital C sees a mixed distribution.

This simulates institutional differences in disease prevalence and case mix.

### Quantity-Skew Split

Hospitals receive different sample counts.

Example:

- Large hospital has many X-rays.
- Small hospital has few X-rays.

This simulates unequal hospital size and data availability.

### Configuration

Important fields:

- `num_hospitals`
- `non_iid_strategy`
- `imbalance_severity`
- `seed`

### Why Non-IID Matters

Federated learning is harder when clients have different data distributions. A model update from one hospital may point in a different optimization direction than another hospital's update. This can slow convergence, destabilize aggregation, and reduce fairness across sites.

FedProx is included because it is designed to reduce local drift under data heterogeneity.

## 9. Model Architecture

Implemented in:

```text
core/model_loader.py
```

The primary model is:

```text
torchvision.models.densenet121
```

The classifier head is replaced with:

```python
nn.Linear(in_features, 1)
```

This produces one logit for binary classification.

### Binary Classification Convention

Class mapping:

```text
NORMAL = 0
PNEUMONIA = 1
```

The model outputs a raw logit. The sigmoid function is applied only during inference and evaluation:

```python
probability_positive = sigmoid(logit)
```

Training uses:

```python
BCEWithLogitsLoss
```

This is numerically more stable than applying sigmoid before binary cross entropy.

### Pretrained Weights

The loader supports ImageNet pretrained DenseNet121 initialization. This is useful for small medical imaging datasets, but ImageNet pretraining is not a substitute for medical validation.

## 10. Training Pipeline

Implemented in:

```text
core/trainer.py
```

The local trainer supports:

- DenseNet121 binary training
- BCEWithLogitsLoss
- Class weighting
- Weighted sampler
- FedProx proximal term
- Early stopping
- Threshold tuning
- Best checkpoint saving
- Model metadata saving

### Training Objective

For local-only or FedAvg local training:

```text
Loss = BCEWithLogitsLoss(logits, labels)
```

For FedProx local training:

```text
Loss = BCEWithLogitsLoss(logits, labels) + (mu / 2) * ||w_local - w_global||^2
```

### Class Imbalance Handling

Two options are available:

1. Positive class weighting in `BCEWithLogitsLoss`
2. Weighted sampling during training

Class weighting is useful when pneumonia and normal counts are imbalanced.

### Early Stopping

Early stopping can monitor:

- Validation loss
- Validation accuracy
- ROC-AUC
- F1-score
- Sensitivity

The default focus is validation loss or a configured validation metric.

### Checkpoint Metadata

Best checkpoints store:

- `state_dict`
- Architecture
- Class names
- Threshold
- Image size
- Metrics
- Training config
- Creation date

## 11. Federated Learning Methodology

Implemented in:

```text
core/fl_engine.py
```

The federated engine supports:

- Round tracking
- Client model distribution
- Partial participation
- Local client training
- Sample-weighted aggregation
- FedAvg
- FedProx
- Optional update clipping/noise simulation
- Global validation metrics

### Round Workflow

Each FL round follows:

1. Select participating clients.
2. Broadcast current global model.
3. Each selected client trains locally.
4. Each client reports:
   - hospital ID
   - sample count
   - local loss
   - local accuracy
   - validation metrics
5. Server aggregates completed updates.
6. Global model is evaluated.
7. Round metadata is stored.

### Partial Participation

The fraction of hospitals participating in a round is controlled by:

```text
participation_fraction
```

Example:

```text
participation_fraction = 0.67
```

means about two thirds of available hospitals are selected per round.

### Weighted FedAvg Formula

Naive uniform averaging is not used.

The implemented aggregation is:

```text
w_global = sum_k ((n_k / N) * w_k)
N = sum_k n_k
```

where:

- `w_k` is client `k` model weights
- `n_k` is client `k` local sample count
- `N` is the total number of samples across participating clients

This gives larger clients proportionally more influence.

### FedProx Formula

FedProx uses the same weighted aggregation on the server, but changes local client training:

```text
Loss = BCE + (mu / 2) * ||w_local - w_global||^2
```

where:

- `mu` controls proximal regularization strength
- `w_global` is the frozen global model sent at the start of the round
- `w_local` is the current local model

The proximal term discourages client updates from drifting too far from the global model.

### Client Dropout And Invalid Updates

Aggregation ignores clients that:

- did not complete local training
- report zero samples
- have missing updates

If no completed clients remain, aggregation raises a clear error.

## 12. Experimental Evaluation

Implemented in:

```text
core/experiment_runner.py
run_experiments.py
```

The experiment runner can compare:

- Local-only training
- Centralized training
- Federated FedAvg
- Federated FedProx

### Local-Only Baseline

Each simulated hospital trains its own model using only its local split. Models are evaluated on the common test split.

This baseline answers:

```text
How well does a hospital do without collaboration?
```

### Centralized Baseline

All registered training data is treated as if it were centrally available.

This baseline is useful as an upper-reference prototype, but it does not represent a privacy-preserving deployment.

### Federated FedAvg

Hospitals train locally. The server aggregates models with sample-weighted FedAvg.

### Federated FedProx

Hospitals train locally with the FedProx proximal term. The server aggregates models with sample-weighted aggregation.

### Experiment Outputs

Reports are written to:

```text
reports/experiments/<run_id>/
```

Typical artifacts:

- `report.json`
- `metrics_summary.csv`
- `per_round_metrics.csv`
- `client_level_metrics.csv`
- `convergence.png`
- `<run_id>_config.json`

## 13. Medical AI Metrics

Implemented in:

```text
core/metrics.py
```

The project does not rely only on accuracy.

Metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Sensitivity
- Specificity
- False negatives
- False positives
- Confusion matrix

### Confusion Matrix Convention

Rows are true labels and columns are predicted labels:

```text
              Pred NORMAL    Pred PNEUMONIA
True NORMAL       TN              FP
True PNEUMONIA    FN              TP
```

### Sensitivity

Sensitivity is recall for pneumonia:

```text
sensitivity = TP / (TP + FN)
```

High sensitivity is important because false negatives mean pneumonia cases were missed.

### Specificity

Specificity measures correct identification of normal cases:

```text
specificity = TN / (TN + FP)
```

### False Negatives

False negatives are explicitly shown in UI tables and reports because they are clinically serious in pneumonia screening.

## 14. Threshold Tuning

The default threshold of `0.5` is not always optimal for medical screening.

The project supports:

- `best_f1`
- `high_sensitivity`
- `balanced`
- `fixed_0_5`

### Best F1

Selects the threshold that maximizes F1-score on validation data.

### High Sensitivity

Prioritizes pneumonia recall. This may increase false positives.

### Balanced Sensitivity/Specificity

Selects a threshold that balances sensitivity and specificity.

### Fixed 0.5

Uses the standard threshold without tuning.

The selected threshold is saved in model metadata and used for evaluation and inference.

## 15. Grad-CAM Explanation Module

Implemented in:

```text
core/gradcam_engine.py
ui/pages/gradcam_page.py
```

Grad-CAM target layer for DenseNet121:

```text
features.norm5
```

Outputs:

- Original image
- Heatmap
- Overlay
- Side-by-side comparison

Saved under:

```text
data/visualizations/
```

### Grad-CAM Disclaimer

Grad-CAM is an explanatory aid only.

It is not:

- clinical proof
- a lesion segmentation tool
- a diagnostic guarantee
- a substitute for radiologist review

## 16. Database Design

Implemented in:

```text
core/db.py
```

The database is SQLite:

```text
database/hospital_client.db
```

Migrations are idempotent and run during app startup.

### Main Tables

#### `hospital_profile`

Stores hospital registry entries:

- hospital ID
- name
- location
- node status

#### `datasets`

Stores dataset registration metadata:

- dataset path
- sample count
- class count
- split counts
- class distribution
- warnings
- imbalance ratio

#### `images`

Stores image references:

- dataset ID
- file path
- label
- split

#### `models`

Legacy/general model records.

#### `model_versions`

Tracks richer model metadata:

- architecture
- version
- file path
- source
- aggregation algorithm
- threshold
- metrics JSON
- training config JSON

#### `training_runs`

Stores local training run metadata and history.

#### `federated_rounds`

Stores FL round records:

- run ID
- project ID
- round number
- aggregation algorithm
- participation fraction
- participating clients
- sample counts
- global metrics
- status

#### `client_updates`

Stores client update metadata:

- hospital ID
- round number
- sample count
- local loss
- local accuracy
- local metrics
- update status

#### `experiment_runs`

Stores experiment-level metadata:

- run ID
- run name
- experiment type
- algorithm
- seed
- config JSON
- environment JSON
- status
- summary JSON

#### `evaluation_metrics`

Stores scalar metrics:

- accuracy
- precision
- recall
- F1
- ROC-AUC
- sensitivity
- specificity
- false negatives
- false positives
- threshold

#### `confusion_matrices`

Stores:

- true negatives
- false positives
- false negatives
- true positives

#### `dataset_distributions`

Stores per-split and per-hospital distribution summaries.

#### `activity_logs`

Stores user-facing activity events.

## 17. User Interface Guide

The UI is implemented in:

```text
ui/pages/
```

### Dashboard

Shows:

- hospital/node information
- local sample count
- latest model
- last prediction
- current FL round
- server URL
- activity logs
- FL network visualizer

### Dataset Management

Allows:

- selecting dataset folder
- optional CSV labels
- setting train/validation/test ratios
- setting random seed
- registering dataset
- viewing warnings and class distributions

### Local Training

Allows:

- selecting FedAvg or FedProx local objective
- setting epochs
- setting batch size
- setting learning rate
- setting FedProx `mu`
- selecting threshold strategy
- enabling class weighting
- enabling weighted sampler

Shows:

- training loss
- training accuracy
- validation loss
- validation accuracy
- F1
- sensitivity
- specificity
- false negatives
- false positives
- selected threshold

### FL Project Runner

Allows:

- creating a federated project
- selecting hospitals
- selecting FedAvg or FedProx
- setting rounds
- setting local epochs
- setting participation fraction
- selecting non-IID strategy
- setting imbalance severity

Shows:

- current round
- participating hospitals
- local training updates
- sample counts
- global validation metrics
- false negatives and false positives

### FL Sync

Allows:

- pinging the mock server
- checking current round
- fetching global model
- sending local model update

This page is explicitly labelled as a prototype. It does not claim secure aggregation.

### Prediction

Allows:

- loading an X-ray image
- running inference
- viewing predicted label
- viewing probability and threshold
- saving prediction metadata

### Grad-CAM

Allows:

- loading an image
- generating a Grad-CAM overlay
- viewing original/overlay comparison
- saving visualization files

### Results

Shows:

- prediction history
- dataset distributions
- evaluation metrics
- confusion matrices
- model version history

False negatives are highlighted with tooltips.

## 18. Mock FL Server

Implemented in:

```text
server/mock_fl_server.py
```

Start with:

```powershell
python -m uvicorn server.mock_fl_server:app --reload
```

For a LAN demo where other machines can connect, run the server machine with:

```powershell
uvicorn server.mock_fl_server:app --host 0.0.0.0 --port 8000
```

Alternatively, use the configured host/port from `config/app_config.json`:

```powershell
python run_server.py
```

Then clients on the same LAN can use:

```powershell
python run_client.py --hospital-id hospital_1 --server-url http://SERVER_IP:8000 --dataset ./data/hospital_1
```

Endpoints:

```text
GET /health
POST /fl/register-client
GET /fl/global-model
POST /fl/upload-update
GET /fl/project-status
POST /fl/aggregate-round
GET /fl/current-round
GET /models/latest
GET /files/{filename}
POST /fl/send-update
```

The `/fl/*` endpoints support a simple multi-machine FL workflow:

1. A client registers with a unique `hospital_id`.
2. The client downloads the current global model.
3. The client trains locally.
4. The client uploads its update for the current round.
5. The server validates parameter names and tensor shapes.
6. The server aggregates only after the minimum client threshold is reached.
7. The server saves a new global model for the next round.

The older `/models/latest` and `/fl/send-update` endpoints remain for UI compatibility.

The mock server is for academic testing only. It does not implement:

- secure aggregation
- authentication beyond simple token headers in the client
- differential privacy
- robust production storage
- audit-grade networking

### Multi-Machine Client CLI

Implemented in:

```text
run_client.py
```

Example:

```powershell
python run_client.py --hospital-id hospital_1 --server-url http://SERVER_IP:8000 --dataset ./data/hospital_1 --epochs 1 --algorithm FedAvg
```

Useful options:

```text
--hospital-id          unique client/hospital ID
--server-url           URL of the server machine
--dataset              local dataset folder with NORMAL/PNEUMONIA subfolders
--algorithm            FedAvg or FedProx
--security-mode        none, secure_agg_sim, or he_demo
--wait-for-clients     wait until N clients are registered
--delay-upload         simulate delayed upload
--simulate-dropout     exit before upload
--request-aggregation  ask server to aggregate after upload
--min-clients          minimum clients required for aggregation
```

Example two-client LAN run:

Server machine:

```powershell
uvicorn server.mock_fl_server:app --host 0.0.0.0 --port 8000
```

Client machine 1:

```powershell
python run_client.py --hospital-id hospital_1 --server-url http://SERVER_IP:8000 --dataset ./data/hospital_1 --wait-for-clients 2
```

Client machine 2:

```powershell
python run_client.py --hospital-id hospital_2 --server-url http://SERVER_IP:8000 --dataset ./data/hospital_2 --wait-for-clients 2 --request-aggregation --min-clients 2
```

### Upload Validation

The server rejects updates when:

- client is not registered
- round number is stale or from the future
- sample count is zero
- uploaded checkpoint cannot be loaded
- parameter names differ from the global model
- tensor shapes differ from the global model

### Dropout And Delay Simulation

Client dropout:

```powershell
python run_client.py --hospital-id hospital_1 --server-url http://SERVER_IP:8000 --dataset ./data/hospital_1 --simulate-dropout
```

Delayed upload:

```powershell
python run_client.py --hospital-id hospital_1 --server-url http://SERVER_IP:8000 --dataset ./data/hospital_1 --delay-upload 30
```

The server reports completed and dropped clients through:

```text
GET /fl/project-status
```

## 19. Reproducibility

Implemented in:

```text
core/reproducibility.py
```

The project records:

- run ID
- random seed
- deterministic PyTorch setting
- Python version
- platform information
- package versions
- experiment config
- output paths

Reproducibility depends on:

- same dataset
- same split ratios
- same random seed
- same non-IID strategy
- same model initialization
- same training parameters
- same package versions

## 20. Privacy And Security Limitations

This project is honest about privacy.

The simulated FL workflow does not upload raw image files during local training. However, this does not mean it is secure.

Important limitations:

- Model weights and gradients can leak information.
- Secure aggregation is not implemented.
- Differential privacy is not implemented by default.
- Optional clipping/noise is a simple simulation option, not a formal DP mechanism.
- The mock server is not production-hardened.
- There is no formal threat model.
- There is no cryptographic protection of updates.

Do not describe this project as secure federated learning. More accurate terms:

- Federated Aggregation Prototype
- Privacy-Preserving Training Simulation
- Federated Learning Research Prototype

### Secure Aggregation Simulation

Implemented in:

```text
core/secure_aggregation.py
```

Enable with:

```json
"security_mode": "secure_agg_sim"
```

or:

```powershell
python run_client.py --security-mode secure_agg_sim ...
```

In this mode, each client masks its weighted model contribution:

```text
masked_update_k = n_k * w_k + mask_k
```

The server aggregates:

```text
sum(masked_update_k) / sum(n_k)
```

Pairwise masks are generated so that masks cancel over the cohort. If a client drops out, the prototype removes only the aggregate residual mask during aggregation. This is done to demonstrate the idea of secure aggregation, not to provide cryptographic security.

Important caveats:

- There is no real key exchange.
- The server coordinates enough metadata to run the simulation.
- There is no collusion resistance.
- There is no authenticated client identity.
- This is not Bonawitz-style secure aggregation.
- This should not be presented as production secure aggregation.

### Homomorphic Encryption Demo Mode

Enable with:

```powershell
python run_client.py --security-mode he_demo ...
```

This mode attempts a tiny Paillier additive homomorphic encryption demo using the optional `phe` package. It encrypts only a toy vector of metrics or values. It does not encrypt DenseNet model weights.

Full homomorphic encryption of neural-network weights is not implemented because it is computationally expensive and outside the intended scope of this prototype.

## 21. Robustness And Error Handling

The project includes handling for:

- empty datasets
- missing class folders
- invalid image files
- incompatible checkpoints
- corrupted checkpoints
- server unavailable
- missing local update checkpoints
- zero-client aggregation
- client training failure
- missing local loaders
- zero-sample client updates

User-facing errors are shown through UI dialogs or activity logs.

## 22. Reports And Artifacts

### Prediction Reports

Prediction JSON files are saved under:

```text
data/predictions/
```

### Grad-CAM Outputs

Saved under:

```text
data/visualizations/
```

Files include:

- original image
- heatmap
- overlay
- comparison image

### Experiment Reports

Saved under:

```text
reports/experiments/<run_id>/
```

Typical files:

```text
report.json
metrics_summary.csv
per_round_metrics.csv
client_level_metrics.csv
convergence.png
<run_id>_config.json
```

## 23. How To Run Experiments

First register a dataset in the UI.

Then run:

```powershell
.\.venv\Scripts\python.exe run_experiments.py --methods local,centralized,fedavg,fedprox --rounds 3 --num-hospitals 3 --split label_skew --severity 0.7 --participation 0.67 --threshold high_sensitivity --seed 42
```

Minimal FedAvg/FedProx smoke experiment:

```powershell
python run_experiments.py --methods fedavg,fedprox --rounds 1 --num-hospitals 2
```

Repeated seed experiment:

```powershell
python run_experiments.py --methods fedavg,fedprox --rounds 1 --num-hospitals 2 --repeats 3
```

Repeated seed outputs include mean and standard deviation summaries under:

```text
reports/experiments/repeated_seed_summary_<seed>_<repeats>/
```

Arguments:

```text
--methods          local,centralized,fedavg,fedprox
--rounds           number of FL rounds
--local-epochs     local epochs per training call
--num-hospitals    simulated hospital count
--split            balanced_iid, label_skew, quantity_skew
--severity         non-IID severity from 0.0 to 0.99
--participation    client participation fraction
--seed             random seed
--batch-size       batch size
--lr               learning rate
--fedprox-mu       FedProx proximal coefficient
--threshold        best_f1, high_sensitivity, balanced, fixed_0_5
--security-mode    none, secure_agg_sim, he_demo
--repeats          number of repeated seed runs
```

Example IID FedAvg only:

```powershell
.\.venv\Scripts\python.exe run_experiments.py --methods fedavg --rounds 5 --num-hospitals 4 --split balanced_iid --seed 42
```

Example label-skew FedProx:

```powershell
.\.venv\Scripts\python.exe run_experiments.py --methods fedprox --rounds 5 --num-hospitals 4 --split label_skew --severity 0.8 --fedprox-mu 0.01 --seed 42
```

## 24. How To Reproduce Results

To reproduce a run:

1. Use the same dataset folder.
2. Register the dataset with the same split ratios.
3. Use the same random seed.
4. Use the same CLI arguments.
5. Use the same package versions if possible.
6. Compare the exported `<run_id>_config.json`.

The run config and environment are exported automatically.

## 25. Troubleshooting

### No Images Found

Check folder structure:

```text
NORMAL/
PNEUMONIA/
```

Check image extensions.

### Missing Class Warning

Make sure class folder names are exactly:

```text
NORMAL
PNEUMONIA
```

### Incompatible Checkpoint

The expected model is DenseNet121 with one output logit.

If loading fails, verify that the checkpoint contains:

```text
state_dict
```

and was trained with a compatible classifier head.

### Server Unavailable

Start the mock server:

```powershell
python -m uvicorn server.mock_fl_server:app --reload
```

Then verify:

```text
http://127.0.0.1:8000/health
```

### Grad-CAM Hook Error

The Grad-CAM engine handles DenseNet's in-place activation by cloning the hooked activation. If errors continue, verify the loaded model is compatible DenseNet121.

### Zero Clients In Aggregation

Check:

- participation fraction is not too low
- hospitals have non-empty datasets
- clients completed local training

## 26. Limitations

Current limitations:

- Single-machine simulation.
- Not a deployed real multi-hospital FL network.
- No formal privacy guarantee.
- No secure aggregation.
- No formal differential privacy accounting.
- No calibration analysis yet.
- No external clinical validation.
- No radiologist benchmark.
- No prospective testing.
- Centralized baseline is simulated using locally available data.
- Results depend heavily on dataset quality and split protocol.

## 27. Future Work

Recommended next steps:

- Add repeated-seed experiment runner.
- Add confidence intervals for metrics.
- Add calibration metrics and reliability diagrams.
- Add external validation set workflow.
- Add formal DP-SGD with privacy accounting.
- Add secure aggregation protocol.
- Add multi-label chest X-ray classification.
- Add model cards for saved checkpoints.
- Add experiment comparison dashboard.
- Add fairness analysis across hospitals.
- Add per-client drift and update-norm monitoring.
- Add server-side authentication and authorization.
- Add encrypted transport and production deployment configuration.

## Final Academic Note

This project is best described as an academic FL simulation and medical AI evaluation prototype. It demonstrates core ideas and supports structured experiments, but it should not be presented as a clinically validated or secure production system.
