# S.H.I.E.L.D: Edge-XAI Model

Cardiovascular stress prediction using Bidirectional LSTM with Federated Learning and SHAP-based explainability.

Calibrated against **UCI Heart Disease Dataset** (920 patients, 4 medical centers) and **MIT-BIH Arrhythmia Database** (47 patients, ECG-derived HR/HRV).

---

## What This Does

Takes 8 physiological features from wearable sensors and predicts a cardiovascular stress score (0-100%) with human-readable reason codes explaining the prediction.

**Input:** HR, Systolic BP, HRV, Diastolic BP, SpO2, Activity Level, Age, Sleep Hours (60-timestep window)

**Output:** Stress score (0-100%) + reason codes (e.g. `HIGH_HR`, `LOW_SPO2`, `HIGH_BP_S`)

---

## Quick Start

### 1. Generate Training Data
```bash
pip install numpy pandas scikit-learn wfdb
python calibrate_dataset.py
```
Loads UCI and MIT-BIH files, generates 200 patients across 5 hospitals, outputs `shield_training_dataset.csv`.

### 2. Train the Model
Open `lstm_training_pipeline.ipynb` on Google Colab (GPU runtime recommended).

Upload `shield_training_dataset.csv`, run all cells. Takes ~45-60 minutes.

The notebook:
- Trains a centralized LSTM baseline
- Runs 10-round FL simulation (5 hospital clients, FedAvg)
- Computes SHAP feature importance
- Exports deployment files to `outputs/`

### 3. Deploy to App
Copy these 3 files from `outputs/` to the React Native app's `assets/models/`:
- `shield.tflite` — the model
- `scaler_stats.json` — normalization parameters
- `shield_weights.json` — SHAP feature weights for reason codes

Update one line in the inference code:
```typescript
// Change shield_v2.tflite → shield.tflite
const modelAsset = Asset.fromModule(require('../assets/models/shield.tflite'));
```

### 4. FL Demo
```bash
pip install fastapi uvicorn tensorflow numpy requests pandas
python fl_demo.py
```
Starts the FL aggregator server, runs 3 hospital clients, demonstrates the full FedAvg lifecycle.

---

## Directory Structure

```
LSTM_Model/
├── datasets/
|   ├── uci_data/                      # UCI Heart Disease files (input)
│   │   ├── processed.cleveland.data
│   │   ├── processed.hungarian.data
│   │   ├── processed.switzerland.data
│   │   └── processed.va.data
├   └── mit_bih_data/                  # MIT-BIH ECG files (input)
│       └── *.dat, *.hea, *.atr
│
├── calibrate_dataset.py               # Generates calibrated training data
├── shield_training_dataset.csv        # Training data (~450MB, generated)
├── lstm_training_pipeline.ipynb       # Model training notebook
│
├── fl_server.py                       # FL aggregator server (FastAPI)
├── fl_client.py                       # FL hospital client
├── fl_demo.py                         # Automated FL demonstration
│
└── outputs/                           # Training outputs
    ├── shield.tflite                  # → app assets/models/
    ├── scaler_stats.json              # → app assets/models/
    ├── shield_weights.json            # → app assets/models/
    ├── shield_global.keras            # Full model (used by FL server)
    ├── training_history.png           # Training curves
    ├── roc_curve.png                  # ROC curve
    ├── confusion_matrix.png           # Confusion matrix
    └── fl_comparison.png              # FL vs centralized comparison
```

---

## Model Architecture

**Bidirectional LSTM** with dropout regularization:
- Input: (batch, 60, 8) — 60 timesteps, 8 features
- BiLSTM Layer 1: 64 units, return sequences
- BiLSTM Layer 2: 32 units, collapse sequence
- Dense: 32 → 16 → 1 (sigmoid)
- Loss: MSE | Optimizer: Adam | Output: 0-1 stress score

---

## Federated Learning

The FL framework has two components:

**Training simulation** (in the notebook): Partitions data by hospital, runs 10 rounds of FedAvg with 5 clients. Proves FL matches centralized performance.

**Deployable server** (`fl_server.py`): FastAPI service with REST endpoints:
- `POST /fl/submit` — clients submit locally-trained weights
- `GET /fl/global-model` — download current global TFLite model
- `GET /fl/status` — check server state and history
- `POST /fl/aggregate` — manually trigger FedAvg

Clients (`fl_client.py`) download the global model, train on local hospital data, and POST weight updates back.

---

## Reason Code Engine (XAI)

SHAP runs offline during training to compute global feature importance weights. At inference time on the device, the app uses a lightweight formula:

```
impact = |z_score(current_value)| × SHAP_weight
```

Features with high impact AND clinical abnormality become reason codes. Direction (HIGH/LOW) is determined by comparing against the population mean.

Clinical thresholds are activity-context-aware: HR of 140 during exercise is normal, at rest it triggers `HIGH_HR`.

---

## RAG Pipeline

The RAG component maps reason codes to natural language explanations:

1. Model produces reason code (e.g. `HIGH_HR`)
2. RAG queries ChromaDB vector database of clinical guidelines
3. Retrieved context is passed to an LLM
4. LLM generates patient-friendly explanation + clinician summary

Runs on the FastAPI cloud backend. Mean retrieval latency: 34.1ms, retrieval accuracy: 76%.

---

## Clinical Data Sources

| Source | What We Extract | Reference |
|--------|----------------|-----------|
| UCI Heart Disease (920 pts) | BP distributions, age-risk curve, max HR | Janosi et al. 1988 |
| MIT-BIH Arrhythmia (47 pts) | HR and HRV (RMSSD) from ECG | Moody & Mark 2001 |
| AHA/ACC 2025 Guidelines | BP classification thresholds | AHA/ACC 2025 |
| Shaffer & Ginsberg 2017 | HRV normative ranges | Front. Public Health |
| Hillebrand et al. 2013 | HRV-CV risk meta-analysis | Europace |
