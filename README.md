# Explainable AI-Based Intelligent Textile Defect Detection

Advanced prototype for textile inspection using transfer learning, Grad-CAM explainability, and a Streamlit inspection dashboard.

The bundled preparation workflow uses the public AFID textile dataset and builds a reproducible 4-class subset from the strongest available labels:

- `clean`
- `broken_yarn`
- `fuzzyball`
- `nep`

## What is included

- MobileNetV2 transfer learning with a fine-tuning phase
- Shared metadata so prediction labels stay aligned with the trained dataset
- Evaluation outputs for conference-paper use
- Grad-CAM overlays saved per inspection
- Streamlit dashboard with:
  - defect prediction
  - confidence score
  - quality score
  - production-style decision state
  - probability breakdown
  - explainability visualization

## Project layout

```text
textile-defect-xai/
|-- dataset/
|   |-- train/
|   |-- validation/
|   `-- test/
|-- models/
|-- outputs/
|   |-- gradcam/
|   `-- reports/
|-- app.py
|-- batch_predict.py
|-- config.py
|-- evaluate_model.py
|-- gradcam.py
|-- predict.py
|-- train_model.py
|-- utils.py
|-- requirements.txt
`-- README.md
```

## Dataset structure

```text
dataset/
|-- train/
|   |-- broken_yarn/
|   |-- clean/
|   |-- fuzzyball/
|   `-- nep/
|-- validation/
|   |-- broken_yarn/
|   |-- clean/
|   |-- fuzzyball/
|   `-- nep/
`-- test/
    |-- broken_yarn/
    |-- clean/
    |-- fuzzyball/
    `-- nep/
```

## Prepare the AFID dataset

After downloading the AFID archive into `dataset_raw/`, build cropped train/validation/test samples with:

```bash
python prepare_afid_dataset.py
```

## Setup

```bash
pip install -r requirements.txt
```

On Windows, this project was successfully run from a short-path virtual environment to avoid TensorFlow path-length issues:

```powershell
python -m venv C:\tfenv
C:\tfenv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train

```bash
python train_model.py
```

Windows short-path environment:

```powershell
C:\tfenv\Scripts\python.exe train_model.py
```

This creates:

- `models/textile_model.keras`
- `models/metadata.json`
- `outputs/reports/training_history.csv`
- `outputs/reports/training_curves.png`

## Evaluate

```bash
python evaluate_model.py
```

Windows short-path environment:

```powershell
C:\tfenv\Scripts\python.exe evaluate_model.py
```

This creates:

- `outputs/reports/classification_report.csv`
- `outputs/reports/binary_classification_report.csv`
- `outputs/reports/confusion_matrix.png`
- `outputs/reports/binary_confusion_matrix.png`

## Run the dashboard

```bash
streamlit run app.py
```

## Run the full web application

```bash
uvicorn backend.api:app --reload
```

Windows short-path environment:

```powershell
C:\tfenv\Scripts\python.exe -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Open the browser at:

```text
http://127.0.0.1:8000
```

The FastAPI backend exposes:

- `GET /api/health`
- `GET /api/model`
- `POST /api/predict`
- `GET /api/artifacts`

## Batch inference

```bash
python batch_predict.py path/to/images outputs/batch_predictions.csv
```

## Research-paper outputs to capture

- Accuracy and loss curves from `outputs/reports/training_curves.png`
- Confusion matrix from `outputs/reports/confusion_matrix.png`
- Binary defect-detection confusion matrix from `outputs/reports/binary_confusion_matrix.png`
- Per-class metrics from `outputs/reports/classification_report.csv`
- Uploaded sample, decision output, confidence result, and Grad-CAM overlay from the dashboard

## Suggested next upgrades

- Add EfficientNetV2 and ResNet50 comparison experiments
- Add batch-folder inference with CSV export
- Add webcam/live-line inspection
- Add defect localization masks if segmentation labels become available
- Add experiment tracking and model versioning
