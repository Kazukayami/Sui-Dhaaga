# Explainable AI-Based Intelligent Textile Defect Detection

Advanced prototype for textile inspection using EfficientNetV2B0 transfer learning, Grad-CAM/Integrated Gradients explainability, and a Streamlit inspection dashboard.

The current trained workflow uses AFID-derived textile classes plus the merged Kaggle fabric-defect archive. The active model recognizes 17 categories:

- `broken_end`
- `broken_pick`
- `broken_yarn`
- `clean`
- `contamination`
- `crease`
- `cut_selvage`
- `fuzzyball`
- `hole`
- `horizontal_defect`
- `knots`
- `line_defect`
- `nep`
- `vertical_defect`
- `warp_ball`
- `weft_crack`
- `weft_curling`

## What is included

- EfficientNetV2B0 transfer learning with warm-up and fine-tuning phases
- Shared metadata so prediction labels stay aligned with the trained dataset
- Evaluation outputs for conference-paper use
- Grad-CAM overlays and Integrated Gradients support for explainability
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

## Prepare or merge datasets

After downloading the AFID archive into `dataset_raw/`, build cropped train/validation/test samples with:

```bash
python prepare_afid_dataset.py --overwrite
```

The AFID script will not delete an existing merged dataset unless `--overwrite` is passed.

To merge the Kaggle archive used in the latest training run:

```powershell
C:\tdxai_env\Scripts\python.exe merge_kaggle_archive.py C:\Users\NISHCHAY\Downloads\archive.zip
C:\tdxai_env\Scripts\python.exe remove_dataset_leakage.py
C:\tdxai_env\Scripts\python.exe dataset_tools.py --audit
```

## Setup

Tested with Python 3.11. TensorFlow 2.16.1 is not recommended with Python 3.12+ for this project.

```bash
pip install -r requirements.txt
```

On Windows, this project was successfully run from a short-path virtual environment to avoid TensorFlow path-length issues:

```powershell
python -m venv C:\tdxai_env
C:\tdxai_env\Scripts\python.exe -m pip install -r requirements.txt
```

## Train

```bash
python train_model.py
```

Windows short-path environment:

```powershell
C:\tdxai_env\Scripts\python.exe train_model.py
```

This creates:

- `models/textile_model.keras`
- timestamped backup models such as `models/textile_model_YYYYMMDD_HHMMSS.keras`
- `models/metadata.json`
- `outputs/reports/training_history.csv`
- `outputs/reports/training_curves.png`

## Evaluate

```bash
python evaluate_model.py
```

Windows short-path environment:

```powershell
C:\tdxai_env\Scripts\python.exe evaluate_model.py
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
C:\tdxai_env\Scripts\python.exe -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
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
