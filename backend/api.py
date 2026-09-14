from functools import lru_cache
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from config import (
    BINARY_CONFUSION_MATRIX_PATH,
    BINARY_REPORT_PATH,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    MODEL_PATH,
    ROOT_DIR,
    TRAINING_CURVES_PATH,
)
from utils import quality_score, severity_from_prediction

FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(
    title="Textile Defect Intelligence API",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@lru_cache(maxsize=1)
def get_predictor():
    from predict import TextilePredictor

    return TextilePredictor()


@lru_cache(maxsize=1)
def get_gradcam():
    from gradcam import GradCAM

    return GradCAM()


@app.get("/")
def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_ready": MODEL_PATH.exists(),
    }


@app.get("/api/model")
def model_info() -> dict:
    if not MODEL_PATH.exists():
        return {
            "ready": False,
            "message": "Model not trained yet.",
            "classes": [],
        }
    predictor = get_predictor()
    return {
        "ready": True,
        "model_name": predictor.metadata["model_name"],
        "classes": predictor.class_names,
        "descriptions": predictor.metadata["class_descriptions"],
    }


@app.get("/api/artifacts")
def artifacts() -> dict:
    return {
        "training_curves": TRAINING_CURVES_PATH.exists(),
        "confusion_matrix": CONFUSION_MATRIX_PATH.exists(),
        "classification_report": CLASSIFICATION_REPORT_PATH.exists(),
        "binary_confusion_matrix": BINARY_CONFUSION_MATRIX_PATH.exists(),
        "binary_report": BINARY_REPORT_PATH.exists(),
    }


@app.get("/api/artifacts/training-curves")
def training_curves():
    if not TRAINING_CURVES_PATH.exists():
        raise HTTPException(status_code=404, detail="Training curves not generated yet.")
    return FileResponse(TRAINING_CURVES_PATH)


@app.get("/api/artifacts/confusion-matrix")
def confusion_matrix():
    if not CONFUSION_MATRIX_PATH.exists():
        raise HTTPException(status_code=404, detail="Confusion matrix not generated yet.")
    return FileResponse(CONFUSION_MATRIX_PATH)


@app.get("/api/artifacts/binary-confusion-matrix")
def binary_confusion_matrix():
    if not BINARY_CONFUSION_MATRIX_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Binary confusion matrix not generated yet.",
        )
    return FileResponse(BINARY_CONFUSION_MATRIX_PATH)


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Train the model before inference.")
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Upload a JPG or PNG image.")

    try:
        payload = await file.read()
        image = Image.open(BytesIO(payload)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    predictor = get_predictor()
    result = predictor.predict_pil(image)
    severity, severity_rank = severity_from_prediction(
        result["predicted_class"],
        result["confidence"],
    )
    gradcam_path = get_gradcam().generate(image, Path(file.filename or "inspection").stem)

    return {
        **result,
        "quality_score": quality_score(
            result["predicted_class"],
            result["confidence"],
        ),
        "decision": severity,
        "decision_rank": severity_rank,
        "gradcam_url": f"/api/gradcam/{gradcam_path.name}",
    }


@app.get("/api/gradcam/{filename}")
def gradcam_image(filename: str):
    path = ROOT_DIR / "outputs" / "gradcam" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Grad-CAM image not found.")
    return FileResponse(path)
