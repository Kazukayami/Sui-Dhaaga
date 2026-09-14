import json
from pathlib import Path

import numpy as np
from PIL import Image

from config import (
    CLASS_DISTRIBUTION_PATH,
    CLEAN_MIN_CONFIDENCE,
    DATASET_AUDIT_PATH,
    DEFAULT_CLASS_DESCRIPTIONS,
    GRADCAM_DIR,
    METADATA_PATH,
    MODELS_DIR,
    OUTPUTS_DIR,
    QUALITY_TIERS,
    REPORTS_DIR,
    UNKNOWN_CONFIDENCE_THRESHOLD,
    UNKNOWN_MARGIN_THRESHOLD,
)


def ensure_project_dirs() -> None:
    for path in (MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR, GRADCAM_DIR):
        path.mkdir(parents=True, exist_ok=True)


def save_metadata(
    class_names: list[str],
    model_name: str,
    class_weights: dict[int, float] | None = None,
) -> None:
    ensure_project_dirs()
    payload = {
        "class_names": class_names,
        "model_name": model_name,
        "unknown_confidence_threshold": UNKNOWN_CONFIDENCE_THRESHOLD,
        "unknown_margin_threshold": UNKNOWN_MARGIN_THRESHOLD,
        "clean_min_confidence": CLEAN_MIN_CONFIDENCE,
        "class_descriptions": {
            name: DEFAULT_CLASS_DESCRIPTIONS.get(name, "Fabric surface condition")
            for name in class_names
        },
        "class_weights": {
            class_names[int(index)]: float(weight)
            for index, weight in (class_weights or {}).items()
        },
    }
    METADATA_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model metadata not found. Train the model first so class labels are saved."
        )
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def pil_to_model_array(image_pil: Image.Image, img_size: int) -> np.ndarray:
    rgb = image_pil.convert("RGB").resize((img_size, img_size))
    arr = np.asarray(rgb, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def validate_image(image_pil: Image.Image) -> None:
    if image_pil.width < 32 or image_pil.height < 32:
        raise ValueError(
            f"Image too small: {image_pil.size}. Minimum 32x32 pixels required."
        )


def severity_from_prediction(predicted_class: str, confidence: float) -> tuple[str, int]:
    if predicted_class in {"unknown_defect", "review"}:
        return ("Unknown / Review", 2)
    if confidence < UNKNOWN_CONFIDENCE_THRESHOLD or (
        predicted_class == "clean" and confidence < CLEAN_MIN_CONFIDENCE
    ):
        return ("Review", 1)
    if predicted_class == "clean":
        return ("Pass", 0)
    if confidence >= 0.92:
        return ("Critical", 3)
    if confidence >= 0.75:
        return ("Major", 2)
    return ("Review", 1)


def quality_score(predicted_class: str, confidence: float) -> int:
    """Return a 0-100 textile quality score.

    Clean, high-confidence fabric receives a high score. Defect confidence is
    treated as inverse quality evidence, so stronger defect evidence lowers the
    score. Unknown/review cases remain intentionally borderline.
    """
    if predicted_class in {"unknown_defect", "review"}:
        return 45
    if confidence < UNKNOWN_CONFIDENCE_THRESHOLD or (
        predicted_class == "clean" and confidence < CLEAN_MIN_CONFIDENCE
    ):
        return 55
    if predicted_class == "clean":
        return int(round(90 + confidence * 10))
    penalty = min(70, int(round(confidence * 70)))
    return max(0, 100 - penalty)


def quality_tier(score: int) -> str:
    for lower, upper, label in QUALITY_TIERS:
        if lower <= score <= upper:
            return label
    return "Grade D - Reject"


def readable_label(label: str) -> str:
    return label.replace("_", " ").title()


def safe_stem(path: str | Path) -> str:
    return Path(path).stem.replace(" ", "_")


def prediction_entropy(probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities.astype(np.float64), 1e-9, 1.0)
    return float(-(clipped * np.log(clipped)).sum())


def is_unknown_prediction(probabilities: np.ndarray, predicted_class: str) -> tuple[bool, dict]:
    ordered = np.sort(probabilities)[::-1]
    confidence = float(ordered[0])
    margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else confidence
    entropy = prediction_entropy(probabilities)
    unknown = confidence < UNKNOWN_CONFIDENCE_THRESHOLD or margin < UNKNOWN_MARGIN_THRESHOLD
    if predicted_class == "clean" and confidence < CLEAN_MIN_CONFIDENCE:
        unknown = True
    return unknown, {
        "confidence": confidence,
        "margin": margin,
        "entropy": entropy,
        "confidence_threshold": UNKNOWN_CONFIDENCE_THRESHOLD,
        "margin_threshold": UNKNOWN_MARGIN_THRESHOLD,
    }
