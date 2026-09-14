from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from tensorflow.keras.models import load_model

from config import IMG_SIZE, MODEL_PATH
from train_model import RandomGaussianNoise, RandomSoftBlur
from utils import (
    is_unknown_prediction,
    load_metadata,
    pil_to_model_array,
    readable_label,
    validate_image,
)

_predictor_instance = None


class TextilePredictor:
    def __init__(self) -> None:
        self.model = load_model(
            MODEL_PATH,
            custom_objects={
                "RandomGaussianNoise": RandomGaussianNoise,
                "RandomSoftBlur": RandomSoftBlur,
            },
        )
        self.metadata = load_metadata()
        self.class_names = self.metadata["class_names"]
        if "clean" not in self.class_names:
            raise ValueError(
                f"'clean' class not found in class_names: {self.class_names}. "
                "Rename the defect-free class folder to 'clean' and retrain."
            )

    def _format_prediction(self, probs: np.ndarray) -> dict:
        class_index = int(np.argmax(probs))
        predicted_class = self.class_names[class_index]
        clean_index = self.class_names.index("clean")
        clean_probability = float(probs[clean_index])
        defect_probability = float(1.0 - clean_probability)
        confidence = float(probs[class_index])
        is_unknown, uncertainty = is_unknown_prediction(probs, predicted_class)
        top_predictions = sorted(
            (
                {
                    "class_name": name,
                    "display_class": readable_label(name),
                    "probability": float(prob),
                }
                for name, prob in zip(self.class_names, probs)
            ),
            key=lambda item: item["probability"],
            reverse=True,
        )[:3]
        display_class = (
            "Unknown Defect / Manual Review"
            if is_unknown
            else readable_label(predicted_class)
        )
        return {
            "predicted_class": predicted_class,
            "display_class": display_class,
            "confidence": confidence,
            "defect_detected": predicted_class != "clean" or is_unknown,
            "defect_probability": defect_probability,
            "clean_probability": clean_probability,
            "needs_review": is_unknown,
            "unknown_defect": is_unknown,
            "uncertainty": uncertainty,
            "top_predictions": top_predictions,
            "probabilities": {
                name: float(prob) for name, prob in zip(self.class_names, probs)
            },
        }

    def predict_pil(self, image_pil: Image.Image, use_tta: bool = False) -> dict:
        validate_image(image_pil)
        if use_tta:
            return self.predict_with_tta(image_pil)
        probs = self.model.predict(pil_to_model_array(image_pil, IMG_SIZE), verbose=0)[0]
        return self._format_prediction(probs)

    def predict_with_tta(self, image_pil: Image.Image, n_augments: int = 6) -> dict:
        validate_image(image_pil)
        versions = [image_pil.convert("RGB"), ImageOps.mirror(image_pil.convert("RGB"))]
        for _ in range(max(0, n_augments - len(versions))):
            enhanced = ImageEnhance.Brightness(image_pil.convert("RGB")).enhance(
                random.uniform(0.9, 1.1)
            )
            enhanced = ImageEnhance.Contrast(enhanced).enhance(random.uniform(0.92, 1.08))
            versions.append(enhanced)
        all_probs = [
            self.model.predict(pil_to_model_array(version, IMG_SIZE), verbose=0)[0]
            for version in versions
        ]
        return self._format_prediction(np.mean(all_probs, axis=0))

    def predict_path(self, image_path: str | Path) -> dict:
        with Image.open(image_path) as image_pil:
            return self.predict_pil(image_pil)


def get_predictor() -> TextilePredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = TextilePredictor()
    return _predictor_instance


if __name__ == "__main__":
    predictor = TextilePredictor()
    result = predictor.predict_path("sample.jpg")
    print(result)
