from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from config import IMG_SIZE, MODEL_PATH
from train_model import RandomGaussianNoise, RandomSoftBlur
from utils import is_unknown_prediction, load_metadata, pil_to_model_array, readable_label


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

    def predict_pil(self, image_pil: Image.Image) -> dict:
        probs = self.model.predict(pil_to_model_array(image_pil, IMG_SIZE), verbose=0)[0]
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

    def predict_path(self, image_path: str | Path) -> dict:
        with Image.open(image_path) as image_pil:
            return self.predict_pil(image_pil)


if __name__ == "__main__":
    predictor = TextilePredictor()
    result = predictor.predict_path("sample.jpg")
    print(result)
