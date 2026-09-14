import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from predict import TextilePredictor
from utils import quality_score, severity_from_prediction


def run_batch(input_dir: Path, output_csv: Path) -> None:
    predictor = TextilePredictor()
    rows = []
    for image_path in sorted(input_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with Image.open(image_path) as image_pil:
            result = predictor.predict_pil(image_pil)
        severity, _ = severity_from_prediction(
            result["predicted_class"],
            result["confidence"],
        )
        rows.append(
            {
                "file": image_path.name,
                "predicted_class": result["predicted_class"],
                "confidence": result["confidence"],
                "quality_score": quality_score(
                    result["predicted_class"],
                    result["confidence"],
                ),
                "decision": severity,
            }
        )
    pd.DataFrame(rows).to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch textile defect inference")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    run_batch(args.input_dir, args.output_csv)
