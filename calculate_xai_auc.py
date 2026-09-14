import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageFilter

from config import IMG_SIZE, MODEL_PATH, TEST_DIR
from gradcam import GradCAM
from train_model import RandomGaussianNoise, RandomSoftBlur
from utils import load_metadata, pil_to_model_array


def collect_test_images(per_class: int | None) -> list[Path]:
    paths = []
    for class_dir in sorted(Path(TEST_DIR).iterdir()):
        if not class_dir.is_dir():
            continue
        class_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if per_class is not None:
            class_paths = class_paths[:per_class]
        paths.extend(class_paths)
    return paths


def normalize_map(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    heatmap -= float(heatmap.min())
    max_value = float(heatmap.max())
    if max_value > 1e-8:
        heatmap /= max_value
    return heatmap


def integrated_gradients_heatmap(model, img_array, class_index: int, n_steps: int) -> np.ndarray:
    img = tf.convert_to_tensor(img_array, dtype=tf.float32)
    baseline = tf.zeros_like(img)
    alphas = tf.linspace(0.0, 1.0, n_steps)
    interpolated = baseline + alphas[:, None, None, None] * (img - baseline)

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        preds = model(interpolated, training=False)
        outputs = preds[:, class_index]

    grads = tape.gradient(outputs, interpolated)
    avg_grads = tf.reduce_mean(grads, axis=0)
    attributions = (img[0] - baseline[0]) * avg_grads
    heatmap = tf.reduce_sum(tf.abs(attributions), axis=-1).numpy()
    return normalize_map(heatmap)


def auc_from_curve(values: list[float]) -> float:
    x = np.linspace(0.0, 1.0, len(values))
    return float(np.trapz(np.asarray(values, dtype=np.float32), x))


def perturbation_auc(model, image: np.ndarray, heatmap: np.ndarray, class_index: int, steps: int):
    flat_order = np.argsort(-heatmap.reshape(-1))
    baseline = (
        Image.fromarray(np.uint8(np.clip(image, 0, 255)))
        .filter(ImageFilter.GaussianBlur(radius=10))
    )
    baseline_arr = np.asarray(baseline, dtype=np.float32)
    h, w, c = image.shape
    total_pixels = h * w
    deletion_images = []
    insertion_images = []

    for fraction in np.linspace(0.0, 1.0, steps):
        k = int(round(total_pixels * fraction))
        mask = np.zeros(total_pixels, dtype=bool)
        if k > 0:
            mask[flat_order[:k]] = True
        mask = mask.reshape(h, w)

        deleted = image.copy()
        deleted[mask] = baseline_arr[mask]
        inserted = baseline_arr.copy()
        inserted[mask] = image[mask]
        deletion_images.append(deleted)
        insertion_images.append(inserted)

    deletion_batch = tf.convert_to_tensor(np.stack(deletion_images), dtype=tf.float32)
    insertion_batch = tf.convert_to_tensor(np.stack(insertion_images), dtype=tf.float32)
    deletion_scores = model.predict(deletion_batch, verbose=0)[:, class_index].tolist()
    insertion_scores = model.predict(insertion_batch, verbose=0)[:, class_index].tolist()
    return auc_from_curve(deletion_scores), auc_from_curve(insertion_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--steps", type=int, default=11)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--output", default="outputs/reports/xai_auc_report.csv")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata()
    class_names = metadata["class_names"]
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "RandomGaussianNoise": RandomGaussianNoise,
            "RandomSoftBlur": RandomSoftBlur,
        },
    )
    gradcam = GradCAM()
    paths = collect_test_images(args.per_class)
    rows = []

    for index, path in enumerate(paths, start=1):
        with Image.open(path) as image_pil:
            image_pil = image_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            img_array = pil_to_model_array(image_pil, IMG_SIZE)
            probs = model.predict(img_array, verbose=0)[0]
            class_index = int(np.argmax(probs))
            predicted_class = class_names[class_index]
            image_arr = np.asarray(image_pil, dtype=np.float32)

            gradcam_map = gradcam._heatmap_for_array(img_array, class_index).numpy()
            gradcam_map = np.asarray(
                Image.fromarray(np.uint8(255 * normalize_map(gradcam_map))).resize(
                    (IMG_SIZE, IMG_SIZE),
                    resample=Image.Resampling.BILINEAR,
                ),
                dtype=np.float32,
            )
            gradcam_map = normalize_map(gradcam_map)

            ig_map = integrated_gradients_heatmap(
                model,
                img_array,
                class_index,
                n_steps=args.ig_steps,
            )

            for method, heatmap in [
                ("Grad-CAM", gradcam_map),
                ("Integrated Gradients", ig_map),
            ]:
                deletion_auc, insertion_auc = perturbation_auc(
                    model,
                    image_arr,
                    heatmap,
                    class_index,
                    steps=args.steps,
                )
                rows.append(
                    {
                        "image": str(path),
                        "true_class": path.parent.name,
                        "predicted_class": predicted_class,
                        "method": method,
                        "deletion_auc": deletion_auc,
                        "insertion_auc": insertion_auc,
                    }
                )
        print(f"Processed {index}/{len(paths)}: {path.name}", flush=True)

    report = pd.DataFrame(rows)
    report.to_csv(output_path, index=False)
    summary = (
        report.groupby("method")[["deletion_auc", "insertion_auc"]]
        .agg(["mean", "std", "count"])
        .round(6)
    )
    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    summary.to_csv(summary_path)
    print(json.dumps({"report": str(output_path), "summary": str(summary_path)}, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
