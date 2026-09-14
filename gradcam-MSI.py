from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from config import GRADCAM_DIR, IMG_SIZE, MODEL_PATH
from train_model import RandomGaussianNoise, RandomSoftBlur
from utils import ensure_project_dirs, pil_to_model_array, safe_stem


class GradCAM:
    def __init__(self) -> None:
        self.model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "RandomGaussianNoise": RandomGaussianNoise,
                "RandomSoftBlur": RandomSoftBlur,
            },
        )
        backbone = next(
            layer
            for layer in self.model.layers
            if layer.name.startswith(("mobilenetv2", "efficientnetv2"))
        )
        self.last_conv_layer = next(
            layer for layer in reversed(backbone.layers) if len(layer.output.shape) == 4
        )
        self.feature_model = tf.keras.models.Model(
            backbone.inputs,
            [self.last_conv_layer.output, backbone.output],
        )
        backbone_index = self.model.layers.index(backbone)
        self.preprocessing_layers = self.model.layers[1:backbone_index]
        self.classifier_layers = self.model.layers[backbone_index + 1 :]

    def _heatmap_for_array(self, img_array, class_index=None):
        with tf.GradientTape() as tape:
            x = img_array
            for layer in self.preprocessing_layers:
                x = layer(x, training=False)
            conv_outputs, x = self.feature_model(x, training=False)
            for layer in self.classifier_layers:
                x = layer(x, training=False)
            predictions = x
            class_idx = tf.argmax(predictions[0]) if class_index is None else class_index
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        max_value = tf.reduce_max(heatmap)
        return heatmap / tf.maximum(max_value, tf.keras.backend.epsilon())

    def generate(
        self,
        image_pil: Image.Image,
        output_name: str = "inspection",
        smooth_samples: int = 4,
        noise_std: float = 6.0,
    ) -> Path:
        ensure_project_dirs()
        img_array = pil_to_model_array(image_pil, IMG_SIZE)
        predictions = self.model(img_array, training=False)[0]
        class_index = int(tf.argmax(predictions).numpy())
        confidence = float(predictions[class_index].numpy())
        heatmaps = [self._heatmap_for_array(img_array, class_index)]
        for _ in range(max(0, smooth_samples - 1)):
            noisy = tf.clip_by_value(
                img_array + tf.random.normal(tf.shape(img_array), stddev=noise_std),
                0.0,
                255.0,
            )
            heatmaps.append(self._heatmap_for_array(noisy, class_index))
        heatmap = tf.reduce_mean(tf.stack(heatmaps), axis=0)
        heatmap = heatmap / tf.maximum(tf.reduce_max(heatmap), tf.keras.backend.epsilon())

        rgb = np.asarray(image_pil.convert("RGB"))
        resized = cv2.resize(heatmap.numpy(), (rgb.shape[1], rgb.shape[0]))
        resized = cv2.GaussianBlur(resized, (0, 0), sigmaX=2.0)
        colored = cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_TURBO)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        heat_weight = 0.20 + 0.25 * confidence
        overlay = cv2.addWeighted(rgb, 1.0 - heat_weight, colored, heat_weight, 0)

        output_path = GRADCAM_DIR / f"{safe_stem(output_name)}_gradcam.png"
        Image.fromarray(overlay).save(output_path)
        return output_path

    def integrated_gradients(
        self,
        image_pil: Image.Image,
        output_name: str = "inspection",
        n_steps: int = 50,
    ) -> Path:
        ensure_project_dirs()
        img = tf.convert_to_tensor(pil_to_model_array(image_pil, IMG_SIZE))
        baseline = tf.zeros_like(img)
        alphas = tf.linspace(0.0, 1.0, n_steps)
        interpolated = baseline + alphas[:, None, None, None] * (img - baseline)

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            preds = self.model(interpolated, training=False)
            class_index = tf.argmax(preds[0])
            outputs = preds[:, class_index]

        grads = tape.gradient(outputs, interpolated)
        avg_grads = tf.reduce_mean(grads, axis=0)
        attributions = (img[0] - baseline[0]) * avg_grads
        heatmap = tf.reduce_sum(tf.abs(attributions), axis=-1).numpy()
        heatmap = heatmap / max(float(np.max(heatmap)), 1e-8)

        output_path = GRADCAM_DIR / f"{safe_stem(output_name)}_integrated_gradients.png"
        Image.fromarray(np.uint8(255 * heatmap)).save(output_path)
        return output_path


def pointing_game_hit(heatmap: np.ndarray, annotation_bbox: tuple[int, int, int, int]) -> bool:
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    x1, y1, x2, y2 = annotation_bbox
    return x1 <= peak_x <= x2 and y1 <= peak_y <= y2
