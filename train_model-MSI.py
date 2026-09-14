import json
import shutil

from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW

from config import (
    BATCH_SIZE,
    CLASS_DISTRIBUTION_PATH,
    FINETUNE_EPOCHS,
    FINETUNE_LEARNING_RATE,
    FINETUNE_UNFREEZE_LAYERS,
    HISTORY_PATH,
    IMG_SIZE,
    LABEL_SMOOTHING,
    MODELS_DIR,
    MODEL_PATH,
    SEED,
    TRAINING_CURVES_PATH,
    TRAIN_DIR,
    VAL_DIR,
    WARMUP_EPOCHS,
    WARMUP_LEARNING_RATE,
    WEIGHT_DECAY,
)
from utils import ensure_project_dirs, save_metadata


class RandomGaussianNoise(layers.Layer):
    def __init__(self, stddev: float = 5.0, probability: float = 0.35, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev
        self.probability = probability

    def call(self, inputs, training=None):
        if not training:
            return inputs
        should_apply = tf.random.uniform(()) < self.probability
        noise = tf.random.normal(tf.shape(inputs), mean=0.0, stddev=self.stddev)
        return tf.cond(
            should_apply,
            lambda: tf.clip_by_value(inputs + noise, 0.0, 255.0),
            lambda: inputs,
        )

    def get_config(self):
        config = super().get_config()
        config.update({"stddev": self.stddev, "probability": self.probability})
        return config


class RandomSoftBlur(layers.Layer):
    def __init__(self, probability: float = 0.22, **kwargs):
        super().__init__(**kwargs)
        self.probability = probability

    def call(self, inputs, training=None):
        if not training:
            return inputs
        should_apply = tf.random.uniform(()) < self.probability
        channels = tf.shape(inputs)[-1]
        kernel = tf.ones([2, 2, channels, 1], dtype=inputs.dtype) / 4.0
        blurred = tf.nn.depthwise_conv2d(
            inputs,
            kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )
        blurred.set_shape(inputs.shape)
        return tf.cond(should_apply, lambda: blurred, lambda: inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"probability": self.probability})
        return config


def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        seed=SEED,
    )
    class_names = train_ds.class_names
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        seed=SEED,
        shuffle=False,
        class_names=class_names,
    )
    autotune = tf.data.AUTOTUNE
    return (
        train_ds.prefetch(autotune),
        val_ds.prefetch(autotune),
        class_names,
    )


def compute_class_distribution(dataset, class_names: list[str]) -> tuple[dict[int, int], pd.DataFrame]:
    counts = Counter()
    for _, labels in dataset.unbatch():
        counts[int(tf.argmax(labels).numpy())] += 1
    total = sum(counts.values())
    rows = []
    for index, name in enumerate(class_names):
        count = counts.get(index, 0)
        rows.append(
            {
                "class_index": index,
                "class_name": name,
                "count": count,
                "fraction": count / total if total else 0,
            }
        )
    return counts, pd.DataFrame(rows)


def make_class_weights(counts: dict[int, int], class_names: list[str]) -> dict[int, float]:
    non_zero = np.array([counts.get(index, 0) for index in range(len(class_names)) if counts.get(index, 0) > 0])
    if len(non_zero) == 0:
        return {}
    median_count = float(np.median(non_zero))
    weights = {}
    for index in range(len(class_names)):
        count = counts.get(index, 0)
        if count == 0:
            continue
        weights[index] = float(np.clip(np.sqrt(median_count / count), 0.35, 4.0))
    return weights


def build_model(num_classes: int) -> tf.keras.Model:
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(0.05, 0.05, fill_mode="reflect"),
            layers.RandomRotation(0.04, fill_mode="reflect"),
            layers.RandomZoom(height_factor=(-0.12, 0.10), width_factor=(-0.12, 0.10), fill_mode="reflect"),
            layers.RandomContrast(0.22),
            layers.RandomBrightness(0.16, value_range=(0.0, 255.0)),
            RandomGaussianNoise(stddev=4.0, probability=0.35),
            RandomSoftBlur(probability=0.18),
        ],
        name="textile_safe_augmentation",
    )

    base_model = EfficientNetV2B0(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_preprocessing=True,
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augmentation(inputs)
    x = base_model(x, training=False)
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    x = layers.Concatenate()([gap, gmp])
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(
        512,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(
        256,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
    )(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="textile_efficientnetv2")
    model.compile(
        optimizer=AdamW(learning_rate=WARMUP_LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
            tf.keras.metrics.AUC(name="auc", multi_label=True),
        ],
    )
    return model


def plot_training_curves(history: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["accuracy"], label="Train")
    axes[0].plot(history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["loss"], label="Train")
    axes[1].plot(history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(TRAINING_CURVES_PATH, dpi=180)
    plt.close(fig)


def train() -> None:
    ensure_project_dirs()
    train_ds, val_ds, class_names = build_datasets()
    model = build_model(len(class_names))
    counts, distribution = compute_class_distribution(train_ds, class_names)
    distribution.to_csv(CLASS_DISTRIBUTION_PATH, index=False)
    class_weight = make_class_weights(counts, class_names)
    save_metadata(class_names, "EfficientNetV2B0", class_weight)

    def training_callbacks(include_warmup_scheduler: bool) -> list:
        callbacks = [
            ModelCheckpoint(
                MODEL_PATH,
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
            ),
            EarlyStopping(
                monitor="val_accuracy",
                patience=12,
                restore_best_weights=True,
                mode="max",
            ),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7),
            CSVLogger(str(HISTORY_PATH.with_name("training_log.csv")), append=True),
        ]
        if include_warmup_scheduler:
            callbacks.insert(
                -1,
                tf.keras.callbacks.LearningRateScheduler(
                    lambda epoch, lr: min(
                        WARMUP_LEARNING_RATE,
                        WARMUP_LEARNING_RATE * (epoch + 1) / 3,
                    )
                    if epoch < 3
                    else lr
                ),
            )
        return callbacks

    warmup = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=WARMUP_EPOCHS,
        callbacks=training_callbacks(include_warmup_scheduler=True),
        class_weight=class_weight,
    )

    base_model = next(
        layer for layer in model.layers if layer.name.startswith("efficientnetv2")
    )
    base_model.trainable = True
    for layer in base_model.layers[:-FINETUNE_UNFREEZE_LAYERS]:
        layer.trainable = False
    model.compile(
        optimizer=AdamW(learning_rate=FINETUNE_LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
            tf.keras.metrics.AUC(name="auc", multi_label=True),
        ],
    )
    finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINETUNE_EPOCHS,
        callbacks=training_callbacks(include_warmup_scheduler=False),
        class_weight=class_weight,
    )

    history = {
        key: warmup.history.get(key, []) + finetune.history.get(key, [])
        for key in set(warmup.history) | set(finetune.history)
    }
    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)
    plot_training_curves(history)
    model.save(MODEL_PATH)
    versioned_model_path = MODELS_DIR / (
        f"textile_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
    )
    shutil.copy2(MODEL_PATH, versioned_model_path)
    print(json.dumps({"model": str(MODEL_PATH), "classes": class_names}, indent=2))


if __name__ == "__main__":
    train()
