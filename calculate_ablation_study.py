import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

from config import (
    BATCH_SIZE,
    IMG_SIZE,
    MODEL_PATH,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
)
from train_model import RandomGaussianNoise, RandomSoftBlur, make_class_weights


OUTPUT_PATH = Path("outputs") / "reports" / "ablation_study.csv"


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
        shuffle=False,
        class_names=class_names,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
        class_names=class_names,
    )
    autotune = tf.data.AUTOTUNE
    return (
        train_ds.prefetch(autotune),
        val_ds.prefetch(autotune),
        test_ds.prefetch(autotune),
        class_names,
    )


def label_counts(dataset, class_names):
    counts = Counter()
    for _, labels in dataset.unbatch():
        counts[int(tf.argmax(labels).numpy())] += 1
    return {i: counts.get(i, 0) for i in range(len(class_names))}


def build_model(num_classes: int, use_augmentation: bool, label_smoothing: float):
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    if use_augmentation:
        x = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="reflect"),
                tf.keras.layers.RandomRotation(0.04, fill_mode="reflect"),
                tf.keras.layers.RandomZoom(
                    height_factor=(-0.12, 0.10),
                    width_factor=(-0.12, 0.10),
                    fill_mode="reflect",
                ),
                tf.keras.layers.RandomContrast(0.22),
                tf.keras.layers.RandomBrightness(0.16, value_range=(0.0, 255.0)),
                RandomGaussianNoise(stddev=4.0, probability=0.35),
                RandomSoftBlur(probability=0.18),
            ],
            name="textile_safe_augmentation",
        )(x)

    backbone = tf.keras.applications.EfficientNetV2B0(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_preprocessing=True,
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    gap = tf.keras.layers.GlobalAveragePooling2D()(x)
    gmp = tf.keras.layers.GlobalMaxPooling2D()(x)
    x = tf.keras.layers.Concatenate()([gap, gmp])
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(
        512,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(
        256,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="ablation_efficientnetv2b0")
    compile_model(model, label_smoothing=label_smoothing, lr=1e-3)
    return model, backbone


def compile_model(model, label_smoothing: float, lr: float):
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )


def evaluate(model, test_ds, class_names):
    probabilities = model.predict(test_ds, verbose=0)
    y_pred = probabilities.argmax(axis=1)
    y_true = np.concatenate([labels.numpy().argmax(axis=1) for _, labels in test_ds])
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }


def existing_full_row():
    report_path = Path("outputs") / "reports" / "classification_report.csv"
    if not report_path.exists() or not Path(MODEL_PATH).exists():
        return None
    report = pd.read_csv(report_path, index_col=0)
    return {
        "configuration": "Full proposed framework (30 warm-up + 24 fine-tune epochs)",
        "training_schedule": "30+24",
        "accuracy": float(report.loc["accuracy", "f1-score"]),
        "macro_f1": float(report.loc["macro avg", "f1-score"]),
        "weighted_f1": float(report.loc["weighted avg", "f1-score"]),
    }


def run(args):
    tf.keras.utils.set_random_seed(SEED)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, test_ds, class_names = build_datasets()
    class_weight = make_class_weights(label_counts(train_ds, class_names), class_names)

    configs = [
        {
            "configuration": "Backbone only",
            "use_augmentation": False,
            "fine_tune": False,
            "label_smoothing": 0.0,
            "class_weight": None,
        },
        {
            "configuration": "+ Textile-safe augmentation",
            "use_augmentation": True,
            "fine_tune": False,
            "label_smoothing": 0.0,
            "class_weight": None,
        },
        {
            "configuration": "+ Fine-tuning",
            "use_augmentation": True,
            "fine_tune": True,
            "label_smoothing": 0.0,
            "class_weight": None,
        },
        {
            "configuration": "+ Label smoothing",
            "use_augmentation": True,
            "fine_tune": True,
            "label_smoothing": 0.05,
            "class_weight": None,
        },
        {
            "configuration": "+ Class imbalance handling",
            "use_augmentation": True,
            "fine_tune": True,
            "label_smoothing": 0.05,
            "class_weight": class_weight,
        },
    ]

    rows = []
    for config in configs:
        print(f"Running: {config['configuration']}", flush=True)
        model, backbone = build_model(
            len(class_names),
            use_augmentation=config["use_augmentation"],
            label_smoothing=config["label_smoothing"],
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.warmup_epochs,
            class_weight=config["class_weight"],
            verbose=2,
        )
        if config["fine_tune"] and args.finetune_epochs > 0:
            backbone.trainable = True
            for layer in backbone.layers[:-args.unfreeze_layers]:
                layer.trainable = False
            compile_model(
                model,
                label_smoothing=config["label_smoothing"],
                lr=1e-5,
            )
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.finetune_epochs,
                class_weight=config["class_weight"],
                verbose=2,
            )
        metrics = evaluate(model, test_ds, class_names)
        row = {
            "configuration": config["configuration"],
            "training_schedule": f"{args.warmup_epochs}+{args.finetune_epochs if config['fine_tune'] else 0}",
            **metrics,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
        tf.keras.backend.clear_session()

    full = existing_full_row()
    if full:
        rows.append(full)
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    print(json.dumps({"output": str(OUTPUT_PATH), "rows": rows}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--unfreeze-layers", type=int, default=40)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
