import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

from config import BATCH_SIZE, IMG_SIZE, SEED, TEST_DIR, TRAIN_DIR, VAL_DIR
from train_model import RandomGaussianNoise, RandomSoftBlur, make_class_weights


EXPERIMENT_DIR = Path("outputs") / "experiments"


BACKBONES = {
    "MobileNetV2": tf.keras.applications.MobileNetV2,
    "ResNet50": tf.keras.applications.ResNet50,
    "EfficientNetV2B0": tf.keras.applications.EfficientNetV2B0,
}

PREPROCESSORS = {
    "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "ResNet50": tf.keras.applications.resnet50.preprocess_input,
}


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


def count_labels(dataset, class_names):
    counts = {index: 0 for index in range(len(class_names))}
    for _, labels in dataset.unbatch():
        counts[int(tf.argmax(labels).numpy())] += 1
    return counts


def focal_loss(gamma=2.0, alpha=None):
    alpha_tensor = None if alpha is None else tf.constant(alpha, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        modulator = tf.pow(1.0 - y_pred, gamma)
        weighted = modulator * cross_entropy
        if alpha_tensor is not None:
            weighted = weighted * alpha_tensor
        return tf.reduce_sum(weighted, axis=-1)

    return loss


def build_model(backbone_name, num_classes):
    augmentation = tf.keras.Sequential(
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
    )
    kwargs = {
        "weights": "imagenet",
        "include_top": False,
        "input_shape": (IMG_SIZE, IMG_SIZE, 3),
    }
    if backbone_name == "EfficientNetV2B0":
        kwargs["include_preprocessing"] = True
    else:
        preprocess = PREPROCESSORS[backbone_name]

    base = BACKBONES[backbone_name](**kwargs)
    base._name = f"{backbone_name.lower()}_backbone"
    base.trainable = False

    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augmentation(inputs)
    if backbone_name != "EfficientNetV2B0":
        x = tf.keras.layers.Lambda(preprocess, name=f"{backbone_name.lower()}_preprocess")(x)
    x = base(x, training=False)
    gap = tf.keras.layers.GlobalAveragePooling2D()(x)
    gmp = tf.keras.layers.GlobalMaxPooling2D()(x)
    x = tf.keras.layers.Concatenate()([gap, gmp])
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(512, activation="gelu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="gelu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name=f"textile_{backbone_name.lower()}")


def get_backbone(model, backbone_name):
    expected = f"{backbone_name.lower()}_backbone"
    normalized = backbone_name.lower().replace("_", "")
    for layer in model.layers:
        if layer.name == expected:
            return layer
    for layer in model.layers:
        layer_name = layer.name.lower().replace("_", "")
        if hasattr(layer, "layers") and normalized in layer_name:
            return layer
    raise ValueError(f"Backbone layer for {backbone_name} was not found.")


def compile_model(model, loss_name, class_weight_values=None, lr=1e-3):
    if loss_name == "focal":
        loss = focal_loss(alpha=class_weight_values)
    else:
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
        loss=loss,
        metrics=["accuracy"],
    )


def evaluate_model(model, test_ds, class_names):
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
    clean_index = class_names.index("clean")
    binary_true = (y_true != clean_index).astype(int)
    binary_pred = (y_pred != clean_index).astype(int)
    binary_report = classification_report(
        binary_true,
        binary_pred,
        target_names=["clean", "defect"],
        output_dict=True,
        zero_division=0,
    )
    return report, binary_report


def benchmark_latency(model, test_ds, batches=10):
    sample_batches = []
    for images, _ in test_ds.take(batches):
        sample_batches.append(images)
    for images in sample_batches[:2]:
        model.predict(images, verbose=0)
    total_images = sum(int(images.shape[0]) for images in sample_batches)
    start = time.perf_counter()
    for images in sample_batches:
        model.predict(images, verbose=0)
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / total_images


def run(args):
    tf.keras.utils.set_random_seed(SEED)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, test_ds, class_names = build_datasets()
    counts = count_labels(train_ds, class_names)
    class_weights = make_class_weights(counts, class_names)
    class_weight_values = [class_weights.get(i, 1.0) for i in range(len(class_names))]
    rows = []

    for backbone_name in args.models:
        model = build_model(backbone_name, len(class_names))
        compile_model(model, args.loss, class_weight_values)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.warmup_epochs,
            class_weight=class_weights if args.loss == "weighted_ce" else None,
            verbose=2,
        )
        warmup_path = EXPERIMENT_DIR / f"{backbone_name}_{args.loss}_warmup.keras"
        model.save(warmup_path)
        if args.finetune_epochs:
            base = get_backbone(model, backbone_name)
            base.trainable = True
            for layer in base.layers[:-args.unfreeze_layers]:
                layer.trainable = False
            compile_model(model, args.loss, class_weight_values, lr=1e-5)
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.finetune_epochs,
                class_weight=class_weights if args.loss == "weighted_ce" else None,
                verbose=2,
            )

        report, binary_report = evaluate_model(model, test_ds, class_names)
        latency_ms = benchmark_latency(model, test_ds)
        row = {
            "model": backbone_name,
            "loss": args.loss,
            "warmup_epochs": args.warmup_epochs,
            "finetune_epochs": args.finetune_epochs,
            "multiclass_accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "binary_accuracy": binary_report["accuracy"],
            "parameters": model.count_params(),
            "inference_ms_per_image": latency_ms,
        }
        rows.append(row)
        pd.DataFrame([row]).to_csv(
            EXPERIMENT_DIR / f"{backbone_name}_{args.loss}_summary.csv",
            index=False,
        )
        model.save(EXPERIMENT_DIR / f"{backbone_name}_{args.loss}.keras")

    output = EXPERIMENT_DIR / f"summary_{args.loss}_{int(time.time())}.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(json.dumps({"output": str(output), "rows": rows}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["MobileNetV2", "ResNet50"])
    parser.add_argument("--loss", choices=["weighted_ce", "focal"], default="weighted_ce")
    parser.add_argument("--warmup-epochs", type=int, default=30)
    parser.add_argument("--finetune-epochs", type=int, default=24)
    parser.add_argument("--unfreeze-layers", type=int, default=40)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
