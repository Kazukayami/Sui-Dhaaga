import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

from config import (
    BATCH_SIZE,
    BINARY_CONFUSION_MATRIX_PATH,
    BINARY_REPORT_PATH,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_PAIRS_PATH,
    CONFUSION_MATRIX_PATH,
    IMG_SIZE,
    MODEL_PATH,
    ROC_AUC_REPORT_PATH,
    TEST_DIR,
)
from train_model import RandomGaussianNoise, RandomSoftBlur
from utils import ensure_project_dirs, load_metadata, readable_label


def evaluate() -> None:
    ensure_project_dirs()
    metadata = load_metadata()
    class_names = metadata["class_names"]
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
        class_names=class_names,
    )
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "RandomGaussianNoise": RandomGaussianNoise,
            "RandomSoftBlur": RandomSoftBlur,
        },
    )
    probabilities = model.predict(test_ds)
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
    pd.DataFrame(report).transpose().to_csv(CLASSIFICATION_REPORT_PATH)

    y_true_one_hot = label_binarize(y_true, classes=list(range(len(class_names))))
    auc_rows = []
    for index, class_name in enumerate(class_names):
        try:
            auc_value = roc_auc_score(y_true_one_hot[:, index], probabilities[:, index])
        except ValueError:
            auc_value = np.nan
        auc_rows.append({"class_name": class_name, "roc_auc_ovr": auc_value})
    try:
        macro_auc = roc_auc_score(
            y_true_one_hot,
            probabilities,
            average="macro",
            multi_class="ovr",
        )
    except ValueError:
        macro_auc = np.nan
    auc_rows.append({"class_name": "macro_avg", "roc_auc_ovr": macro_auc})
    pd.DataFrame(auc_rows).to_csv(ROC_AUC_REPORT_PATH, index=False)

    binary_true = (y_true != class_names.index("clean")).astype(int)
    binary_pred = (y_pred != class_names.index("clean")).astype(int)
    binary_report = classification_report(
        binary_true,
        binary_pred,
        target_names=["clean", "defect"],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(binary_report).transpose().to_csv(BINARY_REPORT_PATH)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    confusion_rows = []
    for true_index, true_name in enumerate(class_names):
        for pred_index, pred_name in enumerate(class_names):
            count = int(cm[true_index, pred_index])
            if true_index != pred_index and count > 0:
                confusion_rows.append(
                    {
                        "true_class": true_name,
                        "predicted_class": pred_name,
                        "count": count,
                    }
                )
    pd.DataFrame(confusion_rows).sort_values(
        "count", ascending=False
    ).to_csv(CONFUSION_PAIRS_PATH, index=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="crest",
        xticklabels=[readable_label(name) for name in class_names],
        yticklabels=[readable_label(name) for name in class_names],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=180)
    plt.close(fig)

    binary_cm = confusion_matrix(binary_true, binary_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        binary_cm,
        annot=True,
        fmt="d",
        cmap="flare",
        xticklabels=["Clean", "Defect"],
        yticklabels=["Clean", "Defect"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Defect Detection Confusion Matrix")
    fig.tight_layout()
    fig.savefig(BINARY_CONFUSION_MATRIX_PATH, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    evaluate()
