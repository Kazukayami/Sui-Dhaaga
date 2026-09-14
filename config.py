from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
GRADCAM_DIR = OUTPUTS_DIR / "gradcam"

MODEL_PATH = MODELS_DIR / "textile_model.keras"
METADATA_PATH = MODELS_DIR / "metadata.json"
HISTORY_PATH = REPORTS_DIR / "training_history.csv"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.csv"
BINARY_REPORT_PATH = REPORTS_DIR / "binary_classification_report.csv"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
BINARY_CONFUSION_MATRIX_PATH = REPORTS_DIR / "binary_confusion_matrix.png"
TRAINING_CURVES_PATH = REPORTS_DIR / "training_curves.png"
CLASS_DISTRIBUTION_PATH = REPORTS_DIR / "class_distribution.csv"
DATASET_AUDIT_PATH = REPORTS_DIR / "dataset_audit.json"
CONFUSION_PAIRS_PATH = REPORTS_DIR / "confusion_pairs.csv"
ROC_AUC_REPORT_PATH = REPORTS_DIR / "roc_auc_report.csv"

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
LAST_CONV_LAYER = "Conv_1"

WARMUP_EPOCHS = 30
FINETUNE_EPOCHS = 24
FINETUNE_UNFREEZE_LAYERS = 80
WARMUP_LEARNING_RATE = 1e-3
FINETUNE_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
UNKNOWN_CONFIDENCE_THRESHOLD = 0.58
UNKNOWN_MARGIN_THRESHOLD = 0.12
CLEAN_MIN_CONFIDENCE = 0.72

DEFAULT_CLASS_DESCRIPTIONS = {
    "broken_end": "Broken warp end or interrupted longitudinal yarn",
    "broken_pick": "Broken weft pick or interrupted crosswise yarn",
    "broken_yarn": "Discontinuity in yarn structure",
    "clean": "No visible defect detected",
    "contamination": "Foreign matter, stain, dirt, or surface contamination",
    "crease": "Fold, wrinkle, or crease visible on the fabric",
    "cut_selvage": "Cut, torn, or damaged fabric edge",
    "fuzzyball": "Loose fiber cluster or surface pilling",
    "knots": "Visible knot in the yarn or fabric surface",
    "nep": "Small entangled fiber knot on the textile surface",
    "warp_ball": "Accumulated warp fiber ball on the fabric",
    "weft_crack": "Gap or crack along the weft direction",
    "weft_curling": "Curled or distorted weft yarn",
    "review": "Uncertain or unsupported fabric condition requiring manual review",
}

CLASS_ALIASES = {
    "good": "clean",
    "normal": "clean",
    "non_defective": "clean",
    "defect_free": "clean",
    "broken end": "broken_end",
    "broken-end": "broken_end",
    "broken pick": "broken_pick",
    "broken-pick": "broken_pick",
    "broken yarn": "broken_yarn",
    "broken-yarn": "broken_yarn",
    "cut selvage": "cut_selvage",
    "cut-selvage": "cut_selvage",
    "warp ball": "warp_ball",
    "warp-ball": "warp_ball",
    "weft crack": "weft_crack",
    "weft-crack": "weft_crack",
    "weft curling": "weft_curling",
    "weft-curling": "weft_curling",
    "stain": "contamination",
    "oil_stain": "contamination",
    "hole": "cut_selvage",
    "tear": "cut_selvage",
    "pilling": "fuzzyball",
    "fuzzy_ball": "fuzzyball",
}
