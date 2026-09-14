from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from config import (
    BINARY_CONFUSION_MATRIX_PATH,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    DATASET_DIR,
    HISTORY_PATH,
    MODEL_PATH,
    TEST_DIR,
    TRAIN_DIR,
    TRAINING_CURVES_PATH,
    VAL_DIR,
)
from gradcam import GradCAM
from predict import TextilePredictor
from utils import quality_score, readable_label, severity_from_prediction

st.set_page_config(
    page_title="Sui-Dhaaga Textile Defect AI",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_predictor() -> TextilePredictor:
    return TextilePredictor()


@st.cache_resource
def load_gradcam() -> GradCAM:
    return GradCAM()


@st.cache_data
def dataset_counts() -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split_name, split_dir in {
        "train": TRAIN_DIR,
        "validation": VAL_DIR,
        "test": TEST_DIR,
    }.items():
        split_counts = {}
        if split_dir.exists():
            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                split_counts[class_dir.name] = len(
                    [
                        path
                        for path in class_dir.iterdir()
                        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                    ]
                )
        counts[split_name] = split_counts
    return counts


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

            * { font-family: 'Inter', sans-serif; }

            .stApp {
                background:
                    radial-gradient(circle at 16% 10%, rgba(0, 229, 255, 0.14), transparent 30%),
                    linear-gradient(135deg, #0A1929 0%, #102844 48%, #1A2332 100%);
                color: #E7EBF0;
            }

            header[data-testid="stHeader"],
            div[data-testid="stToolbar"],
            div[data-testid="stDecoration"],
            div[data-testid="stStatusWidget"] {
                display: none;
            }

            header[data-testid="stHeader"] {
                display: block;
                height: 2.75rem;
                background: transparent;
                pointer-events: none;
            }

            header[data-testid="stHeader"] button {
                pointer-events: auto;
            }

            section.main,
            [data-testid="stAppViewContainer"] {
                padding-top: 0;
            }

            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 2.2rem;
                max-width: 1380px;
            }

            [data-testid="stSidebar"] {
                background: rgba(10, 25, 41, 0.96);
                border-right: 1px solid rgba(0, 229, 255, 0.24);
                padding-top: 0;
                z-index: 999;
            }

            [data-testid="stSidebar"] * {
                color: #DDE7F0;
            }

            [data-testid="stSidebarContent"] {
                padding-top: 1.5rem;
            }

            [data-testid="stSidebar"] [role="radiogroup"] label {
                background: rgba(26, 35, 50, 0.58);
                border: 1px solid rgba(0, 229, 255, 0.14);
                border-radius: 8px;
                margin: 0.35rem 0;
                padding: 0.45rem 0.65rem;
            }

            [data-testid="stSidebar"] [role="radiogroup"] label:hover {
                border-color: rgba(0, 229, 255, 0.54);
                background: rgba(0, 229, 255, 0.08);
            }

            div[role="radiogroup"] label,
            div[role="radiogroup"] label p,
            div[role="radiogroup"] label span {
                color: #DDE7F0 !important;
                font-weight: 700;
            }

            h1, h2, h3 {
                color: #F7FBFF;
                letter-spacing: 0;
            }

            .main-header {
                font-size: 3rem;
                line-height: 1.05;
                font-weight: 800;
                margin: 0 0 0.35rem 0;
                background: linear-gradient(135deg, #00E5FF 0%, #8AF3FF 60%, #00B8D4 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
                text-shadow: 0 0 30px rgba(0, 229, 255, 0.36);
            }

            .sub-header {
                color: #B8C7D9;
                font-size: 1.05rem;
                margin: 0 0 1.4rem 0;
                text-align: center;
            }

            .glass-card {
                background: rgba(26, 35, 50, 0.72);
                border: 1px solid rgba(0, 229, 255, 0.25);
                border-radius: 8px;
                padding: 1.15rem;
                box-shadow: 0 16px 44px rgba(0, 0, 0, 0.24);
                backdrop-filter: blur(14px);
                margin-bottom: 1rem;
                transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
            }

            .glass-card.tight { padding: 0.9rem; }

            .glass-card:hover {
                transform: translateY(-2px);
                border-color: rgba(0, 229, 255, 0.46);
                box-shadow: 0 18px 50px rgba(0, 229, 255, 0.1);
            }

            .card-title {
                color: #00E5FF;
                font-size: 0.92rem;
                font-weight: 700;
                text-transform: uppercase;
                margin-bottom: 0.45rem;
            }

            .card-value {
                color: #F7FBFF;
                font-size: 1.55rem;
                font-weight: 800;
                margin: 0;
            }

            .muted { color: #AAB8C7; }
            .cyan { color: #00E5FF; }
            .good { color: #00E57F; }
            .warn { color: #FFA726; }
            .bad { color: #FF5252; }

            div[data-testid="stMetric"] {
                background: rgba(26, 35, 50, 0.76);
                border: 1px solid rgba(0, 229, 255, 0.25);
                border-radius: 8px;
                padding: 0.9rem 1rem;
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.18);
            }

            [data-testid="stMetricLabel"] { color: #B8C7D9; }
            [data-testid="stMetricValue"] { color: #00E5FF; font-weight: 800; }

            div[data-testid="stMetric"] {
                min-height: 6rem;
                overflow: visible;
            }

            div[data-testid="stMetric"] label,
            div[data-testid="stMetric"] [data-testid="stMetricLabel"],
            div[data-testid="stMetric"] [data-testid="stMetricValue"],
            div[data-testid="stMetric"] [data-testid="stMetricValue"] div {
                white-space: normal !important;
                overflow: visible !important;
                text-overflow: clip !important;
                word-break: break-word;
                line-height: 1.15;
            }

            div[data-testid="stMetric"] [data-testid="stMetricValue"] {
                font-size: clamp(1.1rem, 1.8vw, 1.75rem);
            }

            .stButton > button {
                background: linear-gradient(135deg, #00E5FF 0%, #00B8D4 100%);
                color: #071521;
                border: 0;
                border-radius: 8px;
                font-weight: 800;
                box-shadow: 0 10px 26px rgba(0, 229, 255, 0.22);
            }

            .stButton > button:hover {
                background: linear-gradient(135deg, #8AF3FF 0%, #00E5FF 100%);
                color: #071521;
                border: 0;
            }

            [data-testid="stFileUploader"] {
                background: rgba(26, 35, 50, 0.62);
                border: 1px dashed rgba(0, 229, 255, 0.42);
                border-radius: 8px;
                padding: 0.8rem;
            }

            [data-testid="stDataFrame"] {
                border: 1px solid rgba(0, 229, 255, 0.18);
                border-radius: 8px;
                overflow: hidden;
            }

            .stTabs [data-baseweb="tab-list"] {
                background: rgba(26, 35, 50, 0.55);
                border-radius: 8px;
                padding: 0.35rem;
            }

            .stTabs [data-baseweb="tab"] {
                color: #B8C7D9;
                border-radius: 8px;
            }

            .stTabs [aria-selected="true"] {
                background: rgba(0, 229, 255, 0.16);
                color: #00E5FF;
            }

            .main-nav-wrap {
                background: rgba(10, 25, 41, 0.76);
                border: 1px solid rgba(0, 229, 255, 0.24);
                border-radius: 8px;
                padding: 0.75rem 0.9rem 0.35rem 0.9rem;
                margin-bottom: 1.2rem;
                box-shadow: 0 12px 32px rgba(0, 0, 0, 0.22);
            }

            .main-nav-wrap .nav-title {
                color: #00E5FF;
                font-size: 0.78rem;
                font-weight: 800;
                text-transform: uppercase;
                margin-bottom: 0.3rem;
            }

            .side-panel {
                background: rgba(10, 25, 41, 0.88);
                border: 1px solid rgba(0, 229, 255, 0.28);
                border-radius: 8px;
                padding: 1.1rem;
                box-shadow: 0 16px 44px rgba(0, 0, 0, 0.28);
                margin-bottom: 0.85rem;
            }

            .brand-title {
                color: #00E5FF;
                font-size: 1.55rem;
                line-height: 1;
                font-weight: 800;
                margin-bottom: 0.35rem;
                white-space: nowrap;
            }

            .brand-subtitle {
                color: #AAB8C7;
                font-size: 0.9rem;
                margin-bottom: 1.2rem;
            }

            .side-panel div[role="radiogroup"] {
                display: flex;
                flex-direction: column;
                gap: 0.35rem;
            }

            div[role="radiogroup"] label {
                background: rgba(26, 35, 50, 0.74);
                border: 1px solid rgba(0, 229, 255, 0.18);
                border-radius: 8px;
                padding: 0.45rem 0.55rem;
                min-height: 2.35rem;
            }

            div[role="radiogroup"] label:hover {
                border-color: rgba(0, 229, 255, 0.6);
                background: rgba(0, 229, 255, 0.1);
            }

            img {
                border-radius: 8px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<h1 class="main-header">{title}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">{subtitle}</p>', unsafe_allow_html=True)


def card(title: str, value: str, tone: str = "cyan", note: str | None = None) -> None:
    note_html = f'<div class="muted" style="margin-top:.25rem;">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="glass-card tight">
            <div class="card-title">{title}</div>
            <p class="card-value {tone}">{value}</p>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_table(probabilities: dict[str, float]) -> None:
    frame = pd.DataFrame(
        {
            "Defect Class": [readable_label(name) for name in probabilities],
            "Probability": [value * 100 for value in probabilities.values()],
        }
    ).sort_values("Probability", ascending=False)
    st.dataframe(
        frame.style.format({"Probability": "{:.2f}%"}),
        hide_index=True,
        use_container_width=True,
    )


def render_last_result() -> dict | None:
    result = st.session_state.get("last_result")
    image = st.session_state.get("last_image")
    if not result or image is None:
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-title">No active inspection</div>
                <div class="muted">Upload and analyze a fabric image first.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return None

    severity, severity_rank = severity_from_prediction(
        result["predicted_class"],
        result["confidence"],
    )
    score = quality_score(result["predicted_class"], result["confidence"])
    col1, col2, col3 = st.columns(3)
    metric_items = [
        (col1, "Detected Class", result["display_class"]),
        (col2, "Confidence", f'{result["confidence"] * 100:.2f}%'),
        (col3, "Quality Score", f"{score}/100"),
    ]
    for target_col, label, value in metric_items:
        with target_col:
            st.markdown(
                f"""
                <div class="glass-card tight" style="min-height: 7rem;">
                    <div class="card-title">{label}</div>
                    <div style="
                        color: #00E5FF;
                        font-weight: 800;
                        font-size: clamp(1rem, 1.45vw, 1.55rem);
                        line-height: 1.18;
                        white-space: normal;
                        overflow-wrap: anywhere;
                        word-break: normal;
                    ">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    if severity_rank == 0:
        st.success(f"Quality decision: {severity}")
    elif severity_rank == 1:
        st.warning(f"Quality decision: {severity}")
    else:
        st.error(f"Quality decision: {severity}")
    return result


def dashboard_page() -> None:
    page_header(
        "Sui-Dhaaga Dashboard",
        "Explainable AI based textile defect detection and quality assessment.",
    )
    counts = dataset_counts()
    train_total = sum(counts["train"].values())
    val_total = sum(counts["validation"].values())
    test_total = sum(counts["test"].values())
    class_count = len(counts["train"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        card("Model Status", "Ready" if MODEL_PATH.exists() else "Not Trained", "good" if MODEL_PATH.exists() else "bad")
    with col2:
        card("Defect Classes", str(class_count))
    with col3:
        card("Architecture", "EfficientNetV2B0", note="Transfer learning")
    with col4:
        card("Input Size", "224 x 224", note="RGB inspection crops")

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-title">System Features</div>
                <p class="muted">Deep learning inspection with confidence scoring, Grad-CAM explainability, defect severity, and research-ready analytics.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        dataset_frame = pd.DataFrame(
            [
                {"Split": split.title(), "Images": sum(class_counts.values())}
                for split, class_counts in counts.items()
            ]
        )
        st.bar_chart(dataset_frame, x="Split", y="Images", color="#00E5FF")

    with right:
        st.markdown('<div class="glass-card"><div class="card-title">Dataset Summary</div>', unsafe_allow_html=True)
        st.metric("Training Images", train_total)
        st.metric("Validation Images", val_total)
        st.metric("Test Images", test_total)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card"><div class="card-title">Supported Defect Categories</div>', unsafe_allow_html=True)
    labels = [readable_label(name) for name in sorted(counts["train"])]
    st.write(", ".join(labels))
    st.markdown("</div>", unsafe_allow_html=True)


def upload_predict_page() -> None:
    page_header("Upload & Predict", "Upload fabric images for real-time defect classification.")
    if not MODEL_PATH.exists():
        st.error("No trained model found. Train the model before inspecting images.")
        return

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.markdown('<div class="glass-card"><div class="card-title">Inspection Input</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a fabric image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        st.caption("Accepted formats: JPG, JPEG, PNG")
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_file is not None:
            image_pil = Image.open(uploaded_file).convert("RGB")
            st.image(image_pil, caption="Uploaded Textile Sample", use_container_width=True)
            if st.button("Analyze Fabric", use_container_width=True):
                predictor = load_predictor()
                result = predictor.predict_pil(image_pil)
                gradcam_path = load_gradcam().generate(image_pil, uploaded_file.name)
                st.session_state["last_image"] = image_pil
                st.session_state["last_result"] = result
                st.session_state["last_gradcam_path"] = str(gradcam_path)
                st.success("Inspection complete.")

    with right:
        st.markdown('<div class="glass-card"><div class="card-title">Prediction Results</div>', unsafe_allow_html=True)
        result = render_last_result()
        if result:
            st.markdown("#### Class Probabilities")
            render_probability_table(result["probabilities"])
        st.markdown("</div>", unsafe_allow_html=True)


def explainable_ai_page() -> None:
    page_header("Explainable AI", "Grad-CAM heatmap visualization for model interpretability.")
    result = render_last_result()
    if not result:
        return

    image = st.session_state["last_image"]
    gradcam_path = st.session_state.get("last_gradcam_path")
    if not gradcam_path or not Path(gradcam_path).exists():
        gradcam_path = str(load_gradcam().generate(image, "inspection"))
        st.session_state["last_gradcam_path"] = gradcam_path

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="glass-card"><div class="card-title">Original Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="glass-card"><div class="card-title">Grad-CAM Overlay</div>', unsafe_allow_html=True)
        st.image(gradcam_path, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card">
            <div class="card-title">Interpretation Guide</div>
            <p class="muted">Warmer regions show where the model placed stronger visual attention while selecting the predicted defect class. Use this view to confirm whether the model is focusing on the actual textile irregularity.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def quality_assessment_page() -> None:
    page_header("Quality Assessment", "Severity analysis and production-oriented recommendations.")
    result = render_last_result()
    if not result:
        return

    predicted_class = result["predicted_class"]
    confidence = result["confidence"]
    severity, severity_rank = severity_from_prediction(predicted_class, confidence)
    descriptions = {
        0: ("Approve", "Fabric can proceed to the next production stage."),
        1: ("Manual Review", "Route the sample for secondary inspection before release."),
        2: ("Isolate Batch", "Hold the sample and inspect nearby material from the same batch."),
        3: ("Reject or Rework", "Reject the sample or trigger a corrective-action workflow."),
    }
    title, action = descriptions.get(severity_rank, descriptions[1])

    col1, col2 = st.columns([1, 1.2], gap="large")
    with col1:
        st.markdown('<div class="glass-card"><div class="card-title">Analyzed Image</div>', unsafe_allow_html=True)
        st.image(st.session_state["last_image"], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-title">Quality Control Report</div>
                <p class="card-value cyan">{readable_label(predicted_class)}</p>
                <p class="muted">Severity: <span class="cyan">{severity}</span></p>
                <p class="muted">Confidence: <span class="cyan">{confidence * 100:.2f}%</span></p>
                <hr style="border-color: rgba(0,229,255,.18);">
                <div class="card-title">Recommended Action</div>
                <p class="card-value">{title}</p>
                <p class="muted">{action}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def analytics_page() -> None:
    page_header("Analytics", "Training curves, confusion matrix, and per-class model metrics.")
    top = st.columns(3)
    if HISTORY_PATH.exists():
        history = pd.read_csv(HISTORY_PATH)
        best_val = history["val_accuracy"].max() if "val_accuracy" in history else 0
        last_train = history["accuracy"].iloc[-1] if "accuracy" in history else 0
        last_top3 = history["top_3_accuracy"].iloc[-1] if "top_3_accuracy" in history else 0
        top[0].metric("Best Validation Accuracy", f"{best_val * 100:.2f}%")
        top[1].metric("Final Training Accuracy", f"{last_train * 100:.2f}%")
        top[2].metric("Final Top-3 Accuracy", f"{last_top3 * 100:.2f}%")
    else:
        st.info("Training history appears here after training.")

    graph_a, graph_b = st.columns(2, gap="large")
    with graph_a:
        st.markdown('<div class="glass-card"><div class="card-title">Training Curves</div>', unsafe_allow_html=True)
        if TRAINING_CURVES_PATH.exists():
            st.image(str(TRAINING_CURVES_PATH), use_container_width=True)
        else:
            st.info("Training curves not found.")
        st.markdown("</div>", unsafe_allow_html=True)
    with graph_b:
        st.markdown('<div class="glass-card"><div class="card-title">Confusion Matrix</div>', unsafe_allow_html=True)
        if CONFUSION_MATRIX_PATH.exists():
            st.image(str(CONFUSION_MATRIX_PATH), use_container_width=True)
        else:
            st.info("Confusion matrix not found.")
        st.markdown("</div>", unsafe_allow_html=True)

    if BINARY_CONFUSION_MATRIX_PATH.exists():
        st.markdown('<div class="glass-card"><div class="card-title">Clean vs Defect Matrix</div>', unsafe_allow_html=True)
        st.image(str(BINARY_CONFUSION_MATRIX_PATH), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card"><div class="card-title">Classification Report</div>', unsafe_allow_html=True)
    if CLASSIFICATION_REPORT_PATH.exists():
        report_df = pd.read_csv(CLASSIFICATION_REPORT_PATH)
        st.dataframe(report_df, use_container_width=True)
    else:
        st.info("Classification report not found.")
    st.markdown("</div>", unsafe_allow_html=True)


def train_model_page() -> None:
    page_header("Train Model", "Training status for the real textile dataset.")
    counts = dataset_counts()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset", "Ready" if DATASET_DIR.exists() else "Missing")
    col2.metric("Train", sum(counts["train"].values()))
    col3.metric("Validation", sum(counts["validation"].values()))
    col4.metric("Test", sum(counts["test"].values()))

    st.markdown(
        """
        <div class="glass-card">
            <div class="card-title">Training Command</div>
            <p class="muted">Use the project environment below when retraining on Windows.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(r"C:\tdxai_env\Scripts\python.exe train_model.py", language="powershell")
    st.code(r"C:\tdxai_env\Scripts\python.exe evaluate_model.py", language="powershell")

    if MODEL_PATH.exists():
        st.success(f"Current trained model: {MODEL_PATH}")
    else:
        st.warning("No trained model found.")


inject_styles()

NAV_OPTIONS = [
    "🏠 Dashboard",
    "📤 Upload & Predict",
    "🔍 Explainable AI",
    "✅ Quality Assessment",
    "📊 Analytics",
    "🎓 Train Model",
]

PAGE_KEYS = {
    "🏠 Dashboard": "Dashboard",
    "📤 Upload & Predict": "Upload & Predict",
    "🔍 Explainable AI": "Explainable AI",
    "✅ Quality Assessment": "Quality Assessment",
    "📊 Analytics": "Analytics",
    "🎓 Train Model": "Train Model",
}

nav_col, content_col = st.columns([0.24, 0.76], gap="large")

with nav_col:
    st.markdown(
        """
        <div class="side-panel">
            <div class="brand-title">🧵 Sui-Dhaaga</div>
            <div class="brand-subtitle">AI Textile Quality Console</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        NAV_OPTIONS,
        label_visibility="collapsed",
        key="panel_nav",
    )
    st.markdown("---")
    if MODEL_PATH.exists():
        predictor = load_predictor()
        st.caption("Inspection model")
        st.write(predictor.metadata["model_name"])
        st.caption("Classes")
        st.write(f"{len(predictor.class_names)} textile categories")
    else:
        st.warning("Model not trained.")

page_key = PAGE_KEYS[page]

with content_col:
    if page_key == "Dashboard":
        dashboard_page()
    elif page_key == "Upload & Predict":
        upload_predict_page()
    elif page_key == "Explainable AI":
        explainable_ai_page()
    elif page_key == "Quality Assessment":
        quality_assessment_page()
    elif page_key == "Analytics":
        analytics_page()
    else:
        train_model_page()
