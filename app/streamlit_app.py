"""
streamlit_app.py
────────────────
Main Streamlit UI for RBC Infection Detection system.
Matches GUI workflow shown in document Chapter 8 (Output Screens),
rebuilt as a modern multi-page web application.

Pages:
    🏠  Home         — project overview, dataset stats
    📂  Data         — upload dataset, preprocessing, splitting
    🧠  Train        — train Custom CNN or MobileNetV2, show metrics
    📊  Evaluate     — confusion matrix, ROC curve, training graphs
    🔬  Predict      — upload image → real-time prediction result card

Run:
    streamlit run app/streamlit_app.py

Author  : C. Vaishnavi (22R91A7325)
Project : AI-Enhanced Microscopic Image Classification for RBC Infection
College : TKREC, Hyderabad
"""

import os
import sys
import io
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to path so utils imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ─── Page Config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title = "RBC Infection Detection",
    page_icon  = "🔬",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29, #302b63, #24243e);
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stRadio > label { font-size: 15px; }

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 18px 24px;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-value { font-size: 2rem; font-weight: 700; color: #e94560; }
.metric-label { font-size: 0.85rem; color: #a0a0c0; margin-top: 4px; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 28px;
    border: 1px solid #302b63;
}
.hero h1 { color: #e94560; font-size: 2.2rem; font-weight: 700; margin: 0; }
.hero p  { color: #c0c0d8; font-size: 1.05rem; margin-top: 10px; }

/* ── Step badges ── */
.step-badge {
    display: inline-block;
    background: #e94560;
    color: white;
    border-radius: 50%;
    width: 28px; height: 28px;
    line-height: 28px;
    text-align: center;
    font-weight: 700;
    margin-right: 10px;
    font-size: 13px;
}

/* ── Prediction result card ── */
.result-infected {
    background: linear-gradient(135deg, #3d0000, #6b0000);
    border: 2px solid #e94560;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
}
.result-healthy {
    background: linear-gradient(135deg, #003d2e, #006b52);
    border: 2px solid #2a9d8f;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
}
.result-title { font-size: 1.8rem; font-weight: 700; }
.result-conf  { font-size: 1.1rem; margin-top: 8px; color: #e0e0e0; }

/* ── Info box ── */
.info-box {
    background: #16213e;
    border-left: 4px solid #0f3460;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 12px 0;
    color: #c0c0d8;
    font-size: 0.92rem;
}

/* ── Tables ── */
.dataframe { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialiser ────────────────────────────────────────────────
def init_state():
    defaults = {
        "X": None, "y": None, "classes": None, "stats": None,
        "splits": None,
        "cnn_model": None,       "cnn_history": None,    "cnn_metrics": None,
        "mob_model": None,       "mob_history": None,    "mob_metrics": None,
        "dtc_model": None,       "dtc_metrics": None,
        "dataset_path": None,
        "active_page": "🏠  Home",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Sidebar Navigation ───────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🔬 RBC Detector")
        st.markdown("*AI Pathologist Mode*")
        st.markdown("---")

        pages = [
            "🏠  Home",
            "📂  Dataset",
            "🧠  Train Model",
            "📊  Evaluate",
            "🔬  Predict",
        ]
        page = st.radio("Navigate", pages, index=pages.index(st.session_state.active_page))
        st.session_state.active_page = page

        st.markdown("---")
        # Status indicators
        st.markdown("**Pipeline Status**")
        _status("Dataset Loaded",  st.session_state.X is not None)
        _status("Data Split",      st.session_state.splits is not None)
        _status("CNN Trained",     st.session_state.cnn_model is not None)
        _status("MobileNet Trained", st.session_state.mob_model is not None)

    return page


def _status(label: str, ok: bool):
    icon = "✅" if ok else "⬜"
    st.markdown(f"{icon} {label}")


# ─── PAGE: Home ───────────────────────────────────────────────────────────────
def page_home():
    st.markdown("""
    <div class="hero">
        <h1>🔬 AI-Enhanced RBC Infection Detection</h1>
        <p>
            Deep learning system for automated classification of Red Blood Cell infections
            (Malaria / Parasitized vs Uninfected) from microscopic blood smear images.<br>
            Built using <b>Custom CNN</b> and <b>MobileNetV2 Transfer Learning</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric overview cards ──────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">94.73%</div>
            <div class="metric-label">CNN Accuracy (doc)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">95.47%</div>
            <div class="metric-label">Sensitivity</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">93.97%</div>
            <div class="metric-label">Specificity</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">2</div>
            <div class="metric-label">Models Compared</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Workflow steps ─────────────────────────────────────
    st.markdown("### 📋 System Workflow")
    steps = [
        ("Upload Dataset",        "Load Parasitized / Uninfected image folders"),
        ("Image Preprocessing",   "Resize → Normalize → Augment → Split 80/20"),
        ("Model Training",        "Train Custom CNN or MobileNetV2 transfer learning"),
        ("Model Evaluation",      "Accuracy, Precision, Recall, F1, Confusion Matrix, ROC"),
        ("Predict",               "Upload any RBC image → instant infected/healthy result"),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(
            f'<span class="step-badge">{i}</span> <b>{title}</b> — {desc}',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tech stack ─────────────────────────────────────────
    st.markdown("### 🛠 Tech Stack")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        | Component | Technology |
        |-----------|------------|
        | Language  | Python 3.10 |
        | DL Framework | TensorFlow / Keras |
        | Image Processing | OpenCV |
        | ML Baseline | Scikit-learn |
        """)
    with col2:
        st.markdown("""
        | Component | Technology |
        |-----------|------------|
        | Data Analysis | NumPy, Pandas |
        | Visualisation | Matplotlib, Seaborn |
        | UI | Streamlit |
        | Dataset | NIH Malaria Cell Images |
        """)

    # ── Dataset info ───────────────────────────────────────
    st.markdown("### 📦 Dataset")
    st.markdown("""
    <div class="info-box">
    <b>NIH / Kaggle Malaria Cell Images Dataset</b><br>
    • <b>27,558</b> cell images total — 13,779 Parasitized + 13,779 Uninfected<br>
    • Source: <a href="https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria" target="_blank">
      Kaggle — Cell Images for Detecting Malaria</a><br>
    • Images captured via Giemsa-stained thin blood smear microscopy<br>
    • Balanced classes — no resampling required for baseline experiments
    </div>
    """, unsafe_allow_html=True)


# ─── PAGE: Dataset ────────────────────────────────────────────────────────────
def page_dataset():
    st.title("📂 Dataset — Load & Preprocess")

    # ── Step 1: Upload path ────────────────────────────────
    st.markdown('<span class="step-badge">1</span> **Select Dataset Root Folder**',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Expected structure:<br>
    <code>dataset/<br>
    ├── Parasitized/  (infected cell images)<br>
    └── Uninfected/   (healthy cell images)</code>
    </div>
    """, unsafe_allow_html=True)

    dataset_path = st.text_input(
        "Dataset path",
        value=st.session_state.dataset_path or "./dataset",
        placeholder="./dataset"
    )

    img_size_choice = st.selectbox(
        "Image resize target",
        options=["64 × 64  (Custom CNN)", "128 × 128  (MobileNetV2)"],
        index=0
    )
    img_size = 64 if "64" in img_size_choice else 128

    col1, col2 = st.columns([1, 3])
    with col1:
        load_btn = st.button("📥 Load Dataset", use_container_width=True)

    if load_btn:
        if not os.path.exists(dataset_path):
            st.error(f"Path not found: `{dataset_path}`  — Please check and retry.")
            return

        with st.spinner("Loading images… this may take a minute for large datasets."):
            try:
                from utils.data_pipeline import load_dataset, dataset_summary
                progress_bar = st.progress(0)

                def prog_cb(class_name, i, total):
                    progress_bar.progress(min(int(i / max(total, 1) * 100), 100))

                X, y, classes, stats = load_dataset(dataset_path, (img_size, img_size), prog_cb)
                progress_bar.progress(100)

                st.session_state.X          = X
                st.session_state.y          = y
                st.session_state.classes    = classes
                st.session_state.stats      = stats
                st.session_state.dataset_path = dataset_path
                st.session_state.splits     = None   # reset on reload

                st.success(f"✅ Dataset loaded — {stats['loaded']} images across {len(classes)} classes")

                # Class distribution
                st.markdown("#### Class Distribution")
                df_summary = dataset_summary(y, classes)
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.dataframe(df_summary, use_container_width=True)
                with c2:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.bar(df_summary["Class"], df_summary["Count"],
                           color=["#e94560", "#2a9d8f"], edgecolor="white")
                    ax.set_title("Class Distribution", fontweight="bold")
                    ax.set_ylabel("Count")
                    st.pyplot(fig)
                    plt.close(fig)

                # Sample images
                st.markdown("#### Sample Images")
                _show_sample_images(X, y, classes)

            except Exception as e:
                st.error(f"Error loading dataset: {e}")
                return

    # ── Step 2: Split ──────────────────────────────────────
    if st.session_state.X is not None:
        st.markdown("---")
        st.markdown('<span class="step-badge">2</span> **Split Dataset**',
                    unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            test_pct = st.slider("Test %",  10, 30, 20, 5)
        with c2:
            val_pct  = st.slider("Val %",   5,  25, 15, 5)

        col1, _ = st.columns([1, 3])
        with col1:
            split_btn = st.button("✂️ Split Dataset", use_container_width=True)

        if split_btn:
            from utils.data_pipeline import split_dataset
            splits = split_dataset(
                st.session_state.X,
                st.session_state.y,
                test_size = test_pct / 100,
                val_size  = val_pct  / 100
            )
            st.session_state.splits = splits

            st.success("✅ Dataset split complete")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Train Samples", splits["X_train"].shape[0])
            sc2.metric("Val Samples",   splits["X_val"].shape[0])
            sc3.metric("Test Samples",  splits["X_test"].shape[0])


def _show_sample_images(X, y, classes, n=8):
    indices = np.random.choice(len(X), min(n, len(X)), replace=False)
    cols    = st.columns(n)
    for col, idx in zip(cols, indices):
        with col:
            st.image(X[idx], caption=classes[y[idx]], use_container_width=True)


# ─── PAGE: Train ──────────────────────────────────────────────────────────────
def page_train():
    st.title("🧠 Model Training")

    if st.session_state.splits is None:
        st.warning("⚠️  Please load and split the dataset first (📂 Dataset page).")
        return

    splits  = st.session_state.splits
    classes = st.session_state.classes

    # ── Model selection ────────────────────────────────────
    st.markdown("#### Select Model to Train")
    model_choice = st.radio(
        "Model",
        ["Custom CNN", "MobileNetV2 Transfer Learning", "Decision Tree (Baseline)"],
        horizontal=True
    )

    # ── Hyperparameters ────────────────────────────────────
    with st.expander("⚙️ Hyperparameters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            epochs      = st.number_input("Epochs",         min_value=1,  max_value=100, value=10)
        with c2:
            batch_size  = st.number_input("Batch Size",     min_value=8,  max_value=256, value=32)
        with c3:
            lr          = st.number_input("Learning Rate",  min_value=1e-6, max_value=1e-1,
                                          value=0.001, format="%.5f")
        with c4:
            dropout     = st.slider("Dropout Rate", 0.1, 0.7, 0.4, 0.05)

    col1, _ = st.columns([1, 3])
    with col1:
        train_btn = st.button("🚀 Start Training", use_container_width=True)

    if train_btn:
        _run_training(model_choice, splits, classes, epochs, batch_size, lr, dropout)


def _run_training(model_choice, splits, classes, epochs, batch_size, lr, dropout):
    from tensorflow.keras.utils import to_categorical

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val   = splits["X_val"]
    y_val   = splits["y_val"]
    X_test  = splits["X_test"]
    y_test  = splits["y_test"]
    n_cls   = len(classes)

    # ── Decision Tree ──────────────────────────────────────
    if model_choice == "Decision Tree (Baseline)":
        st.info("Training Decision Tree Classifier (baseline)…")
        from utils.models import build_decision_tree
        from utils.evaluation import compute_metrics, plot_confusion_matrix

        dtc = build_decision_tree()
        X_tr_flat = X_train.reshape(len(X_train), -1)
        X_te_flat = X_test.reshape(len(X_test),  -1)

        with st.spinner("Fitting Decision Tree…"):
            dtc.fit(X_tr_flat, y_train)

        y_pred   = dtc.predict(X_te_flat)
        metrics  = compute_metrics(y_test, y_pred, classes)

        st.session_state.dtc_model   = dtc
        st.session_state.dtc_metrics = metrics
        _show_metrics(metrics, "Decision Tree")

        fig_cm = plot_confusion_matrix(metrics["confusion_matrix"], classes, "Decision Tree")
        st.pyplot(fig_cm)
        plt.close(fig_cm)
        return

    # ── Deep Learning models ───────────────────────────────
    y_train_cat = to_categorical(y_train, n_cls)
    y_val_cat   = to_categorical(y_val,   n_cls)
    y_test_cat  = to_categorical(y_test,  n_cls)

    os.makedirs("models", exist_ok=True)

    if model_choice == "Custom CNN":
        from utils.models import build_custom_cnn, get_training_callbacks
        from utils.data_pipeline import build_augmentation_generator

        input_shape = X_train.shape[1:]
        model = build_custom_cnn(input_shape, n_cls, dropout, lr)
        save_path = "models/cnn_model.h5"
        callbacks = get_training_callbacks(save_path)

        aug = build_augmentation_generator()

        st.info(f"Training **Custom CNN** for {epochs} epochs…")
        progress_text = st.empty()
        progress_bar  = st.progress(0)

        history_data = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

        for epoch in range(epochs):
            hist = model.fit(
                aug.flow(X_train, y_train_cat, batch_size=batch_size),
                validation_data = (X_val, y_val_cat),
                epochs=1, verbose=0,
                callbacks=callbacks
            )
            for k in history_data:
                history_data[k].append(hist.history[k][0])

            progress_bar.progress(int((epoch + 1) / epochs * 100))
            acc = hist.history["accuracy"][0] * 100
            val = hist.history["val_accuracy"][0] * 100
            progress_text.markdown(
                f"Epoch **{epoch+1}/{epochs}** — Train Acc: `{acc:.2f}%` | Val Acc: `{val:.2f}%`"
            )

        # Save & evaluate
        model.save(save_path)
        y_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        metrics = compute_metrics_full(y_test, y_pred, y_prob, classes)

        st.session_state.cnn_model   = model
        st.session_state.cnn_history = history_data
        st.session_state.cnn_metrics = metrics

        st.success(f"✅ Custom CNN trained and saved → `{save_path}`")
        _show_metrics(metrics, "Custom CNN")
        _show_training_plots(history_data, "Custom CNN")
        _show_confusion_matrix(metrics, classes, "Custom CNN")

    elif model_choice == "MobileNetV2 Transfer Learning":
        from utils.models import build_mobilenet_v2, unfreeze_mobilenet, get_training_callbacks

        # Resize for MobileNetV2
        X_tr_big = _resize_batch(X_train, 128)
        X_va_big = _resize_batch(X_val,   128)
        X_te_big = _resize_batch(X_test,  128)

        model, base_model = build_mobilenet_v2((128, 128, 3), n_cls, dropout, lr)
        save_path = "models/mobilenet_model.h5"
        callbacks = get_training_callbacks(save_path)

        st.info(f"Training **MobileNetV2** — Phase 1 (frozen base) for {epochs} epochs…")
        progress_bar  = st.progress(0)
        progress_text = st.empty()

        history_data = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

        for epoch in range(epochs):
            hist = model.fit(
                X_tr_big, y_train_cat,
                validation_data = (X_va_big, y_val_cat),
                batch_size=batch_size, epochs=1, verbose=0
            )
            for k in history_data:
                history_data[k].append(hist.history[k][0])

            progress_bar.progress(int((epoch + 1) / epochs * 100))
            acc = hist.history["accuracy"][0] * 100
            val = hist.history["val_accuracy"][0] * 100
            progress_text.markdown(
                f"Epoch **{epoch+1}/{epochs}** — Train Acc: `{acc:.2f}%` | Val Acc: `{val:.2f}%`"
            )

        model.save(save_path)
        y_prob = model.predict(X_te_big, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        metrics = compute_metrics_full(y_test, y_pred, y_prob, classes)

        st.session_state.mob_model   = model
        st.session_state.mob_history = history_data
        st.session_state.mob_metrics = metrics

        st.success(f"✅ MobileNetV2 trained and saved → `{save_path}`")
        _show_metrics(metrics, "MobileNetV2")
        _show_training_plots(history_data, "MobileNetV2")
        _show_confusion_matrix(metrics, classes, "MobileNetV2")


def compute_metrics_full(y_true, y_pred, y_prob, classes):
    from utils.evaluation import compute_metrics
    metrics = compute_metrics(y_true, y_pred, classes)
    metrics["y_prob"] = y_prob
    metrics["y_pred"] = y_pred
    metrics["y_true"] = y_true
    return metrics


def _resize_batch(X, size):
    return np.array([cv2.resize(img, (size, size)) for img in X])


def _show_metrics(metrics, model_name):
    st.markdown(f"#### 📈 {model_name} — Performance Metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    cols = [c1, c2, c3, c4, c5, c6]
    keys = ["accuracy", "precision", "recall", "f1_score", "sensitivity", "specificity"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "Sensitivity", "Specificity"]
    for col, key, label in zip(cols, keys, labels):
        col.metric(label, f"{metrics[key]:.2f}%")


def _show_training_plots(history_data, model_name):
    from utils.evaluation import plot_training_history
    fig = plot_training_history(history_data, model_name)
    st.pyplot(fig)
    plt.close(fig)


def _show_confusion_matrix(metrics, classes, model_name):
    from utils.evaluation import plot_confusion_matrix
    fig = plot_confusion_matrix(metrics["confusion_matrix"], classes, model_name)
    st.pyplot(fig)
    plt.close(fig)


# ─── PAGE: Evaluate ───────────────────────────────────────────────────────────
def page_evaluate():
    st.title("📊 Model Evaluation")

    trained_models = {}
    if st.session_state.cnn_metrics:
        trained_models["Custom CNN"] = st.session_state.cnn_metrics
    if st.session_state.mob_metrics:
        trained_models["MobileNetV2"] = st.session_state.mob_metrics
    if st.session_state.dtc_metrics:
        trained_models["Decision Tree"] = st.session_state.dtc_metrics

    if not trained_models:
        st.warning("⚠️  No trained models found. Go to 🧠 Train Model first.")
        return

    classes = st.session_state.classes or ["Parasitized", "Uninfected"]

    selected = st.selectbox("Select model to evaluate", list(trained_models.keys()))
    metrics  = trained_models[selected]

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Metrics", "🔲 Confusion Matrix", "📉 ROC Curve", "🔍 Compare Models"]
    )

    with tab1:
        _show_metrics(metrics, selected)
        st.markdown("#### Detailed Classification Report")
        st.text(metrics.get("classification_report", "N/A"))

    with tab2:
        from utils.evaluation import plot_confusion_matrix
        fig = plot_confusion_matrix(metrics["confusion_matrix"], classes, selected)
        st.pyplot(fig)
        plt.close(fig)

        # Download button
        buf = io.BytesIO()
        fig2 = plot_confusion_matrix(metrics["confusion_matrix"], classes, selected)
        fig2.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig2)
        st.download_button(
            "⬇️  Download Confusion Matrix",
            data=buf.getvalue(),
            file_name=f"confusion_matrix_{selected.replace(' ', '_')}.png",
            mime="image/png"
        )

    with tab3:
        if "y_prob" in metrics and "y_true" in metrics:
            from utils.evaluation import plot_roc_curve
            fig = plot_roc_curve(
                metrics["y_true"], metrics["y_prob"], classes, selected
            )
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("ROC curve only available for neural network models (not Decision Tree).")

    with tab4:
        if len(trained_models) > 1:
            from utils.evaluation import plot_model_comparison
            fig = plot_model_comparison(trained_models)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Train at least 2 models to see a comparison chart.")


# ─── PAGE: Predict ────────────────────────────────────────────────────────────
def page_predict():
    st.title("🔬 Predict — Upload RBC Image")

    # ── Model selection ────────────────────────────────────
    available_models = {}
    if st.session_state.cnn_model is not None:
        available_models["Custom CNN (in session)"] = ("cnn", 64)
    if st.session_state.mob_model is not None:
        available_models["MobileNetV2 (in session)"] = ("mob", 128)
    if os.path.exists("models/cnn_model.h5"):
        available_models["Custom CNN (saved .h5)"] = ("cnn_file", 64)
    if os.path.exists("models/mobilenet_model.h5"):
        available_models["MobileNetV2 (saved .h5)"] = ("mob_file", 128)

    if not available_models:
        st.warning("⚠️  No trained model available. Train a model first (🧠 Train Model).")
        return

    model_key = st.selectbox("Select model for prediction", list(available_models.keys()))
    model_tag, img_size = available_models[model_key]

    # ── Image upload ───────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload a microscopic RBC blood smear image",
        type=["jpg", "jpeg", "png", "bmp", "tiff"]
    )

    if uploaded_file is None:
        st.markdown("""
        <div class="info-box">
        📌 Upload a microscopic blood smear image (JPG / PNG / BMP).<br>
        The model will classify it as <b>Parasitized (Infected)</b> or <b>Uninfected (Healthy)</b>.
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Load & display uploaded image ─────────────────────
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    # ── Run prediction ─────────────────────────────────────
    with col2:
        with st.spinner("Analysing image…"):
            model    = _get_model(model_tag)
            img_inp  = cv2.resize(img_rgb, (img_size, img_size))
            img_norm = img_inp.astype(np.float32) / 255.0
            img_bat  = np.expand_dims(img_norm, axis=0)

            probs    = model.predict(img_bat, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            classes  = st.session_state.classes or ["Parasitized", "Uninfected"]
            pred_cls = classes[pred_idx]
            conf     = float(probs[pred_idx]) * 100

        # ── Result card ────────────────────────────────────
        is_infected = (pred_cls == "Parasitized")
        card_class  = "result-infected" if is_infected else "result-healthy"
        icon        = "⚠️" if is_infected else "✅"
        colour      = "#e94560" if is_infected else "#2a9d8f"
        message     = "INFECTED — Please consult a doctor" if is_infected else "HEALTHY — No infection detected"

        st.markdown(f"""
        <div class="{card_class}">
            <div class="result-title" style="color:{colour};">{icon}  {pred_cls.upper()}</div>
            <div class="result-conf">{message}</div>
            <div class="result-conf" style="margin-top:12px;">
                Confidence: <b>{conf:.2f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Probability bars ───────────────────────────────
        st.markdown("**Class Probabilities**")
        for i, cls_name in enumerate(classes):
            p = float(probs[i]) * 100
            bar_col = "#e94560" if cls_name == "Parasitized" else "#2a9d8f"
            st.markdown(f"`{cls_name}`")
            st.progress(int(p))
            st.markdown(f"<small style='color:{bar_col};'>{p:.2f}%</small>",
                        unsafe_allow_html=True)

    # ── Confidence interpretation ──────────────────────────
    st.markdown("---")
    if conf >= 90:
        st.success(f"🟢 **High Confidence ({conf:.1f}%)** — Model is very certain about this prediction.")
    elif conf >= 70:
        st.warning(f"🟡 **Medium Confidence ({conf:.1f}%)** — Consider re-examining the sample.")
    else:
        st.error(f"🔴 **Low Confidence ({conf:.1f}%)** — Result may be unreliable. Please verify manually.")


def _get_model(model_tag):
    import tensorflow as tf
    if model_tag == "cnn":
        return st.session_state.cnn_model
    elif model_tag == "mob":
        return st.session_state.mob_model
    elif model_tag == "cnn_file":
        return tf.keras.models.load_model("models/cnn_model.h5")
    elif model_tag == "mob_file":
        return tf.keras.models.load_model("models/mobilenet_model.h5")


# ─── Router ───────────────────────────────────────────────────────────────────
def main():
    page = render_sidebar()

    if   page == "🏠  Home":        page_home()
    elif page == "📂  Dataset":     page_dataset()
    elif page == "🧠  Train Model": page_train()
    elif page == "📊  Evaluate":    page_evaluate()
    elif page == "🔬  Predict":     page_predict()


if __name__ == "__main__":
    main()
