"""
app/pages/3_Predict.py
───────────────────────
Streamlit sub-page: Predict
Real-time RBC infection prediction with:
    - Single image upload → instant result card
    - Probability bars per class
    - Confidence tier (HIGH / MEDIUM / LOW)
    - Side-by-side original vs preprocessed image
    - Batch folder prediction with CSV export
    - Prediction history table (current session)

Author  : C. Vaishnavi (22R91A7325)
Project : AI-Enhanced Microscopic Image Classification for RBC Infection
College : TKREC, Hyderabad
"""

import os
import sys
import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Predict — RBC Detector",
    page_icon  = "🔬",
    layout     = "wide",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.page-header {
    background: linear-gradient(135deg, #0f0c29, #302b63);
    border-radius: 14px;
    padding: 28px 36px;
    margin-bottom: 24px;
    border: 1px solid #302b63;
}
.page-header h2 { color: #e94560; margin: 0; font-size: 1.8rem; }
.page-header p  { color: #c0c0d8; margin-top: 6px; font-size: 0.95rem; }

.result-infected {
    background: linear-gradient(135deg, #2d0000, #5a0000);
    border: 2px solid #e94560;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}
.result-healthy {
    background: linear-gradient(135deg, #002d22, #005a43);
    border: 2px solid #2a9d8f;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}
.result-title   { font-size: 2rem;  font-weight: 700; }
.result-sub     { font-size: 1rem;  color: #e0e0e0; margin-top: 8px; }
.result-conf    { font-size: 1.2rem; margin-top: 14px; font-weight: 600; }

.conf-high   { color: #2a9d8f; }
.conf-medium { color: #f4a261; }
.conf-low    { color: #e94560; }

.info-box {
    background: #16213e;
    border-left: 4px solid #e94560;
    border-radius: 6px;
    padding: 12px 16px;
    color: #c0c0d8;
    font-size: 0.9rem;
    margin: 10px 0;
}

.history-row-infected   { color: #e94560; }
.history-row-uninfected { color: #2a9d8f; }
</style>
""", unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _check_models():
    has_session = (
        st.session_state.get("cnn_model") is not None or
        st.session_state.get("mob_model") is not None
    )
    has_saved = (
        os.path.exists("models/cnn_model.h5") or
        os.path.exists("models/mobilenet_model.h5")
    )
    if not has_session and not has_saved:
        st.warning("⚠️  No trained model found.")
        st.markdown("Please go to **🧠 Train Model** and train a model first.")
        st.stop()


def _get_model(tag: str):
    import tensorflow as tf
    if tag == "cnn_session":
        return st.session_state.cnn_model, 64
    elif tag == "mob_session":
        return st.session_state.mob_model, 128
    elif tag == "cnn_file":
        return tf.keras.models.load_model("models/cnn_model.h5"), 64
    elif tag == "mob_file":
        return tf.keras.models.load_model("models/mobilenet_model.h5"), 128


def _preprocess(img_rgb: np.ndarray, size: int) -> np.ndarray:
    img_resized = cv2.resize(img_rgb, (size, size))
    img_norm    = img_resized.astype(np.float32) / 255.0
    return np.expand_dims(img_norm, axis=0)


def _confidence_tier(conf: float) -> tuple:
    """Returns (label, css_class, streamlit_fn)."""
    if conf >= 90:
        return "HIGH Confidence",   "conf-high",   st.success
    elif conf >= 70:
        return "MEDIUM Confidence", "conf-medium", st.warning
    else:
        return "LOW Confidence",    "conf-low",    st.error


def _result_card(pred_class: str, confidence: float, probabilities: np.ndarray,
                 classes: list):
    is_infected = (pred_class == "Parasitized")
    card_class  = "result-infected" if is_infected else "result-healthy"
    colour      = "#e94560"        if is_infected else "#2a9d8f"
    icon        = "⚠️"             if is_infected else "✅"
    sub_msg     = "Malaria parasite detected — consult a doctor immediately" \
                  if is_infected else "No infection detected — RBC appears healthy"

    tier_label, tier_css, _ = _confidence_tier(confidence)

    st.markdown(f"""
    <div class="{card_class}">
        <div class="result-title" style="color:{colour};">{icon}  {pred_class.upper()}</div>
        <div class="result-sub">{sub_msg}</div>
        <div class="result-conf">
            Confidence: <span class="{tier_css}">{confidence:.2f}%</span>
            &nbsp;·&nbsp;
            <span class="{tier_css}">{tier_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Probability bars
    st.markdown("**Class Probabilities**")
    for i, cls_name in enumerate(classes):
        p      = float(probabilities[i]) * 100
        colour = "#e94560" if cls_name == "Parasitized" else "#2a9d8f"
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(int(p))
        with col2:
            st.markdown(
                f"<span style='color:{colour}; font-weight:600;'>"
                f"{cls_name}: {p:.2f}%</span>",
                unsafe_allow_html=True
            )


def _prob_bar_chart(probabilities: np.ndarray, classes: list,
                    pred_class: str) -> plt.Figure:
    """Horizontal bar chart of class probabilities."""
    colours = ["#e94560" if c == "Parasitized" else "#2a9d8f" for c in classes]
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")

    bars = ax.barh(classes, [p * 100 for p in probabilities],
                   color=colours, edgecolor="white", linewidth=0.4, height=0.5)
    for bar, prob in zip(bars, probabilities):
        ax.text(bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{prob*100:.2f}%",
                va="center", ha="left",
                color="white", fontsize=11, fontweight="bold")

    ax.set_xlim(0, 115)
    ax.set_xlabel("Probability (%)", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.xaxis.label.set_color("white")
    fig.tight_layout()
    return fig


# ─── Single Image Prediction ──────────────────────────────────────────────────
def _single_predict(model_tag: str):
    classes = st.session_state.get("classes") or ["Parasitized", "Uninfected"]

    uploaded = st.file_uploader(
        "Upload a microscopic RBC blood smear image",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        key="single_uploader"
    )

    if uploaded is None:
        st.markdown("""
        <div class="info-box">
        📌 Upload a blood smear image (JPG / PNG / BMP / TIFF).<br>
        The model will classify it as <b>Parasitized (Infected)</b>
        or <b>Uninfected (Healthy)</b> in real time.
        </div>
        """, unsafe_allow_html=True)
        return

    # Decode
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with st.spinner("Loading model and running inference…"):
        model, img_size = _get_model(model_tag)
        img_input       = _preprocess(img_rgb, img_size)
        probabilities   = model.predict(img_input, verbose=0)[0]
        pred_idx        = int(np.argmax(probabilities))
        pred_class      = classes[pred_idx]
        confidence      = float(probabilities[pred_idx]) * 100

    # ── Layout ────────────────────────────────────────────
    col_img, col_result = st.columns([1, 1.4])

    with col_img:
        st.markdown("**Uploaded Image**")
        st.image(img_rgb, use_container_width=True)

        # Preprocessed preview
        st.markdown("**Preprocessed (model input)**")
        preview = cv2.resize(img_rgb, (img_size, img_size))
        st.image(preview, use_container_width=True,
                 caption=f"Resized to {img_size}×{img_size}")

    with col_result:
        st.markdown("**Prediction Result**")
        _result_card(pred_class, confidence, probabilities, classes)

        # Probability chart
        fig = _prob_bar_chart(probabilities, classes, pred_class)
        st.pyplot(fig)
        plt.close(fig)

    # ── Confidence interpretation ──────────────────────────
    tier_label, _, tier_fn = _confidence_tier(confidence)
    tier_fn(f"{tier_label} ({confidence:.1f}%) — "
            + ("Model is very certain." if confidence >= 90
               else "Consider re-examining the sample." if confidence >= 70
               else "Low confidence — verify manually with a pathologist."))

    # ── Save to history ────────────────────────────────────
    st.session_state.prediction_history.append({
        "Image":      uploaded.name,
        "Prediction": pred_class,
        "Confidence": f"{confidence:.2f}%",
        "Parasitized %": f"{float(probabilities[0])*100:.2f}",
        "Uninfected %":  f"{float(probabilities[1])*100:.2f}",
        "Model":      model_tag,
    })


# ─── Batch Prediction ─────────────────────────────────────────────────────────
def _batch_predict(model_tag: str):
    classes = st.session_state.get("classes") or ["Parasitized", "Uninfected"]

    st.markdown("""
    <div class="info-box">
    📌 Upload <b>multiple</b> RBC images at once for batch classification.<br>
    Results will be displayed in a table and available as a CSV download.
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload multiple RBC images",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        accept_multiple_files=True,
        key="batch_uploader"
    )

    if not uploaded_files:
        return

    col1, _ = st.columns([1, 3])
    with col1:
        run_btn = st.button("▶️  Run Batch Prediction", use_container_width=True)

    if not run_btn:
        return

    with st.spinner(f"Running prediction on {len(uploaded_files)} images…"):
        model, img_size = _get_model(model_tag)
        results = []
        preview_images = []
        preview_labels = []

        progress = st.progress(0)
        for i, f in enumerate(uploaded_files):
            file_bytes = np.frombuffer(f.read(), np.uint8)
            img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_input = _preprocess(img_rgb, img_size)
            probs     = model.predict(img_input, verbose=0)[0]
            pred_idx  = int(np.argmax(probs))
            pred_cls  = classes[pred_idx]
            conf      = float(probs[pred_idx]) * 100

            results.append({
                "Image":         f.name,
                "Prediction":    pred_cls,
                "Confidence %":  round(conf, 2),
                "Parasitized %": round(float(probs[0]) * 100, 2),
                "Uninfected %":  round(float(probs[1]) * 100, 2),
            })
            preview_images.append(cv2.resize(img_rgb, (96, 96)))
            preview_labels.append(f"{pred_cls} ({conf:.0f}%)")
            progress.progress(int((i + 1) / len(uploaded_files) * 100))

    # ── Summary ────────────────────────────────────────────
    df = pd.DataFrame(results)
    infected_n   = (df["Prediction"] == "Parasitized").sum()
    uninfected_n = len(df) - infected_n

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Images",   len(df))
    c2.metric("⚠️  Infected",   infected_n)
    c3.metric("✅ Healthy",      uninfected_n)

    # ── Results table ──────────────────────────────────────
    st.markdown("#### Batch Results")
    st.dataframe(df, use_container_width=True)

    # CSV download
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label     = "⬇️  Download CSV",
        data      = csv_bytes,
        file_name = "batch_predictions.csv",
        mime      = "text/csv"
    )

    # ── Image grid preview ─────────────────────────────────
    st.markdown("#### Preview Grid")
    cols = st.columns(min(8, len(preview_images)))
    for col, img, label in zip(cols, preview_images, preview_labels):
        colour = "#e94560" if "Parasitized" in label else "#2a9d8f"
        with col:
            st.image(img, use_container_width=True)
            st.markdown(
                f"<small style='color:{colour};'>{label}</small>",
                unsafe_allow_html=True
            )


# ─── Prediction History ───────────────────────────────────────────────────────
def _show_history():
    history = st.session_state.prediction_history
    if not history:
        st.info("No predictions made yet in this session.")
        return

    df = pd.DataFrame(history)
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.download_button(
            "⬇️  Download History CSV",
            data=csv_bytes,
            file_name="prediction_history.csv",
            mime="text/csv"
        )
    with col2:
        if st.button("🗑️  Clear History"):
            st.session_state.prediction_history = []
            st.rerun()


# ─── Main page ────────────────────────────────────────────────────────────────
def main():
    _check_models()

    st.markdown("""
    <div class="page-header">
        <h2>🔬 Predict — RBC Infection Detection</h2>
        <p>Upload a microscopic blood smear image and get an instant
        AI-powered prediction — Parasitized (Infected) or Uninfected (Healthy).</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Model selector ─────────────────────────────────────
    available_models = {}
    if st.session_state.get("cnn_model") is not None:
        available_models["Custom CNN (session)"] = "cnn_session"
    if st.session_state.get("mob_model") is not None:
        available_models["MobileNetV2 (session)"] = "mob_session"
    if os.path.exists("models/cnn_model.h5"):
        available_models["Custom CNN (saved .h5)"] = "cnn_file"
    if os.path.exists("models/mobilenet_model.h5"):
        available_models["MobileNetV2 (saved .h5)"] = "mob_file"

    col1, col2 = st.columns([1, 2])
    with col1:
        model_label = st.selectbox("Select Model", list(available_models.keys()))
    model_tag = available_models[model_label]

    st.markdown("---")

    # ── Mode tabs ──────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🖼️  Single Image",
        "📂  Batch Prediction",
        "📋  Prediction History",
    ])

    with tab1:
        _single_predict(model_tag)

    with tab2:
        _batch_predict(model_tag)

    with tab3:
        st.markdown("#### Session Prediction History")
        _show_history()


main()
