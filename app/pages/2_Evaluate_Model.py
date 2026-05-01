"""
app/pages/2_Evaluate_Model.py
──────────────────────────────
Streamlit sub-page: Model Evaluation
Full evaluation dashboard with:
    - Metrics table (Accuracy, Precision, Recall, F1, Sensitivity, Specificity)
    - Confusion matrix heatmap (downloadable)
    - ROC / AUC curve
    - Training history graphs
    - Side-by-side model comparison
    - Sample prediction grid

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
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Evaluate — RBC Detector",
    page_icon  = "📊",
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

.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.metric-value { font-size: 1.7rem; font-weight: 700; color: #e94560; }
.metric-label { font-size: 0.8rem; color: #a0a0c0; margin-top: 4px; }

.info-box {
    background: #16213e;
    border-left: 4px solid #e94560;
    border-radius: 6px;
    padding: 12px 16px;
    color: #c0c0d8;
    font-size: 0.9rem;
    margin: 10px 0;
}

.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e0e0f0;
    margin: 20px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #302b63;
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _check_state():
    if not any([
        st.session_state.get("cnn_metrics"),
        st.session_state.get("mob_metrics"),
        st.session_state.get("dtc_metrics"),
    ]):
        st.warning("⚠️  No trained models found.")
        st.markdown("Please go to **🧠 Train Model** page and train at least one model first.")
        st.stop()


def _fig_to_bytes(fig) -> bytes:
    """Convert matplotlib figure to PNG bytes for download."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf.getvalue()


def _metric_cards(metrics: dict):
    keys   = ["accuracy", "precision", "recall", "f1_score", "sensitivity", "specificity"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "Sensitivity", "Specificity"]
    cols   = st.columns(6)
    for col, key, label in zip(cols, keys, labels):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics[key]:.1f}%</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def _confusion_matrix_tab(metrics: dict, classes: list, model_name: str):
    from utils.evaluation import plot_confusion_matrix

    st.markdown('<p class="section-title">Confusion Matrix</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        fig = plot_confusion_matrix(metrics["confusion_matrix"], classes, model_name)
        st.pyplot(fig)
        plt.close(fig)

        # Download button
        fig2  = plot_confusion_matrix(metrics["confusion_matrix"], classes, model_name)
        bytes_ = _fig_to_bytes(fig2)
        plt.close(fig2)
        st.download_button(
            label     = "⬇️  Download PNG",
            data      = bytes_,
            file_name = f"confusion_matrix_{model_name.replace(' ', '_')}.png",
            mime      = "image/png"
        )

    with col2:
        cm = metrics["confusion_matrix"]
        st.markdown("**Matrix Breakdown**")
        st.markdown(f"""
        <div class="info-box">
        ✅ <b>True Positives</b>  (Parasitized → Parasitized): <b>{cm[0,0]}</b><br>
        ✅ <b>True Negatives</b>  (Uninfected → Uninfected):   <b>{cm[1,1]}</b><br>
        ❌ <b>False Positives</b> (Uninfected → Parasitized):  <b>{cm[1,0]}</b><br>
        ❌ <b>False Negatives</b> (Parasitized → Uninfected):  <b>{cm[0,1]}</b>
        </div>
        """, unsafe_allow_html=True)

        total    = cm.sum()
        correct  = cm[0,0] + cm[1,1]
        wrong    = total - correct
        st.metric("Total Predictions", total)
        st.metric("Correct",  correct, delta=f"+{correct}")
        st.metric("Incorrect", wrong,  delta=f"-{wrong}", delta_color="inverse")


def _roc_tab(metrics: dict, classes: list, model_name: str):
    from utils.evaluation import plot_roc_curve

    if "y_prob" not in metrics or metrics["y_prob"] is None:
        st.info("ℹ️  ROC curve is only available for neural network models (CNN / MobileNetV2).")
        return

    st.markdown('<p class="section-title">ROC — AUC Curve</p>', unsafe_allow_html=True)

    fig   = plot_roc_curve(metrics["y_true"], metrics["y_prob"], classes, model_name)
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.pyplot(fig)
        plt.close(fig)

        fig2   = plot_roc_curve(metrics["y_true"], metrics["y_prob"], classes, model_name)
        bytes_ = _fig_to_bytes(fig2)
        plt.close(fig2)
        st.download_button(
            label     = "⬇️  Download ROC PNG",
            data      = bytes_,
            file_name = f"roc_curve_{model_name.replace(' ', '_')}.png",
            mime      = "image/png"
        )

    with col2:
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import label_binarize

        y_bin = label_binarize(metrics["y_true"], classes=list(range(len(classes))))
        try:
            auc_score = roc_auc_score(y_bin, metrics["y_prob"], average="macro")
            st.markdown(f"""
            <div class="info-box">
            <b>Macro AUC Score</b><br>
            <span style="font-size:2rem; color:#e94560; font-weight:700;">
                {auc_score:.4f}
            </span><br><br>
            AUC closer to <b>1.0</b> = perfect classifier.<br>
            AUC = <b>0.5</b> = random guessing.
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.info("AUC score could not be computed.")


def _history_tab(history: dict, model_name: str):
    from utils.evaluation import plot_training_history

    if history is None:
        st.info("ℹ️  Training history not available for this model.")
        return

    st.markdown('<p class="section-title">Training History</p>', unsafe_allow_html=True)

    fig    = plot_training_history(history, model_name)
    st.pyplot(fig)
    plt.close(fig)

    # Stats table
    n      = len(history["accuracy"])
    best_epoch = int(np.argmax(history["val_accuracy"])) + 1
    best_acc   = max(history["val_accuracy"]) * 100
    final_loss = history["val_loss"][-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Epochs",       n)
    col2.metric("Best Val Acc",       f"{best_acc:.2f}%")
    col3.metric("Best Epoch",         best_epoch)
    col4.metric("Final Val Loss",     f"{final_loss:.4f}")

    # Epoch detail table
    with st.expander("📋 Full Epoch Log"):
        rows = []
        for i in range(n):
            rows.append({
                "Epoch":      i + 1,
                "Train Acc %": f"{history['accuracy'][i]*100:.2f}",
                "Val Acc %":   f"{history['val_accuracy'][i]*100:.2f}",
                "Train Loss":  f"{history['loss'][i]:.4f}",
                "Val Loss":    f"{history['val_loss'][i]:.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _report_tab(metrics: dict):
    st.markdown('<p class="section-title">Classification Report</p>', unsafe_allow_html=True)
    st.code(metrics.get("classification_report", "Not available"), language="text")

    # Download as txt
    report_bytes = metrics.get("classification_report", "").encode("utf-8")
    st.download_button(
        label     = "⬇️  Download Report TXT",
        data      = report_bytes,
        file_name = "classification_report.txt",
        mime      = "text/plain"
    )


def _comparison_tab(all_results: dict, classes: list):
    from utils.evaluation import plot_model_comparison, plot_confusion_matrix

    st.markdown('<p class="section-title">Model Comparison</p>', unsafe_allow_html=True)

    if len(all_results) < 2:
        st.info("ℹ️  Train at least 2 models to see a comparison chart.")
        return

    # Bar chart
    fig    = plot_model_comparison(all_results)
    st.pyplot(fig)
    plt.close(fig)

    # Summary table
    st.markdown("**Summary Table**")
    rows = []
    for name, m in all_results.items():
        rows.append({
            "Model":         name,
            "Accuracy %":    m["accuracy"],
            "Precision %":   m["precision"],
            "Recall %":      m["recall"],
            "F1-Score %":    m["f1_score"],
            "Sensitivity %": m["sensitivity"],
            "Specificity %": m["specificity"],
        })
    df = pd.DataFrame(rows)
    st.dataframe(df.style.highlight_max(
        subset=["Accuracy %", "Precision %", "Recall %", "F1-Score %"],
        color="#1a4a2e"
    ), use_container_width=True)

    # Side-by-side confusion matrices
    st.markdown("**Confusion Matrices — Side by Side**")
    cols = st.columns(len(all_results))
    for col, (name, m) in zip(cols, all_results.items()):
        with col:
            fig = plot_confusion_matrix(m["confusion_matrix"], classes, name)
            st.pyplot(fig)
            plt.close(fig)


# ─── Main page ────────────────────────────────────────────────────────────────
def main():
    _check_state()

    st.markdown("""
    <div class="page-header">
        <h2>📊 Model Evaluation</h2>
        <p>Deep-dive into model performance — metrics, confusion matrix,
        ROC curve, training history and multi-model comparison.</p>
    </div>
    """, unsafe_allow_html=True)

    classes = st.session_state.get("classes") or ["Parasitized", "Uninfected"]

    # ── Collect trained models ─────────────────────────────
    available = {}
    if st.session_state.get("cnn_metrics"):
        available["Custom CNN"] = {
            "metrics": st.session_state.cnn_metrics,
            "history": st.session_state.get("cnn_history"),
        }
    if st.session_state.get("mob_metrics"):
        available["MobileNetV2"] = {
            "metrics": st.session_state.mob_metrics,
            "history": st.session_state.get("mob_history"),
        }
    if st.session_state.get("dtc_metrics"):
        available["Decision Tree"] = {
            "metrics": st.session_state.dtc_metrics,
            "history": None,
        }

    # ── Model selector ─────────────────────────────────────
    selected_name = st.selectbox(
        "Select model to evaluate",
        list(available.keys())
    )
    selected = available[selected_name]
    metrics  = selected["metrics"]
    history  = selected["history"]

    # ── Top metric cards ───────────────────────────────────
    st.markdown("#### Performance at a Glance")
    _metric_cards(metrics)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔲 Confusion Matrix",
        "📉 ROC Curve",
        "📈 Training History",
        "📋 Classification Report",
        "🏆 Compare Models",
    ])

    with tab1:
        _confusion_matrix_tab(metrics, classes, selected_name)

    with tab2:
        _roc_tab(metrics, classes, selected_name)

    with tab3:
        _history_tab(history, selected_name)

    with tab4:
        _report_tab(metrics)

    with tab5:
        all_metrics = {name: data["metrics"] for name, data in available.items()}
        _comparison_tab(all_metrics, classes)


main()
