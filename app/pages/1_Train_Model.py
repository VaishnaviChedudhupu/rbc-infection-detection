"""
app/pages/1_Train_Model.py
──────────────────────────
Streamlit sub-page: Model Training
Dedicated full-page training interface with:
    - Model selection (CNN / MobileNetV2 / Decision Tree)
    - Hyperparameter controls
    - Live epoch-by-epoch progress table
    - Real-time accuracy/loss chart updates
    - Post-training metrics, confusion matrix, ROC curve

Author  : C. Vaishnavi (22R91A7325)
Project : AI-Enhanced Microscopic Image Classification for RBC Infection
College : TKREC, Hyderabad
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Train Model — RBC Detector",
    page_icon  = "🧠",
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

.metric-pill {
    display: inline-block;
    background: #16213e;
    border: 1px solid #0f3460;
    border-radius: 20px;
    padding: 6px 18px;
    margin: 4px;
    font-size: 0.88rem;
    color: #c0c0d8;
}
.metric-pill b { color: #e94560; }

.epoch-row-good { color: #2a9d8f; }
.epoch-row-bad  { color: #e94560; }

.info-box {
    background: #16213e;
    border-left: 4px solid #e94560;
    border-radius: 6px;
    padding: 12px 16px;
    color: #c0c0d8;
    font-size: 0.9rem;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Session state check ──────────────────────────────────────────────────────
def _check_state():
    if "splits" not in st.session_state or st.session_state.get("splits") is None:
        st.warning("⚠️  Dataset not loaded or split yet.")
        st.markdown("Please go to **📂 Dataset** page first and complete Steps 1 & 2.")
        st.stop()

    for key in ["cnn_model", "mob_model", "dtc_model",
                "cnn_history", "mob_history",
                "cnn_metrics", "mob_metrics", "dtc_metrics",
                "classes"]:
        if key not in st.session_state:
            st.session_state[key] = None


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _resize_batch(X, size):
    import cv2
    return np.array([cv2.resize(img, (size, size)) for img in X])


def _metrics_row(metrics: dict):
    keys   = ["accuracy", "precision", "recall", "f1_score", "sensitivity", "specificity"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "Sensitivity", "Specificity"]
    pills  = "".join(
        f'<span class="metric-pill">{lbl}: <b>{metrics[k]:.2f}%</b></span>'
        for k, lbl in zip(keys, labels)
    )
    st.markdown(pills, unsafe_allow_html=True)


def _save_model_outputs(metrics, history, model_name_slug):
    """Persist metrics and history to session state by model slug."""
    if model_name_slug == "cnn":
        st.session_state.cnn_metrics = metrics
        st.session_state.cnn_history = history
    elif model_name_slug == "mob":
        st.session_state.mob_metrics = metrics
        st.session_state.mob_history = history
    elif model_name_slug == "dtc":
        st.session_state.dtc_metrics = metrics


def _live_loss_chart(placeholder, history_data: dict, model_name: str):
    """Render a compact live accuracy+loss chart inside a placeholder."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    fig.patch.set_facecolor("#0d0d1a")

    for ax in axes:
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    epochs_done = list(range(1, len(history_data["accuracy"]) + 1))

    axes[0].plot(epochs_done, history_data["accuracy"],
                 color="#2a9d8f", lw=2, label="Train")
    axes[0].plot(epochs_done, history_data["val_accuracy"],
                 color="#e94560", lw=2, linestyle="--", label="Val")
    axes[0].set_title("Accuracy", color="white", fontsize=11)
    axes[0].set_xlabel("Epoch", color="white", fontsize=9)
    axes[0].legend(fontsize=8, labelcolor="white",
                   facecolor="#0d0d1a", edgecolor="#333355")
    axes[0].grid(True, alpha=0.15)

    axes[1].plot(epochs_done, history_data["loss"],
                 color="#2a9d8f", lw=2, label="Train")
    axes[1].plot(epochs_done, history_data["val_loss"],
                 color="#e94560", lw=2, linestyle="--", label="Val")
    axes[1].set_title("Loss", color="white", fontsize=11)
    axes[1].set_xlabel("Epoch", color="white", fontsize=9)
    axes[1].legend(fontsize=8, labelcolor="white",
                   facecolor="#0d0d1a", edgecolor="#333355")
    axes[1].grid(True, alpha=0.15)

    fig.tight_layout()
    placeholder.pyplot(fig)
    plt.close(fig)


# ─── Decision Tree Training ───────────────────────────────────────────────────
def _train_dtc(splits, classes):
    from utils.models import build_decision_tree
    from utils.evaluation import compute_metrics, plot_confusion_matrix

    st.info("Training Decision Tree Classifier…")

    X_tr = splits["X_train"].reshape(len(splits["X_train"]), -1)
    X_te = splits["X_test"].reshape(len(splits["X_test"]),  -1)

    with st.spinner("Fitting Decision Tree…"):
        dtc = build_decision_tree(max_depth=3)
        dtc.fit(X_tr, splits["y_train"])

    y_pred  = dtc.predict(X_te)
    metrics = compute_metrics(splits["y_test"], y_pred, classes)

    # Save
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(dtc, "models/dtc_model.pkl")
    _save_model_outputs(metrics, None, "dtc")

    st.success("✅ Decision Tree trained and saved → `models/dtc_model.pkl`")

    st.markdown("#### Performance Metrics")
    _metrics_row(metrics)

    st.markdown("#### Confusion Matrix")
    fig = plot_confusion_matrix(metrics["confusion_matrix"], classes, "Decision Tree")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Classification Report")
    st.code(metrics["classification_report"])


# ─── Custom CNN Training ──────────────────────────────────────────────────────
def _train_cnn(splits, classes, epochs, batch_size, lr, dropout):
    from tensorflow.keras.utils import to_categorical
    from utils.models import build_custom_cnn, get_training_callbacks
    from utils.data_pipeline import build_augmentation_generator, compute_class_weights
    from utils.evaluation import (compute_metrics, plot_confusion_matrix,
                                   plot_roc_curve, plot_training_history)

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val,   y_val   = splits["X_val"],   splits["y_val"]
    X_test,  y_test  = splits["X_test"],  splits["y_test"]
    n_cls            = len(classes)

    y_tr_cat = to_categorical(y_train, n_cls)
    y_va_cat = to_categorical(y_val,   n_cls)

    model = build_custom_cnn(X_train.shape[1:], n_cls, dropout, lr)

    os.makedirs("models", exist_ok=True)
    save_path = "models/cnn_model.h5"
    callbacks = get_training_callbacks(save_path)
    aug       = build_augmentation_generator()
    cw        = compute_class_weights(y_train)

    # ── Live training loop ─────────────────────────────────
    st.markdown("#### 🔄 Live Training Progress")
    progress_bar   = st.progress(0)
    status_text    = st.empty()
    chart_holder   = st.empty()
    log_holder     = st.empty()

    history_data   = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    epoch_log      = []

    for epoch in range(epochs):
        hist = model.fit(
            aug.flow(X_train, y_tr_cat, batch_size=batch_size),
            validation_data = (X_val, y_va_cat),
            epochs          = 1,
            class_weight    = cw,
            callbacks       = callbacks,
            verbose         = 0
        )

        for k in history_data:
            history_data[k].append(float(hist.history[k][0]))

        tr_acc  = history_data["accuracy"][-1]   * 100
        val_acc = history_data["val_accuracy"][-1] * 100
        tr_loss = history_data["loss"][-1]
        val_loss= history_data["val_loss"][-1]

        progress_bar.progress(int((epoch + 1) / epochs * 100))
        status_text.markdown(
            f"**Epoch {epoch+1}/{epochs}** — "
            f"Train Acc: `{tr_acc:.2f}%` | Val Acc: `{val_acc:.2f}%` | "
            f"Train Loss: `{tr_loss:.4f}` | Val Loss: `{val_loss:.4f}`"
        )

        epoch_log.append({
            "Epoch": epoch + 1,
            "Train Acc %": f"{tr_acc:.2f}",
            "Val Acc %":   f"{val_acc:.2f}",
            "Train Loss":  f"{tr_loss:.4f}",
            "Val Loss":    f"{val_loss:.4f}",
        })

        if epoch >= 1:
            _live_loss_chart(chart_holder, history_data, "Custom CNN")

        import pandas as pd
        log_holder.dataframe(
            pd.DataFrame(epoch_log).tail(8),
            use_container_width=True
        )

    # Load best weights
    model.load_weights(save_path)

    # Evaluate
    y_prob  = model.predict(X_test, verbose=0)
    y_pred  = np.argmax(y_prob, axis=1)
    metrics = compute_metrics(y_test, y_pred, classes)
    metrics.update({"y_true": y_test, "y_pred": y_pred, "y_prob": y_prob})

    # Save history
    with open("models/cnn_history.pkl", "wb") as f:
        pickle.dump(history_data, f)

    _save_model_outputs(metrics, history_data, "cnn")

    st.success("✅ Custom CNN trained and saved → `models/cnn_model.h5`")

    # ── Results ────────────────────────────────────────────
    st.markdown("#### 📈 Performance Metrics")
    _metrics_row(metrics)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Confusion Matrix")
        fig_cm = plot_confusion_matrix(metrics["confusion_matrix"], classes, "Custom CNN")
        st.pyplot(fig_cm)
        plt.close(fig_cm)
    with col2:
        st.markdown("#### ROC Curve")
        fig_roc = plot_roc_curve(y_test, y_prob, classes, "Custom CNN")
        st.pyplot(fig_roc)
        plt.close(fig_roc)

    st.markdown("#### Training History")
    fig_hist = plot_training_history(history_data, "Custom CNN")
    st.pyplot(fig_hist)
    plt.close(fig_hist)

    st.markdown("#### Classification Report")
    st.code(metrics["classification_report"])


# ─── MobileNetV2 Training ─────────────────────────────────────────────────────
def _train_mobilenet(splits, classes, epochs, finetune_epochs,
                     batch_size, lr, dropout, fine_tune_at):
    from tensorflow.keras.utils import to_categorical
    from utils.models import (build_mobilenet_v2, unfreeze_mobilenet,
                               get_training_callbacks)
    from utils.data_pipeline import compute_class_weights
    from utils.evaluation import (compute_metrics, plot_confusion_matrix,
                                   plot_roc_curve, plot_training_history)

    st.info("Resizing images to 128×128 for MobileNetV2…")
    X_train = _resize_batch(splits["X_train"], 128)
    X_val   = _resize_batch(splits["X_val"],   128)
    X_test  = _resize_batch(splits["X_test"],  128)
    y_train, y_val, y_test = splits["y_train"], splits["y_val"], splits["y_test"]
    n_cls   = len(classes)

    y_tr_cat = to_categorical(y_train, n_cls)
    y_va_cat = to_categorical(y_val,   n_cls)

    model, base_model = build_mobilenet_v2((128, 128, 3), n_cls, dropout, lr, fine_tune_at)

    os.makedirs("models", exist_ok=True)
    save_path = "models/mobilenet_model.h5"
    callbacks = get_training_callbacks(save_path)
    cw        = compute_class_weights(y_train)

    history_data = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

    # Phase 1
    st.markdown(f"#### 🔄 Phase 1 — Frozen Base ({epochs} epochs)")
    progress_bar = st.progress(0)
    status_text  = st.empty()
    chart_holder = st.empty()
    log_holder   = st.empty()
    epoch_log    = []

    for epoch in range(epochs):
        hist = model.fit(
            X_train, y_tr_cat,
            validation_data=(X_val, y_va_cat),
            epochs=1, batch_size=batch_size,
            class_weight=cw, callbacks=callbacks, verbose=0
        )
        for k in history_data:
            history_data[k].append(float(hist.history[k][0]))

        tr_acc  = history_data["accuracy"][-1]   * 100
        val_acc = history_data["val_accuracy"][-1] * 100
        progress_bar.progress(int((epoch + 1) / epochs * 100))
        status_text.markdown(
            f"**Epoch {epoch+1}/{epochs}** — "
            f"Train Acc: `{tr_acc:.2f}%` | Val Acc: `{val_acc:.2f}%`"
        )
        epoch_log.append({
            "Epoch": epoch + 1, "Phase": "Frozen",
            "Train Acc %": f"{tr_acc:.2f}",
            "Val Acc %":   f"{val_acc:.2f}",
        })
        if epoch >= 1:
            _live_loss_chart(chart_holder, history_data, "MobileNetV2")

        import pandas as pd
        log_holder.dataframe(pd.DataFrame(epoch_log).tail(8), use_container_width=True)

    # Phase 2
    if finetune_epochs > 0:
        st.markdown(f"#### 🔄 Phase 2 — Fine-Tuning from Layer {fine_tune_at} ({finetune_epochs} epochs)")
        model    = unfreeze_mobilenet(model, base_model, fine_tune_at, lr / 10)
        pb2      = st.progress(0)
        st2      = st.empty()
        ch2      = st.empty()
        log2     = st.empty()
        elog2    = []

        for epoch in range(finetune_epochs):
            hist = model.fit(
                X_train, y_tr_cat,
                validation_data=(X_val, y_va_cat),
                epochs=1, batch_size=batch_size,
                class_weight=cw, callbacks=callbacks, verbose=0
            )
            for k in history_data:
                history_data[k].append(float(hist.history[k][0]))

            tr_acc  = history_data["accuracy"][-1]   * 100
            val_acc = history_data["val_accuracy"][-1] * 100
            pb2.progress(int((epoch + 1) / finetune_epochs * 100))
            st2.markdown(
                f"**Fine-tune Epoch {epoch+1}/{finetune_epochs}** — "
                f"Train Acc: `{tr_acc:.2f}%` | Val Acc: `{val_acc:.2f}%`"
            )
            elog2.append({
                "Epoch": epoch + 1, "Phase": "Fine-tune",
                "Train Acc %": f"{tr_acc:.2f}",
                "Val Acc %":   f"{val_acc:.2f}",
            })
            if epoch >= 1:
                _live_loss_chart(ch2, history_data, "MobileNetV2")
            import pandas as pd
            log2.dataframe(pd.DataFrame(elog2).tail(8), use_container_width=True)

    model.load_weights(save_path)

    y_prob  = model.predict(X_test, verbose=0)
    y_pred  = np.argmax(y_prob, axis=1)
    metrics = compute_metrics(y_test, y_pred, classes)
    metrics.update({"y_true": y_test, "y_pred": y_pred, "y_prob": y_prob})

    with open("models/mobilenet_history.pkl", "wb") as f:
        pickle.dump(history_data, f)

    _save_model_outputs(metrics, history_data, "mob")

    st.success("✅ MobileNetV2 trained and saved → `models/mobilenet_model.h5`")

    st.markdown("#### 📈 Performance Metrics")
    _metrics_row(metrics)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Confusion Matrix")
        fig_cm = plot_confusion_matrix(metrics["confusion_matrix"], classes, "MobileNetV2")
        st.pyplot(fig_cm)
        plt.close(fig_cm)
    with col2:
        st.markdown("#### ROC Curve")
        fig_roc = plot_roc_curve(y_test, y_prob, classes, "MobileNetV2")
        st.pyplot(fig_roc)
        plt.close(fig_roc)

    st.markdown("#### Training History")
    fig_hist = plot_training_history(history_data, "MobileNetV2")
    st.pyplot(fig_hist)
    plt.close(fig_hist)

    st.markdown("#### Classification Report")
    st.code(metrics["classification_report"])


# ─── Main page ────────────────────────────────────────────────────────────────
def main():
    _check_state()

    st.markdown("""
    <div class="page-header">
        <h2>🧠 Train Model</h2>
        <p>Select a model, configure hyperparameters and start training.
        Live epoch progress and charts update in real time.</p>
    </div>
    """, unsafe_allow_html=True)

    splits  = st.session_state.splits
    classes = st.session_state.classes or ["Parasitized", "Uninfected"]

    # ── Dataset summary ────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Train Samples", splits["X_train"].shape[0])
    c2.metric("Val Samples",   splits["X_val"].shape[0])
    c3.metric("Test Samples",  splits["X_test"].shape[0])

    st.markdown("---")

    # ── Model selector ─────────────────────────────────────
    st.markdown("#### 1️⃣ Select Model")
    model_choice = st.radio(
        "Model",
        ["Custom CNN", "MobileNetV2 Transfer Learning", "Decision Tree (Baseline)"],
        horizontal=True
    )

    # ── Architecture info ──────────────────────────────────
    with st.expander("ℹ️ Model Architecture Details"):
        if model_choice == "Custom CNN":
            st.markdown("""
            ```
            Input (64×64×3)
             → Conv2D(32) → BatchNorm → MaxPool
             → Conv2D(32) → BatchNorm → MaxPool
             → Conv2D(64) → BatchNorm → MaxPool
             → Conv2D(64) → BatchNorm → MaxPool
             → Flatten → Dense(256, ReLU) → Dropout
             → Dense(128, ReLU) → Dropout
             → Dense(2, Softmax)
            Optimizer: Adam  |  Loss: Categorical Cross-Entropy
            ```
            """)
        elif model_choice == "MobileNetV2 Transfer Learning":
            st.markdown("""
            ```
            Input (128×128×3)
             → MobileNetV2 base (ImageNet weights)
             → GlobalAveragePooling2D
             → Dense(256, ReLU) → Dropout
             → Dense(128, ReLU) → Dropout
             → Dense(2, Softmax)
            Phase 1: frozen base  |  Phase 2: fine-tune from layer N
            ```
            """)
        else:
            st.markdown("""
            ```
            Decision Tree Classifier (Scikit-learn)
            Criterion : Gini Impurity
            Max Depth : 3 (configurable)
            Features  : Flattened pixel values (64×64×3 = 12,288)
            ```
            """)

    # ── Hyperparameters ────────────────────────────────────
    st.markdown("#### 2️⃣ Hyperparameters")

    if model_choice == "Decision Tree (Baseline)":
        st.markdown("""
        <div class="info-box">
        Decision Tree uses fixed hyperparameters (max_depth=3).
        No epoch or learning-rate settings needed.
        </div>
        """, unsafe_allow_html=True)
        epochs = finetune_epochs = batch_size = fine_tune_at = 0
        lr = dropout = 0.0

    elif model_choice == "Custom CNN":
        c1, c2, c3, c4 = st.columns(4)
        epochs      = c1.number_input("Epochs",        1,  100, 10)
        batch_size  = c2.number_input("Batch Size",    8,  256, 32)
        lr          = c3.number_input("Learning Rate", 1e-6, 1e-1, 0.001, format="%.5f")
        dropout     = c4.slider("Dropout", 0.1, 0.7, 0.4, 0.05)
        finetune_epochs = fine_tune_at = 0

    else:  # MobileNetV2
        c1, c2, c3, c4 = st.columns(4)
        epochs          = c1.number_input("Phase 1 Epochs",    1,  50, 10)
        finetune_epochs = c2.number_input("Fine-tune Epochs",  0,  30,  5)
        batch_size      = c3.number_input("Batch Size",        8, 256, 32)
        lr              = c4.number_input("Learning Rate", 1e-6, 1e-2, 0.0001, format="%.6f")
        c5, c6          = st.columns(2)
        dropout         = c5.slider("Dropout", 0.1, 0.7, 0.4, 0.05)
        fine_tune_at    = c6.number_input("Fine-tune from Layer", 50, 155, 100)

    # ── Train button ───────────────────────────────────────
    st.markdown("#### 3️⃣ Start Training")
    col1, _ = st.columns([1, 3])
    with col1:
        train_btn = st.button("🚀 Train Now", use_container_width=True)

    if train_btn:
        st.markdown("---")
        if model_choice == "Decision Tree (Baseline)":
            _train_dtc(splits, classes)

        elif model_choice == "Custom CNN":
            _train_cnn(splits, classes, epochs, batch_size, lr, dropout)

        else:
            _train_mobilenet(
                splits, classes, epochs, finetune_epochs,
                batch_size, lr, dropout, fine_tune_at
            )


main()
