"""
evaluation.py
─────────────
Model evaluation utilities:
    - Classification metrics  (Accuracy, Precision, Recall, F1, Sensitivity, Specificity)
    - Confusion matrix heatmap
    - ROC curve (AUC)
    - Accuracy / Loss training graphs
    - Feature map visualisation
    - Prediction confidence display

Author  : C. Vaishnavi (22R91A7325)
Project : AI-Enhanced Microscopic Image Classification for RBC Infection
College : TKREC, Hyderabad
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # Non-interactive backend for saving
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import logging
import os

log = logging.getLogger(__name__)

# ─── Style ────────────────────────────────────────────────────────────────────
PALETTE    = "#E63946"          # Parasitized / infected colour
PALETTE2   = "#2A9D8F"          # Uninfected colour
PALETTE3   = "#264653"          # Dark background accent
FIG_DPI    = 150
FONT_TITLE = 15
FONT_LABEL = 12


# ─── Core Metrics ─────────────────────────────────────────────────────────────
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list
) -> dict:
    """
    Compute all metrics described in document Section 6.4 and 9 (Conclusion).

    Returns a dict with:
        accuracy, precision, recall, f1, sensitivity, specificity,
        per_class_report (str), confusion_matrix (ndarray)
    """
    acc  = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
    cm   = confusion_matrix(y_true, y_pred)
    cr   = classification_report(y_true, y_pred, target_names=class_names)

    # Sensitivity = TP / (TP + FN)   [Row 0 = Parasitized]
    # Specificity = TN / (TN + FP)   [Row 1 = Uninfected]
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1] + 1e-9) * 100
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1] + 1e-9) * 100

    metrics = {
        "accuracy":          round(acc, 2),
        "precision":         round(prec, 2),
        "recall":            round(rec, 2),
        "f1_score":          round(f1, 2),
        "sensitivity":       round(sensitivity, 2),
        "specificity":       round(specificity, 2),
        "confusion_matrix":  cm,
        "classification_report": cr,
    }

    log.info(f"Accuracy:    {acc:.2f}%")
    log.info(f"Precision:   {prec:.2f}%")
    log.info(f"Recall:      {rec:.2f}%")
    log.info(f"F1-Score:    {f1:.2f}%")
    log.info(f"Sensitivity: {sensitivity:.2f}%")
    log.info(f"Specificity: {specificity:.2f}%")
    log.info(f"\n{cr}")

    return metrics


def metrics_to_dataframe(metrics: dict) -> pd.DataFrame:
    """Convert scalar metrics to a display-ready DataFrame."""
    rows = [
        ("Accuracy (%)",    metrics["accuracy"]),
        ("Precision (%)",   metrics["precision"]),
        ("Recall (%)",      metrics["recall"]),
        ("F1-Score (%)",    metrics["f1_score"]),
        ("Sensitivity (%)", metrics["sensitivity"]),
        ("Specificity (%)", metrics["specificity"]),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


# ─── Confusion Matrix Plot ─────────────────────────────────────────────────────
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    model_name: str   = "Model",
    save_path: str    = None
) -> plt.Figure:
    """
    Plot confusion matrix as a Seaborn heatmap (document Section 6.5).
    Colours: dark = high count; annotations show exact counts.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=FIG_DPI)

    sns.heatmap(
        cm,
        annot     = True,
        fmt       = "d",
        cmap      = "Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths= 0.5,
        linecolor ="grey",
        ax        = ax,
        cbar_kws  = {"shrink": 0.8}
    )
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=FONT_TITLE, fontweight="bold", pad=12)
    ax.set_ylabel("True Label",      fontsize=FONT_LABEL)
    ax.set_xlabel("Predicted Label", fontsize=FONT_LABEL)
    ax.set_ylim(len(class_names), 0)   # Avoid label clipping
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        log.info(f"Confusion matrix saved: {save_path}")
    return fig


# ─── ROC Curve ────────────────────────────────────────────────────────────────
def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,           # Shape (N, num_classes) – softmax probabilities
    class_names: list,
    model_name: str = "Model",
    save_path: str  = None
) -> plt.Figure:
    """
    Plot ROC curve for each class (One-vs-Rest) and the macro-average.
    """
    from sklearn.preprocessing import label_binarize

    n_classes = len(class_names)
    y_bin     = label_binarize(y_true, classes=list(range(n_classes)))

    colours = [PALETTE, PALETTE2, PALETTE3, "#F4A261", "#8338EC"]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=FIG_DPI)

    roc_auc_vals = []
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        roc_auc_vals.append(roc_auc)
        ax.plot(fpr, tpr, lw=2, color=colours[i % len(colours)],
                label=f"{name}  (AUC = {roc_auc:.3f})")

    # Macro average
    all_fpr  = np.unique(np.concatenate([
        roc_curve(y_bin[:, i], y_prob[:, i])[0] for i in range(n_classes)
    ]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        mean_tpr   += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, "k--", lw=2, label=f"Macro avg  (AUC = {macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], "grey", lw=1, linestyle=":")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=FONT_LABEL)
    ax.set_ylabel("True Positive Rate",  fontsize=FONT_LABEL)
    ax.set_title(f"{model_name} — ROC Curve", fontsize=FONT_TITLE, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        log.info(f"ROC curve saved: {save_path}")
    return fig


# ─── Training History Graphs ──────────────────────────────────────────────────
def plot_training_history(
    history,                   # Keras History object or dict
    model_name: str = "Model",
    save_path: str  = None
) -> plt.Figure:
    """
    Plot accuracy and loss curves for training and validation.
    Matches output shown in document Section 8.5.
    """
    hist = history.history if hasattr(history, "history") else history

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=FIG_DPI)

    # ── Accuracy ────────────────────────────────────────────
    axes[0].plot(hist["accuracy"],     color=PALETTE2, lw=2, label="Train Accuracy")
    axes[0].plot(hist["val_accuracy"], color=PALETTE,  lw=2, linestyle="--", label="Val Accuracy")
    axes[0].set_title(f"{model_name} — Accuracy",  fontsize=FONT_TITLE, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=FONT_LABEL)
    axes[0].set_ylabel("Accuracy", fontsize=FONT_LABEL)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # ── Loss ─────────────────────────────────────────────────
    axes[1].plot(hist["loss"],     color=PALETTE2, lw=2, label="Train Loss")
    axes[1].plot(hist["val_loss"], color=PALETTE,  lw=2, linestyle="--", label="Val Loss")
    axes[1].set_title(f"{model_name} — Loss",      fontsize=FONT_TITLE, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=FONT_LABEL)
    axes[1].set_ylabel("Loss",  fontsize=FONT_LABEL)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        log.info(f"Training history saved: {save_path}")
    return fig


# ─── Feature Map Visualisation ────────────────────────────────────────────────
def visualize_feature_maps(
    model,
    image: np.ndarray,           # Shape (H, W, C)  – normalised
    layer_name: str = None,
    max_filters: int = 16,
    save_path: str = None
) -> plt.Figure:
    """
    Display convolutional feature maps for a single image.
    Useful for explainability (document Section 10: Grad-CAM future work).
    """
    import tensorflow as tf

    # Pick first Conv2D layer if not specified
    if layer_name is None:
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break

    feature_model = tf.keras.Model(
        inputs  = model.input,
        outputs = model.get_layer(layer_name).output
    )

    img_batch  = np.expand_dims(image, 0)
    feature_maps = feature_model.predict(img_batch, verbose=0)[0]  # (H, W, filters)

    n_filters  = min(feature_maps.shape[-1], max_filters)
    cols       = 4
    rows       = (n_filters + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), dpi=FIG_DPI)
    fig.suptitle(f"Feature Maps — Layer: {layer_name}", fontsize=FONT_TITLE, fontweight="bold")

    for i in range(rows * cols):
        ax = axes.flat[i]
        if i < n_filters:
            ax.imshow(feature_maps[:, :, i], cmap="viridis")
            ax.set_title(f"Filter {i+1}", fontsize=9)
        ax.axis("off")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        log.info(f"Feature map saved: {save_path}")
    return fig


# ─── Sample Predictions Grid ──────────────────────────────────────────────────
def plot_sample_predictions(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list,
    n_samples: int = 12,
    save_path: str = None
) -> plt.Figure:
    """
    Grid of sample images with true/predicted labels and confidence scores.
    """
    n_samples = min(n_samples, len(images))
    indices   = np.random.choice(len(images), n_samples, replace=False)

    cols = 4
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5), dpi=FIG_DPI)
    fig.suptitle("Sample Predictions", fontsize=FONT_TITLE, fontweight="bold")

    for plot_idx, data_idx in enumerate(indices):
        ax   = axes.flat[plot_idx]
        img  = images[data_idx]
        true = class_names[y_true[data_idx]]
        pred = class_names[y_pred[data_idx]]
        conf = y_prob[data_idx, y_pred[data_idx]] * 100
        correct = (y_true[data_idx] == y_pred[data_idx])

        ax.imshow(img)
        colour = PALETTE2 if correct else PALETTE
        ax.set_title(
            f"True: {true}\nPred: {pred}  ({conf:.1f}%)",
            fontsize=8, color=colour, fontweight="bold"
        )
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(2)
            spine.set_visible(True)

    # Hide empty subplots
    for i in range(n_samples, rows * cols):
        axes.flat[i].axis("off")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ─── Model Comparison Bar Chart ───────────────────────────────────────────────
def plot_model_comparison(
    results: dict,                 # {"CNN": metrics_dict, "MobileNetV2": metrics_dict}
    save_path: str = None
) -> plt.Figure:
    """
    Side-by-side bar chart comparing multiple models across key metrics.
    """
    metric_keys  = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_names  = list(results.keys())

    x    = np.arange(len(metric_labels))
    width = 0.35
    colours = [PALETTE2, PALETTE, PALETTE3, "#F4A261"]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=FIG_DPI)

    for i, (model_name, metrics) in enumerate(results.items()):
        values = [metrics[k] for k in metric_keys]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name, color=colours[i % len(colours)], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=FONT_LABEL)
    ax.set_ylabel("Score (%)", fontsize=FONT_LABEL)
    ax.set_ylim(0, 115)
    ax.set_title("Model Comparison", fontsize=FONT_TITLE, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
