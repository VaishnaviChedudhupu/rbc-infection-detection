"""
train.py
────────
Main training script for RBC Infection Detection.
Trains Custom CNN and/or MobileNetV2 from command line.

Usage examples:
    # Train Custom CNN (default)
    python train.py --dataset ./dataset --model cnn

    # Train MobileNetV2
    python train.py --dataset ./dataset --model mobilenet --epochs 20

    # Train both and compare
    python train.py --dataset ./dataset --model both --epochs 15 --batch 32

    # Train with custom hyperparameters
    python train.py --dataset ./dataset --model cnn --epochs 30 --batch 16 --lr 0.0005

Author  : C. Vaishnavi (22R91A7325)
Project : AI-Enhanced Microscopic Image Classification for RBC Infection
College : TKREC, Hyderabad
"""

import os
import sys
import argparse
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_pipeline import (
    load_dataset, split_dataset, compute_class_weights,
    dataset_summary, IMG_SIZE, IMG_SIZE_CNN
)
from utils.models import (
    build_custom_cnn, build_mobilenet_v2, unfreeze_mobilenet,
    build_decision_tree, get_training_callbacks
)
from utils.evaluation import (
    compute_metrics, plot_confusion_matrix, plot_roc_curve,
    plot_training_history, plot_sample_predictions, plot_model_comparison
)

MODELS_DIR  = "models"
OUTPUTS_DIR = "outputs"


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _banner(text: str):
    line = "─" * 60
    print(f"\n{line}\n  {text}\n{line}")


def _save_figure(fig, filename: str):
    import matplotlib.pyplot as plt
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(OUTPUTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [saved] {path}")


def _resize_batch(X: np.ndarray, size: int) -> np.ndarray:
    import cv2
    return np.array([cv2.resize(img, (size, size)) for img in X])


def _print_metrics(metrics: dict, model_name: str):
    print(f"\n  {'Metric':<18} {'Value':>10}")
    print(f"  {'──────':<18} {'─────':>10}")
    for key, label in [
        ("accuracy",    "Accuracy"),
        ("precision",   "Precision"),
        ("recall",      "Recall"),
        ("f1_score",    "F1-Score"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
    ]:
        print(f"  {label:<18} {metrics[key]:>9.2f}%")
    print()


# ─── Decision Tree ────────────────────────────────────────────────────────────
def train_decision_tree(splits: dict, classes: list) -> dict:
    _banner("Training Decision Tree Classifier (Baseline)")

    X_train = splits["X_train"].reshape(len(splits["X_train"]), -1)
    X_test  = splits["X_test"].reshape(len(splits["X_test"]),  -1)

    dtc = build_decision_tree(max_depth=3)
    print("  Fitting Decision Tree…")
    dtc.fit(X_train, splits["y_train"])

    y_pred  = dtc.predict(X_test)
    metrics = compute_metrics(splits["y_test"], y_pred, classes)
    _print_metrics(metrics, "Decision Tree")

    import joblib
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "dtc_model.pkl")
    joblib.dump(dtc, model_path)
    print(f"  [saved] {model_path}")

    _save_figure(
        plot_confusion_matrix(metrics["confusion_matrix"], classes, "Decision Tree"),
        "confusion_matrix_dtc.png"
    )
    return metrics


# ─── Custom CNN ───────────────────────────────────────────────────────────────
def train_custom_cnn(
    splits: dict, classes: list,
    epochs=10, batch_size=32, lr=0.001, dropout=0.4
) -> tuple:
    _banner("Training Custom CNN")

    from tensorflow.keras.utils import to_categorical
    from utils.data_pipeline import build_augmentation_generator

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val,   y_val   = splits["X_val"],   splits["y_val"]
    X_test,  y_test  = splits["X_test"],  splits["y_test"]
    n_cls            = len(classes)

    y_tr_cat = to_categorical(y_train, n_cls)
    y_va_cat = to_categorical(y_val,   n_cls)

    model = build_custom_cnn(X_train.shape[1:], n_cls, dropout, lr)
    model.summary()

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "cnn_model.h5")
    callbacks = get_training_callbacks(save_path)
    aug       = build_augmentation_generator()
    cw        = compute_class_weights(y_train)

    print(f"\n  Epochs: {epochs}  Batch: {batch_size}  LR: {lr}  Dropout: {dropout}")
    print(f"  Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}\n")

    history = model.fit(
        aug.flow(X_train, y_tr_cat, batch_size=batch_size),
        validation_data=(X_val, y_va_cat),
        epochs=epochs,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1
    )

    model.load_weights(save_path)

    y_prob  = model.predict(X_test, verbose=0)
    y_pred  = np.argmax(y_prob, axis=1)
    metrics = compute_metrics(y_test, y_pred, classes)
    metrics.update({"y_true": y_test, "y_pred": y_pred, "y_prob": y_prob})
    _print_metrics(metrics, "Custom CNN")

    # Save history
    hist_path = os.path.join(MODELS_DIR, "cnn_history.pkl")
    with open(hist_path, "wb") as f:
        pickle.dump(history.history, f)
    print(f"  [saved] {hist_path}")

    # All output plots
    _save_figure(plot_training_history(history, "Custom CNN"),          "training_history_cnn.png")
    _save_figure(plot_confusion_matrix(metrics["confusion_matrix"],
                                       classes, "Custom CNN"),          "confusion_matrix_cnn.png")
    _save_figure(plot_roc_curve(y_test, y_prob, classes, "Custom CNN"), "roc_curve_cnn.png")
    _save_figure(plot_sample_predictions(X_test, y_test, y_pred,
                                         y_prob, classes, 12),          "sample_predictions_cnn.png")

    return model, metrics, history.history


# ─── MobileNetV2 ──────────────────────────────────────────────────────────────
def train_mobilenet(
    splits: dict, classes: list,
    epochs=10, finetune_epochs=5,
    batch_size=32, lr=0.0001,
    dropout=0.4, fine_tune_at=100
) -> tuple:
    _banner("Training MobileNetV2 Transfer Learning")

    from tensorflow.keras.utils import to_categorical

    print("  Resizing images to 128×128 for MobileNetV2…")
    X_train = _resize_batch(splits["X_train"], 128)
    X_val   = _resize_batch(splits["X_val"],   128)
    X_test  = _resize_batch(splits["X_test"],  128)
    y_train, y_val, y_test = splits["y_train"], splits["y_val"], splits["y_test"]
    n_cls   = len(classes)

    y_tr_cat = to_categorical(y_train, n_cls)
    y_va_cat = to_categorical(y_val,   n_cls)

    model, base_model = build_mobilenet_v2(
        (128, 128, 3), n_cls, dropout, lr, fine_tune_at
    )
    model.summary()

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "mobilenet_model.h5")
    callbacks = get_training_callbacks(save_path)
    cw        = compute_class_weights(y_train)

    # Phase 1
    print(f"\n  Phase 1 — Frozen base ({epochs} epochs)…\n")
    hist1 = model.fit(
        X_train, y_tr_cat,
        validation_data=(X_val, y_va_cat),
        epochs=epochs, batch_size=batch_size,
        class_weight=cw, callbacks=callbacks, verbose=1
    )

    # Phase 2
    combined = dict(hist1.history)
    if finetune_epochs > 0:
        print(f"\n  Phase 2 — Fine-tuning from layer {fine_tune_at} ({finetune_epochs} epochs)…\n")
        model = unfreeze_mobilenet(model, base_model, fine_tune_at, lr / 10)
        hist2 = model.fit(
            X_train, y_tr_cat,
            validation_data=(X_val, y_va_cat),
            epochs=finetune_epochs, batch_size=batch_size,
            class_weight=cw, callbacks=callbacks, verbose=1
        )
        for k in combined:
            combined[k] = combined[k] + hist2.history.get(k, [])

    model.load_weights(save_path)

    y_prob  = model.predict(X_test, verbose=0)
    y_pred  = np.argmax(y_prob, axis=1)
    metrics = compute_metrics(y_test, y_pred, classes)
    metrics.update({"y_true": y_test, "y_pred": y_pred, "y_prob": y_prob})
    _print_metrics(metrics, "MobileNetV2")

    hist_path = os.path.join(MODELS_DIR, "mobilenet_history.pkl")
    with open(hist_path, "wb") as f:
        pickle.dump(combined, f)
    print(f"  [saved] {hist_path}")

    _save_figure(plot_training_history(combined, "MobileNetV2"),            "training_history_mobilenet.png")
    _save_figure(plot_confusion_matrix(metrics["confusion_matrix"],
                                       classes, "MobileNetV2"),             "confusion_matrix_mobilenet.png")
    _save_figure(plot_roc_curve(y_test, y_prob, classes, "MobileNetV2"),    "roc_curve_mobilenet.png")
    _save_figure(plot_sample_predictions(X_test, y_test, y_pred,
                                         y_prob, classes, 12),              "sample_predictions_mobilenet.png")

    return model, metrics, combined


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(args):
    gpus = tf.config.list_physical_devices("GPU")
    device_msg = f"{len(gpus)} GPU(s)" if gpus else "CPU only"
    print(f"\n  Device: {device_msg}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    _banner("Loading Dataset")
    img_size = IMG_SIZE_CNN if args.model == "mobilenet" else IMG_SIZE
    X, y, classes, _ = load_dataset(args.dataset, img_size)
    print(f"\n{dataset_summary(y, classes).to_string(index=False)}\n")

    _banner("Splitting Dataset")
    splits = split_dataset(X, y)

    all_results = {}

    if args.model in ("dtc", "both"):
        all_results["Decision Tree"] = train_decision_tree(splits, classes)

    if args.model in ("cnn", "both"):
        _, m, _ = train_custom_cnn(splits, classes, args.epochs,
                                   args.batch, args.lr, args.dropout)
        all_results["Custom CNN"] = m

    if args.model in ("mobilenet", "both"):
        _, m, _ = train_mobilenet(splits, classes, args.epochs,
                                  args.finetune_epochs, args.batch,
                                  args.lr, args.dropout, args.fine_tune_at)
        all_results["MobileNetV2"] = m

    if len(all_results) > 1:
        _banner("Model Comparison")
        _save_figure(plot_model_comparison(all_results), "model_comparison.png")
        print(f"\n  {'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'─────':<22} {'────────':>10} {'─────────':>10} {'──────':>10} {'──':>10}")
        for name, m in all_results.items():
            print(f"  {name:<22} {m['accuracy']:>9.2f}%"
                  f" {m['precision']:>9.2f}%"
                  f" {m['recall']:>9.2f}%"
                  f" {m['f1_score']:>9.2f}%")

    _banner("Training Complete")
    print(f"  Models  → {MODELS_DIR}/")
    print(f"  Outputs → {OUTPUTS_DIR}/\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="RBC Infection Detection — Training Script",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train.py --dataset ./dataset --model cnn\n"
            "  python train.py --dataset ./dataset --model mobilenet --epochs 20\n"
            "  python train.py --dataset ./dataset --model both --epochs 15\n"
        )
    )
    p.add_argument("--dataset",        type=str,   default="./dataset",
                   help="Dataset root folder (must have Parasitized/ & Uninfected/ subfolders)")
    p.add_argument("--model",          type=str,   default="cnn",
                   choices=["cnn", "mobilenet", "dtc", "both"],
                   help="Model to train: cnn | mobilenet | dtc | both  (default: cnn)")
    p.add_argument("--epochs",         type=int,   default=10)
    p.add_argument("--finetune-epochs",type=int,   default=5,   dest="finetune_epochs",
                   help="MobileNetV2 fine-tune epochs (default: 5)")
    p.add_argument("--batch",          type=int,   default=32)
    p.add_argument("--lr",             type=float, default=0.001)
    p.add_argument("--dropout",        type=float, default=0.4)
    p.add_argument("--fine-tune-at",   type=int,   default=100, dest="fine_tune_at",
                   help="MobileNetV2 unfreeze from this layer index (default: 100)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
