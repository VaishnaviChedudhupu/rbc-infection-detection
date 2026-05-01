"""
predict.py
──────────
Standalone prediction script for RBC infection detection.
Run from command line:
    python predict.py --image path/to/image.png --model models/cnn_model.h5
    python predict.py --image path/to/image.png --model models/mobilenet_model.h5 --size 128

Author  : C. Vaishnavi (22R91A7325)
Project : AI-Enhanced Microscopic Image Classification for RBC Infection
College : TKREC, Hyderabad
"""

import argparse
import os
import sys
import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES     = ["Parasitized", "Uninfected"]
DEFAULT_SIZE    = 64       # Custom CNN default
MOBILENET_SIZE  = 128      # MobileNetV2 default
CONFIDENCE_HIGH = 90.0     # Threshold for high-confidence label
CONFIDENCE_MED  = 70.0     # Threshold for medium-confidence label


# ─── Image Preprocessor ───────────────────────────────────────────────────────
def preprocess_image(image_path: str, img_size: int) -> tuple:
    """
    Load, resize, and normalise a single image for inference.

    Returns
    -------
    img_array : np.ndarray  shape (1, img_size, img_size, 3)
    img_rgb   : np.ndarray  shape (img_size, img_size, 3)  for display
    """
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[ERROR] Could not read image: {image_path}")
        sys.exit(1)

    img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_norm    = img_resized.astype(np.float32) / 255.0
    img_batch   = np.expand_dims(img_norm, axis=0)   # (1, H, W, C)

    return img_batch, img_resized


# ─── Model Loader ─────────────────────────────────────────────────────────────
def load_model(model_path: str) -> tf.keras.Model:
    """Load a saved Keras model (.h5 or SavedModel format)."""
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print("  Train a model first using:  python train.py")
        sys.exit(1)

    print(f"[INFO]  Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"[INFO]  Model loaded successfully.")
    return model


# ─── Confidence Label ─────────────────────────────────────────────────────────
def confidence_label(confidence: float) -> str:
    """Return a human-readable confidence tier."""
    if confidence >= CONFIDENCE_HIGH:
        return "HIGH confidence"
    elif confidence >= CONFIDENCE_MED:
        return "MEDIUM confidence"
    else:
        return "LOW confidence — result may be unreliable"


# ─── Core Prediction ──────────────────────────────────────────────────────────
def predict(
    image_path : str,
    model_path : str,
    img_size   : int  = DEFAULT_SIZE,
    save_result: bool = True,
    output_dir : str  = "outputs"
) -> dict:
    """
    Run inference on a single image and return prediction details.

    Returns
    -------
    result : dict with keys:
        image_path, predicted_class, confidence,
        probabilities, confidence_tier
    """
    # ── Load & preprocess ────────────────────────────────────
    img_batch, img_display = preprocess_image(image_path, img_size)

    # ── Load model ───────────────────────────────────────────
    model = load_model(model_path)

    # ── Inference ────────────────────────────────────────────
    probabilities = model.predict(img_batch, verbose=0)[0]   # shape (num_classes,)
    pred_idx      = int(np.argmax(probabilities))
    pred_class    = CLASS_NAMES[pred_idx]
    confidence    = float(probabilities[pred_idx]) * 100
    conf_tier     = confidence_label(confidence)

    # ── Console output ───────────────────────────────────────
    print("\n" + "=" * 50)
    print("  RBC INFECTION DETECTION — RESULT")
    print("=" * 50)
    print(f"  Image     : {os.path.basename(image_path)}")
    print(f"  Prediction: {pred_class}")
    print(f"  Confidence: {confidence:.2f}%  ({conf_tier})")
    print("-" * 50)
    for i, name in enumerate(CLASS_NAMES):
        bar_len = int(probabilities[i] * 30)
        bar     = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {name:<14} [{bar}]  {probabilities[i]*100:.2f}%")
    print("=" * 50)

    if pred_class == "Parasitized":
        print("\n  ⚠  INFECTED RBC DETECTED.")
        print("     Please consult a medical professional immediately.")
    else:
        print("\n  ✔  No infection detected. RBC appears healthy.")
    print()

    # ── Save result image ────────────────────────────────────
    if save_result:
        result_fig = _render_result_image(
            img_display, pred_class, confidence, probabilities
        )
        os.makedirs(output_dir, exist_ok=True)
        base_name   = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"prediction_{base_name}.png")
        result_fig.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(result_fig)
        print(f"[INFO]  Result image saved: {output_path}")

    return {
        "image_path":       image_path,
        "predicted_class":  pred_class,
        "confidence":       round(confidence, 2),
        "probabilities":    {CLASS_NAMES[i]: round(float(probabilities[i]) * 100, 2)
                             for i in range(len(CLASS_NAMES))},
        "confidence_tier":  conf_tier,
    }


# ─── Result Image Renderer ────────────────────────────────────────────────────
def _render_result_image(
    img: np.ndarray,
    pred_class: str,
    confidence: float,
    probabilities: np.ndarray
) -> plt.Figure:
    """
    Render a result card:
        left  — input microscopic image with overlay label
        right — probability bar chart
    """
    is_infected = (pred_class == "Parasitized")
    accent_col  = "#E63946" if is_infected else "#2A9D8F"

    fig = plt.figure(figsize=(10, 4), facecolor="#1A1A2E")
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.35)

    # ── Left: image ──────────────────────────────────────────
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(img)
    ax_img.set_title(
        f"{'⚠  INFECTED' if is_infected else '✔  UNINFECTED'}",
        fontsize=14, fontweight="bold", color=accent_col, pad=8
    )
    ax_img.text(
        0.5, -0.05,
        f"Confidence: {confidence:.1f}%",
        transform=ax_img.transAxes,
        ha="center", fontsize=11, color="white"
    )
    ax_img.axis("off")
    for spine in ax_img.spines.values():
        spine.set_edgecolor(accent_col)
        spine.set_linewidth(3)
        spine.set_visible(True)

    # ── Right: probability bars ───────────────────────────────
    ax_bar = fig.add_subplot(gs[1])
    ax_bar.set_facecolor("#1A1A2E")

    colours = ["#E63946", "#2A9D8F"]
    bars    = ax_bar.barh(
        CLASS_NAMES,
        [p * 100 for p in probabilities],
        color=colours, edgecolor="white", linewidth=0.5, height=0.5
    )
    for bar, prob in zip(bars, probabilities):
        ax_bar.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{prob * 100:.2f}%",
            va="center", ha="left", color="white", fontsize=12, fontweight="bold"
        )

    ax_bar.set_xlim(0, 115)
    ax_bar.set_xlabel("Probability (%)", color="white", fontsize=11)
    ax_bar.set_title("Class Probabilities", color="white", fontsize=13, fontweight="bold")
    ax_bar.tick_params(colors="white", labelsize=11)
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#444466")
    ax_bar.xaxis.label.set_color("white")

    fig.suptitle(
        "RBC Infection Detection — Prediction Result",
        fontsize=14, fontweight="bold", color="white", y=1.02
    )
    return fig


# ─── Batch Prediction ─────────────────────────────────────────────────────────
def predict_batch(
    image_dir  : str,
    model_path : str,
    img_size   : int = DEFAULT_SIZE,
    output_dir : str = "outputs"
) -> list:
    """
    Run prediction on all images inside a folder.
    Returns list of result dicts.
    """
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images    = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in supported
    ]

    if not images:
        print(f"[ERROR] No valid images found in: {image_dir}")
        sys.exit(1)

    print(f"\n[INFO]  Running batch prediction on {len(images)} images...\n")

    model = load_model(model_path)
    results = []

    for img_path in images:
        img_batch, img_display = preprocess_image(img_path, img_size)
        probs      = model.predict(img_batch, verbose=0)[0]
        pred_idx   = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx]) * 100

        results.append({
            "image":            os.path.basename(img_path),
            "predicted_class":  pred_class,
            "confidence":       round(confidence, 2),
            "Parasitized_%":    round(float(probs[0]) * 100, 2),
            "Uninfected_%":     round(float(probs[1]) * 100, 2),
        })
        status = "⚠ INFECTED" if pred_class == "Parasitized" else "✔ HEALTHY"
        print(f"  {status:14}  {os.path.basename(img_path):<40}  {confidence:.1f}%")

    # ── Save summary CSV ─────────────────────────────────────
    import pandas as pd
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "batch_predictions.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\n[INFO]  Batch results saved: {csv_path}")

    infected_count   = sum(1 for r in results if r["predicted_class"] == "Parasitized")
    uninfected_count = len(results) - infected_count
    print(f"\n  Summary: {infected_count} Infected  |  {uninfected_count} Uninfected  |  Total: {len(results)}")

    return results


# ─── CLI Entry Point ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="RBC Infection Detection — Prediction Script",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Single image :\n"
            "    python predict.py --image test_images/cell.png --model models/cnn_model.h5\n\n"
            "  MobileNetV2  :\n"
            "    python predict.py --image test_images/cell.png --model models/mobilenet_model.h5 --size 128\n\n"
            "  Batch folder :\n"
            "    python predict.py --folder test_images/ --model models/cnn_model.h5 --batch\n"
        )
    )
    parser.add_argument("--image",   type=str, help="Path to a single input image")
    parser.add_argument("--folder",  type=str, help="Path to folder for batch prediction")
    parser.add_argument("--model",   type=str, required=True,
                        help="Path to trained model (.h5 file)")
    parser.add_argument("--size",    type=int, default=DEFAULT_SIZE,
                        help=f"Input image size (default: {DEFAULT_SIZE}; use 128 for MobileNetV2)")
    parser.add_argument("--batch",   action="store_true",
                        help="Enable batch prediction mode (use with --folder)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save result image to outputs/")
    parser.add_argument("--output",  type=str, default="outputs",
                        help="Directory to save output files (default: outputs/)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.batch and args.folder:
        predict_batch(
            image_dir  = args.folder,
            model_path = args.model,
            img_size   = args.size,
            output_dir = args.output
        )
    elif args.image:
        predict(
            image_path  = args.image,
            model_path  = args.model,
            img_size    = args.size,
            save_result = not args.no_save,
            output_dir  = args.output
        )
    else:
        print("[ERROR] Provide --image <path> for single prediction")
        print("        or --folder <path> --batch for batch prediction")
        print("        Run with --help for full usage.")
        sys.exit(1)
