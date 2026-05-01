"""
data_pipeline.py
────────────────
Complete data pipeline for RBC infection detection.
Handles dataset loading, cleaning, preprocessing, augmentation, and splitting.

Author  : C. Vaishnavi (22R91A7325)
Project : AI-Enhanced Microscopic Image Classification for RBC Infection
College : TKREC, Hyderabad
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE      = (64, 64)          # Resize target (matches original project)
IMG_SIZE_CNN  = (128, 128)        # Larger size for MobileNetV2 transfer learning
CHANNELS      = 3                 # RGB
RANDOM_STATE  = 42
TEST_SPLIT    = 0.20              # 80 / 20 split (matches document)
VAL_SPLIT     = 0.15             # 15% of training for validation
CLASS_NAMES   = ["Parasitized", "Uninfected"]


# ─── Image Validation ─────────────────────────────────────────────────────────
def is_valid_image(filepath: str) -> bool:
    """Check if a file is a readable, non-corrupt image."""
    try:
        img = cv2.imread(filepath)
        if img is None or img.size == 0:
            return False
        return True
    except Exception:
        return False


# ─── Dataset Loader ───────────────────────────────────────────────────────────
def load_dataset(
    dataset_path: str,
    img_size: tuple = IMG_SIZE,
    progress_callback=None
) -> tuple[np.ndarray, np.ndarray, list, dict]:
    """
    Load all images from <dataset_path>/Parasitized and <dataset_path>/Uninfected folders.

    Returns
    -------
    X        : np.ndarray  shape (N, H, W, C)  – normalised float32 [0, 1]
    y        : np.ndarray  shape (N,)           – integer labels 0/1
    classes  : list of str                      – class names
    stats    : dict                             – loading statistics
    """
    dataset_path = Path(dataset_path)
    classes = [
        d for d in sorted(os.listdir(dataset_path))
        if (dataset_path / d).is_dir()
    ]
    if not classes:
        raise ValueError(f"No class sub-folders found inside: {dataset_path}")

    log.info(f"Classes detected: {classes}")

    X, y = [], []
    stats = {"total": 0, "loaded": 0, "skipped": 0, "per_class": {}}

    for label_idx, class_name in enumerate(classes):
        class_dir   = dataset_path / class_name
        image_files = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        ]
        stats["per_class"][class_name] = {"found": len(image_files), "loaded": 0}

        for i, img_path in enumerate(image_files):
            stats["total"] += 1

            if not is_valid_image(str(img_path)):
                log.warning(f"Skipping corrupt image: {img_path.name}")
                stats["skipped"] += 1
                continue

            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # BGR → RGB
            img = cv2.resize(img, img_size)                      # Resize
            img = img.astype(np.float32) / 255.0                 # Normalise [0,1]

            X.append(img)
            y.append(label_idx)

            stats["loaded"]  += 1
            stats["per_class"][class_name]["loaded"] += 1

            if progress_callback and i % 100 == 0:
                progress_callback(class_name, i, len(image_files))

        log.info(f"  [{class_name}] loaded {stats['per_class'][class_name]['loaded']} images")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    log.info(f"Dataset  total={stats['total']}  loaded={stats['loaded']}  skipped={stats['skipped']}")
    log.info(f"X shape: {X.shape}   y shape: {y.shape}")
    return X, y, classes, stats


# ─── Train / Val / Test Split ─────────────────────────────────────────────────
def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SPLIT,
    val_size: float  = VAL_SPLIT,
    random_state: int = RANDOM_STATE
) -> dict:
    """
    Split into train / validation / test subsets.
    Stratified split ensures equal class representation.
    """
    # First split off the test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Split remaining into train + val
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio, stratify=y_train_val, random_state=random_state
    )

    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
    }

    log.info(f"Split  train={X_train.shape[0]}  val={X_val.shape[0]}  test={X_test.shape[0]}")
    return splits


# ─── Class Weight Computation ─────────────────────────────────────────────────
def compute_class_weights(y_train: np.ndarray) -> dict:
    """Return class weights to handle imbalanced datasets."""
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    weight_dict = {i: w for i, w in enumerate(weights)}
    log.info(f"Class weights: {weight_dict}")
    return weight_dict


# ─── Data Augmentation Generators ─────────────────────────────────────────────
def build_augmentation_generator(
    rotation_range: int    = 20,
    zoom_range: float      = 0.15,
    width_shift: float     = 0.10,
    height_shift: float    = 0.10,
    horizontal_flip: bool  = True,
    vertical_flip: bool    = True,
    brightness_range: tuple = (0.8, 1.2),
    fill_mode: str          = "nearest"
) -> ImageDataGenerator:
    """
    Build an augmentation pipeline (training only).
    Techniques from document: rotation, flipping, zooming, brightness adjustment.
    """
    return ImageDataGenerator(
        rotation_range     = rotation_range,
        zoom_range         = zoom_range,
        width_shift_range  = width_shift,
        height_shift_range = height_shift,
        horizontal_flip    = horizontal_flip,
        vertical_flip      = vertical_flip,
        brightness_range   = brightness_range,
        fill_mode          = fill_mode
    )


def build_val_test_generator() -> ImageDataGenerator:
    """No augmentation for validation / test (only rescaling if needed)."""
    return ImageDataGenerator()


# ─── Batch Generators ─────────────────────────────────────────────────────────
def get_generators(splits: dict, batch_size: int = 32) -> tuple:
    """
    Wrap numpy arrays in Keras ImageDataGenerators.
    Returns (train_gen, val_gen, test_gen).
    """
    aug_gen = build_augmentation_generator()
    val_gen = build_val_test_generator()

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val   = splits["X_val"]
    y_val   = splits["y_val"]
    X_test  = splits["X_test"]
    y_test  = splits["y_test"]

    # Convert labels to categorical
    from tensorflow.keras.utils import to_categorical
    num_classes = len(np.unique(y_train))

    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat   = to_categorical(y_val,   num_classes)
    y_test_cat  = to_categorical(y_test,  num_classes)

    train_generator = aug_gen.flow(X_train, y_train_cat, batch_size=batch_size, shuffle=True)
    val_generator   = val_gen.flow(X_val,   y_val_cat,   batch_size=batch_size, shuffle=False)
    test_generator  = val_gen.flow(X_test,  y_test_cat,  batch_size=batch_size, shuffle=False)

    return train_generator, val_generator, test_generator


# ─── Single Image Preprocessor ────────────────────────────────────────────────
def preprocess_single_image(
    image_path: str,
    img_size: tuple = IMG_SIZE
) -> np.ndarray:
    """
    Preprocess one image for inference.
    Returns numpy array of shape (1, H, W, C).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)   # shape (1, H, W, C)


# ─── Dataset Statistics ───────────────────────────────────────────────────────
def dataset_summary(y: np.ndarray, classes: list) -> pd.DataFrame:
    """Return a DataFrame summarising class distribution."""
    unique, counts = np.unique(y, return_counts=True)
    df = pd.DataFrame({
        "Class":      [classes[i] for i in unique],
        "Count":      counts,
        "Percentage": (counts / len(y) * 100).round(2)
    })
    return df


# ─── Quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Adjust path to your local dataset
    DATA_PATH = "./dataset"
    if os.path.exists(DATA_PATH):
        X, y, classes, stats = load_dataset(DATA_PATH)
        splits = split_dataset(X, y)
        print(dataset_summary(y, classes).to_string(index=False))
    else:
        print("Dataset not found. Place images in ./dataset/Parasitized and ./dataset/Uninfected")
