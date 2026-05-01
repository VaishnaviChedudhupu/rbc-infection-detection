"""
models.py
─────────
Model definitions for RBC infection classification.

Model A : Custom CNN          (matches architecture in project document)
Model B : MobileNetV2         (transfer learning – future enhancement from document)

Author  : C. Vaishnavi (22R91A7325)
Project : AI-Enhanced Microscopic Image Classification for RBC Infection
College : TKREC, Hyderabad
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import logging

log = logging.getLogger(__name__)


# ─── Model A : Custom CNN ─────────────────────────────────────────────────────
def build_custom_cnn(
    input_shape: tuple = (64, 64, 3),
    num_classes: int   = 2,
    dropout_rate: float = 0.4,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Custom CNN architecture as described in the project document.

    Architecture (Section 5.5.2):
        Input → Conv(32) → MaxPool → Conv(32) → MaxPool →
        Conv(64) → MaxPool → Conv(64) → MaxPool →
        Flatten → Dense(256, ReLU) → Dropout → Dense(num_classes, Softmax)

    Improvements over original:
        - Added L2 regularisation to reduce overfitting
        - Added BatchNormalisation for stable training
        - Added extra conv block for richer feature extraction
    """
    model = models.Sequential(name="Custom_CNN")

    # ── Block 1 ──────────────────────────────────────────────
    model.add(layers.Conv2D(
        32, (3, 3), activation="relu",
        input_shape=input_shape,
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # ── Block 2 ──────────────────────────────────────────────
    model.add(layers.Conv2D(
        32, (3, 3), activation="relu",
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # ── Block 3 ──────────────────────────────────────────────
    model.add(layers.Conv2D(
        64, (3, 3), activation="relu",
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # ── Block 4 ──────────────────────────────────────────────
    model.add(layers.Conv2D(
        64, (3, 3), activation="relu",
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # ── Classifier Head ──────────────────────────────────────
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(dropout_rate / 2))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    log.info("Custom CNN built successfully.")
    model.summary(print_fn=log.info)
    return model


# ─── Model B : MobileNetV2 Transfer Learning ──────────────────────────────────
def build_mobilenet_v2(
    input_shape: tuple  = (128, 128, 3),
    num_classes: int    = 2,
    dropout_rate: float = 0.4,
    learning_rate: float = 0.0001,
    fine_tune_at: int   = 100,           # Unfreeze layers from this index onwards
) -> tf.keras.Model:
    """
    MobileNetV2 transfer learning model.
    Referenced in document Section 10 (Future Enhancements) as ResNet/VGG16/EfficientNet.
    MobileNetV2 chosen for its balance of accuracy and lightweight deployment.

    Strategy:
        Phase 1 – Train only the top layers (base frozen)
        Phase 2 – Fine-tune from `fine_tune_at` layer onwards
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,          # Remove ImageNet classifier
        weights="imagenet"          # Pre-trained weights
    )

    # Phase 1: freeze the whole base
    base_model.trainable = False

    inputs  = tf.keras.Input(shape=input_shape)
    x       = base_model(inputs, training=False)   # Inference mode for BN layers
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(dropout_rate)(x)
    x       = layers.Dense(128, activation="relu")(x)
    x       = layers.Dropout(dropout_rate / 2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="MobileNetV2_Transfer")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    log.info(f"MobileNetV2 built (base frozen). Fine-tune from layer {fine_tune_at}.")
    return model, base_model


def unfreeze_mobilenet(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    fine_tune_at: int   = 100,
    learning_rate: float = 1e-5
) -> tf.keras.Model:
    """
    Unfreeze upper layers of MobileNetV2 for fine-tuning (Phase 2).
    Use a very low learning rate to avoid destroying pre-trained weights.
    """
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    log.info(f"MobileNetV2 unfrozen from layer {fine_tune_at}. Fine-tuning enabled.")
    return model


# ─── Baseline: Decision Tree (sklearn) ────────────────────────────────────────
def build_decision_tree(max_depth: int = 3, random_state: int = 42):
    """
    Decision Tree Classifier – baseline model from project document (Section 5.5.1).
    Operates on flattened pixel features.
    """
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    log.info(f"DecisionTreeClassifier built (max_depth={max_depth})")
    return dtc


# ─── Callbacks ────────────────────────────────────────────────────────────────
def get_training_callbacks(
    model_save_path: str,
    patience_early_stop: int = 10,
    patience_lr: int         = 5,
    min_lr: float            = 1e-7
) -> list:
    """
    Standard training callbacks:
    - ModelCheckpoint  : save best weights
    - EarlyStopping    : stop when val_loss stops improving
    - ReduceLROnPlateau: lower LR when plateau detected
    """
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath     = model_save_path,
            monitor      = "val_accuracy",
            save_best_only=True,
            verbose      = 1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor      = "val_loss",
            patience     = patience_early_stop,
            restore_best_weights=True,
            verbose      = 1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor      = "val_loss",
            factor       = 0.5,
            patience     = patience_lr,
            min_lr       = min_lr,
            verbose      = 1
        ),
    ]
    return callbacks
