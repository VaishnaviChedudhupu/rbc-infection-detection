"""
utils/__init__.py
─────────────────
Utility package for RBC Infection Detection project.

Exposes core modules:
    - data_pipeline  : dataset loading, preprocessing, augmentation, splitting
    - models         : Custom CNN, MobileNetV2, Decision Tree, callbacks
    - evaluation     : metrics, confusion matrix, ROC curve, training graphs

Author  : C. Vaishnavi (22R91A7325)
Project : AI-Enhanced Microscopic Image Classification for RBC Infection
College : TKREC, Hyderabad
"""

from utils.data_pipeline import (
    load_dataset,
    split_dataset,
    compute_class_weights,
    build_augmentation_generator,
    build_val_test_generator,
    get_generators,
    preprocess_single_image,
    dataset_summary,
    IMG_SIZE,
    IMG_SIZE_CNN,
    CLASS_NAMES,
)

from utils.models import (
    build_custom_cnn,
    build_mobilenet_v2,
    unfreeze_mobilenet,
    build_decision_tree,
    get_training_callbacks,
)

from utils.evaluation import (
    compute_metrics,
    metrics_to_dataframe,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_history,
    visualize_feature_maps,
    plot_sample_predictions,
    plot_model_comparison,
)

__all__ = [
    # data_pipeline
    "load_dataset",
    "split_dataset",
    "compute_class_weights",
    "build_augmentation_generator",
    "build_val_test_generator",
    "get_generators",
    "preprocess_single_image",
    "dataset_summary",
    "IMG_SIZE",
    "IMG_SIZE_CNN",
    "CLASS_NAMES",
    # models
    "build_custom_cnn",
    "build_mobilenet_v2",
    "unfreeze_mobilenet",
    "build_decision_tree",
    "get_training_callbacks",
    # evaluation
    "compute_metrics",
    "metrics_to_dataframe",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_training_history",
    "visualize_feature_maps",
    "plot_sample_predictions",
    "plot_model_comparison",
]
