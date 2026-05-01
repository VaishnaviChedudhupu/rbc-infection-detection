# 🔬 AI-Enhanced Microscopic Image Classification for RBC Infection


---

## 📌 Overview

An end-to-end deep learning system that automatically classifies **Red Blood Cell (RBC) infections** — specifically **malaria (Parasitized)** vs **healthy (Uninfected)** cells — from microscopic blood smear images.

Addresses the real-world problem of time-consuming, error-prone manual microscopic diagnosis by providing:
- **Automated classification** using Convolutional Neural Networks
- **High accuracy** (94.73% CNN vs 68.77% Decision Tree baseline)
- **Real-time prediction** via a modern Streamlit web UI
- **Deployable** in hospitals, rural clinics, and telemedicine platforms

---

## 🏆 Results

| Model | Accuracy | Precision | Recall | F1-Score | Sensitivity | Specificity |
|-------|----------|-----------|--------|----------|-------------|-------------|
| Decision Tree (Baseline) | 68.77% | — | — | — | — | — |
| **Custom CNN** | **94.73%** | **94.74%** | **94.72%** | **94.73%** | **95.47%** | **93.97%** |
| MobileNetV2 (Transfer) | ~97%+ | — | — | — | — | — |

---

## 🗂 Project Structure

```
rbc_project/
│
├── train.py                  # CLI training script (CNN / MobileNetV2 / DTC)
├── predict.py                # CLI prediction script (single image / batch)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── setup.py                  # Package setup
│
├── utils/
│   ├── __init__.py           # Package exports
│   ├── data_pipeline.py      # Dataset loading, preprocessing, augmentation, splitting
│   ├── models.py             # Custom CNN, MobileNetV2, Decision Tree, callbacks
│   └── evaluation.py        # Metrics, confusion matrix, ROC curve, visualisations
│
├── app/
│   └── streamlit_app.py      # Full Streamlit web UI (5 pages)
│
├── dataset/
│   ├── Parasitized/          # Infected RBC images (.png)
│   └── Uninfected/           # Healthy RBC images (.png)
│
├── models/                   # Saved model files (.h5, .pkl)
├── outputs/                  # Generated plots and CSVs
└── test_images/              # Sample images for quick prediction testing
```

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10 |
| Deep Learning | TensorFlow 2.13 / Keras |
| Transfer Learning | MobileNetV2 (ImageNet weights) |
| Image Processing | OpenCV |
| ML Baseline | Scikit-learn (Decision Tree) |
| Data | NumPy, Pandas |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Web UI | Streamlit |
| Dataset | NIH Malaria Cell Images (Kaggle) |

---

## 📦 Dataset

**NIH Malaria Cell Images Dataset**
- **27,558** total cell images
- **13,779** Parasitized (infected)
- **13,779** Uninfected (healthy)
- Balanced classes — no resampling required

**Download:**
```
https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
```

After downloading, extract so that:
```
dataset/
├── Parasitized/   ← infected images
└── Uninfected/    ← healthy images
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/yourusername/rbc-infection-detection.git
cd rbc-infection-detection

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Download dataset

Place Kaggle dataset inside `dataset/` as shown above.

### 3. Train a model

```bash
# Train Custom CNN (recommended for beginners)
python train.py --dataset ./dataset --model cnn --epochs 10

# Train MobileNetV2 transfer learning
python train.py --dataset ./dataset --model mobilenet --epochs 15

# Train all models and compare
python train.py --dataset ./dataset --model both --epochs 10
```

### 4. Launch web UI

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

### 5. Predict on a single image

```bash
# Using saved CNN model
python predict.py --image test_images/cell.png --model models/cnn_model.h5

# Using MobileNetV2
python predict.py --image test_images/cell.png --model models/mobilenet_model.h5 --size 128

# Batch prediction on a folder
python predict.py --folder test_images/ --model models/cnn_model.h5 --batch
```

---

## 🧠 Model Architecture

### Custom CNN (Section 5.5.2 of project document)

```
Input (64×64×3)
    │
    ├── Conv2D(32) → BatchNorm → MaxPool
    ├── Conv2D(32) → BatchNorm → MaxPool
    ├── Conv2D(64) → BatchNorm → MaxPool
    ├── Conv2D(64) → BatchNorm → MaxPool
    │
    ├── Flatten
    ├── Dense(256, ReLU) → Dropout(0.4)
    ├── Dense(128, ReLU) → Dropout(0.2)
    └── Dense(2, Softmax)  → [Parasitized, Uninfected]

Optimizer : Adam
Loss      : Categorical Cross-Entropy
```

### MobileNetV2 Transfer Learning (Future Enhancement from document)

```
Input (128×128×3)
    │
    ├── MobileNetV2 base (ImageNet weights, frozen in Phase 1)
    ├── GlobalAveragePooling2D
    ├── Dense(256, ReLU) → Dropout(0.4)
    ├── Dense(128, ReLU) → Dropout(0.2)
    └── Dense(2, Softmax)

Phase 1: Train top layers only (base frozen)
Phase 2: Fine-tune from layer 100 onwards (very low LR)
```

---

## 🖥 Streamlit UI Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Project overview, metric cards, workflow steps, tech stack |
| 📂 Dataset | Load folders, class distribution chart, sample grid, train/val/test split |
| 🧠 Train | Select model, set hyperparameters, live epoch progress, metrics display |
| 📊 Evaluate | Confusion matrix, ROC curve, model comparison, downloadable PNG exports |
| 🔬 Predict | Upload image → result card (Infected/Healthy) + confidence bars |

---

## 📊 Output Files Generated

After training, the following files are saved:

```
models/
├── cnn_model.h5              # Saved Custom CNN weights
├── mobilenet_model.h5        # Saved MobileNetV2 weights
├── dtc_model.pkl             # Saved Decision Tree
├── cnn_history.pkl           # Training history (CNN)
└── mobilenet_history.pkl     # Training history (MobileNetV2)

outputs/
├── training_history_cnn.png         # Accuracy/Loss curves
├── confusion_matrix_cnn.png         # Confusion matrix heatmap
├── roc_curve_cnn.png                # ROC AUC curve
├── sample_predictions_cnn.png       # Grid of predictions
├── training_history_mobilenet.png
├── confusion_matrix_mobilenet.png
├── roc_curve_mobilenet.png
├── model_comparison.png             # Side-by-side bar chart
└── batch_predictions.csv            # Batch prediction results
```

---

## 🔧 Training CLI Arguments

```
--dataset         Path to dataset root folder (default: ./dataset)
--model           cnn | mobilenet | dtc | both  (default: cnn)
--epochs          Training epochs (default: 10)
--finetune-epochs MobileNetV2 fine-tune epochs (default: 5)
--batch           Batch size (default: 32)
--lr              Learning rate (default: 0.001)
--dropout         Dropout rate (default: 0.4)
--fine-tune-at    MobileNetV2 unfreeze layer index (default: 100)
```

---

## 🔮 Future Enhancements

As described in project document Chapter 10:

- **Grad-CAM** visualisation for explainability
- **SHAP / LIME** for model transparency
- **Multi-class classification** (different malaria stages)
- **ResNet / EfficientNet** transfer learning
- **Mobile app deployment** using TensorFlow Lite
- **Cloud API** with FastAPI / Flask backend

---

## 📚 References

1. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016
2. Simonyan & Zisserman, "Very Deep CNNs for Large-Scale Image Recognition," ICLR 2015
3. LeCun, Bengio & Hinton, "Deep Learning," Nature 2015
4. Rajkomar et al., "Machine Learning in Medicine," NEJM 2019
5. Krizhevsky et al., "ImageNet Classification with Deep CNNs," NeurIPS 2012
6. Litjens et al., "Survey on Deep Learning in Medical Image Analysis," MedIA 2017

---

## 👩‍💻 Author

**C. Vaishnavi**  
Roll No: 22R91A7325  
B.Tech — Artificial Intelligence & Machine Learning  
Teegala Krishna Reddy Engineering College (TKREC), Hyderabad  
Guide: Mrs. G. Mounika, Assistant Professor  

---

*Built as Major Project for partial fulfillment of B.Tech degree, affiliated to JNTUH.*
