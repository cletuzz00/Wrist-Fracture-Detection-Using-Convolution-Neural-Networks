# Wrist Fracture Detection using Deep Learning on the MURA Dataset

This project implements a deep learning-based approach for automatic detection of wrist fractures from X-ray images using Convolutional Neural Networks (CNNs). The dataset used is a subset (wrist-only) from the **MURA (Musculoskeletal Radiographs)** dataset.

## 🧠 Project Overview

Musculoskeletal injuries are common, and X-ray interpretation remains time-consuming and subjective. This project aims to assist radiologists in identifying wrist fractures using state-of-the-art deep learning models trained on real-world data.

Key components:
- Wrist-only data filtering from the MURA dataset.
- Preprocessing and augmentation for model generalization.
- Transfer learning with models such as ResNet, EfficientNet, etc.
- Performance evaluation using metrics such as Accuracy, ROC AUC, Sensitivity, and Specificity.
- Visualization of predictions and model interpretability.

## 🗂️ Dataset

- **Source**: [MURA Dataset by Stanford ML Group](https://stanfordmlgroup.github.io/competitions/mura/)
- **Original Source**: [MURA Dataset by Stanford ML Group](https://stanfordmlgroup.github.io/competitions/mura/)
- **Content**: 40,561 musculoskeletal studies from 14,863 patients.
- **Subset Used**: Only wrist images.
- **Classes**: 
  - `Normal`
  - `Fracture`

## 🧰 Dependencies

This project uses Python 3.x and the following main libraries:

```bash
tensorflow
opencv-python
matplotlib
numpy
pandas
seaborn
scikit-learn
```

To install the dependencies:

```bash
pip install -r requirements.txt
```

## 📁 Folder Structure

```
├── model_final.ipynb       # Main training and evaluation notebook
├── better-mura/            # MURA dataset (wrist images only)
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

## 🔄 Data Preparation

A function `get_data()` is implemented to:
- Read the dataset metadata.
- Filter wrist X-ray images.
- Load and resize images.
- Label them as `fracture` or `normal`.

## 🏗️ Model Architecture

The notebook supports using transfer learning via pretrained CNN architectures such as:
- ResNet50
- EfficientNetB0
- MobileNetV2 (optional)

Each model is fine-tuned with:
- Global Average Pooling
- Fully Connected Layers
- Softmax activation (for binary classification)

## 🔁 Training Pipeline

- Dataset split into training, validation, and test sets.
- Data augmentation using `ImageDataGenerator`.
- Early stopping to prevent overfitting.
- Performance tracking using `ROC AUC`, `confusion matrix`, and classification reports.

## 📊 Evaluation

The following metrics are computed:
- **Accuracy**
- **Sensitivity (Recall)**
- **Specificity**
- **F1-score**
- **AUC-ROC**

Visualization includes:
- Confusion matrix
- ROC curve
- Sample predictions

## 🧠 Model Interpretability

Optionally, Grad-CAM or other heatmap-based methods can be used to interpret which parts of the image the model focuses on during prediction.

## 🚀 Results

(Add your final results here once available. Example below:)

| Model         | Accuracy | AUC-ROC | Sensitivity | Specificity |
|---------------|----------|---------|-------------|-------------|
| ResNet50      | 88.2%    | 0.91    | 0.86        | 0.90        |
| EfficientNetB0| 89.5%    | 0.93    | 0.88        | 0.91        |

## 📝 License

This project is for educational and research purposes. For commercial use, please check the licensing of the MURA dataset and pretrained models.

## 🙌 Acknowledgements

- [Stanford ML Group](https://stanfordmlgroup.github.io/)
- TensorFlow/Keras for deep learning frameworks
- Original authors of the MURA dataset


