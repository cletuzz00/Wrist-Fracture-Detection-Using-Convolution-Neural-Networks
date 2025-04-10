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

- **Original Source**: [MURA Dataset by Stanford ML Group](https://stanfordmlgroup.github.io/competitions/mura/)
- **Version Used**:[Kaggle - Better Mura](https://www.kaggle.com/datasets/sudhanshusuryawanshi/better-mura)
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

