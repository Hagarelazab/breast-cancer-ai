# Breast Cancer Classification Using Ultrasound Images
### Transfer Learning & Vision Transformer Approach

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Deep%20Learning-TensorFlow%20%2F%20PyTorch-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Maintained-brightgreen)

## Project Overview

Breast cancer is one of the most common cancers among women worldwide and remains a leading cause of cancer-related mortality. **Early and accurate diagnosis is critical for improving treatment outcomes and patient survival.**

Ultrasound imaging is widely used for screening due to its non-invasive nature and effectiveness in dense breast tissue. However, manual interpretation is highly dependent on radiologist expertise and subject to inter-observer variability.

This project presents an **automated deep learning framework** to classify breast ultrasound images into three diagnostic categories:
* **Normal**
* **Benign**
* **Malignant**

The proposed system integrates advanced **ROI-based preprocessing**, **Transfer Learning** (CNNs), and **Vision Transformers (ViT)** to improve generalization on limited medical data.

---

## Dataset: BUSI
This study utilizes the **Breast Ultrasound Images Dataset (BUSI)** introduced by Al-Dhabyani et al. (2020).

| Feature | Description |
| :--- | :--- |
| **Total Images** | 780 |
| **Classes** | Normal, Benign, Malignant |
| **Format** | PNG (~500 × 500 pixels) |
| **Annotations** | Ground-truth segmentation masks |

> **Reference:**
> Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. (2020). *Dataset of breast ultrasound images*. Data in Brief.

---

## Preprocessing Pipeline

Ultrasound images often suffer from speckle noise, scanning artifacts, and irrelevant background regions. To address this, we implemented a custom preprocessing pipeline.

### 1. Artifact & Text Removal
* **Thresholding:** Detects scanner text and markers.
* **Morphological Operations:** Refines the selection.
* **Image Inpainting:** Reconstructs affected regions to remove textual noise.

### 2. Speckle Noise Reduction
* **Median Filtering:** Applied to suppress ultrasound-specific speckle noise without blurring edges.

### 3. ROI Cropping (Mask-Based)
* Tumor regions are extracted using **ground-truth segmentation masks**.
* Bounding box cropping is applied with padding to preserve surrounding tissue context.

### 4. Image Standardization
* **Resize:** All images resized to 224 × 224.
* **Normalization:** Pixel normalization for faster convergence.

### 5. Data Augmentation (Training Only)
To prevent overfitting, the following augmentations were applied dynamically:
* Rotation
* Width/Height Shifts
* Zoom & Horizontal Flipping

---

## Deep Learning Models

We explored two distinct architectural approaches to solve this classification problem.

### 1. Transfer Learning (CNN-Based)
To overcome the limited dataset size, we utilized **ImageNet-pretrained backbones**.

* **DenseNet121**
* **DenseNet201**
* **ResNet50**
* **EfficientNetB0**

**Training Strategy:**
* Pretrained backbone frozen initially.
* Custom classification head added.
* **Class-weighted loss** to handle data imbalance.
* Progressive fine-tuning with reduced learning rates.
* Batch Normalization layers frozen during fine-tuning.

### 2. Vision Transformer (ViT)
In addition to CNNs, we implemented a **Vision Transformer (ViT)** model from scratch.

* **Patch Extraction:** Images divided into 16 × 16 patches.
* **Embedding:** Patch embedding with positional encoding.
* **Mechanism:** Multi-head self-attention layers and Transformer encoder blocks.
* **Head:** Global average pooling for classification.

---

## Training Configuration

The models were trained using a consistent strategy to ensure fair comparison.

```yaml
Optimizer:       Adam
Loss Function:   Categorical Cross-Entropy
Batch Size:      16
Epochs:          30 (with Early Stopping)
Imbalance Fix:   Class Weights
Callbacks:       EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
```

---

## Results & Performance

| Model              | Architecture            | Best Validation Accuracy |
| ------------------ | ----------------------- | ------------------------ |
| DenseNet121        | CNN (Transfer Learning) | < 90%                    |
| **DenseNet201** | CNN (Transfer Learning) | 90.20%                   |
| ResNet50           | CNN (Transfer Learning) | < 90%                    |
| EfficientNetB0     | CNN (Transfer Learning) | < 90%                    |
| Vision Transformer | Transformer             | < DenseNet201            |

 


