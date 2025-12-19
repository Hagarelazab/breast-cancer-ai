
Breast Cancer Classification Using Ultrasound Images

Transfer Learning & Vision Transformer Approach

Breast cancer is one of the most common cancers among women worldwide and remains a leading cause of cancer-related mortality. Early and accurate diagnosis is critical for improving treatment outcomes and patient survival.

Ultrasound imaging is widely used for breast cancer screening due to its non-invasive nature, low cost, and effectiveness in dense breast tissue. However, manual interpretation of ultrasound images is highly dependent on radiologist expertise and is subject to inter-observer variability.

This project presents an automated deep learning framework for breast ultrasound image classification, aiming to classify images into three categories:

Normal
Benign
Malignant

The proposed system integrates:

 Advanced ROI-based preprocessing
 Transfer learning  using state-of-the-art CNN architectures
 A Vision Transformer (ViT) model
Progressive fine-tuning strategies to improve generalization on limited medical data

---

Dataset Description – BUSI Dataset

This study uses the Breast Ultrasound Images Dataset (BUSI) introduced by Al-Dhabyani et al. (2020).

 Dataset Overview

Total images: 780
Classes:

  Normal
  Benign
  Malignant
Image format: PNG
Resolution: ~500 × 500 pixels
Annotations: Ground-truth segmentation masks for benign and malignant tumors

Reference:
Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. (2020). *Dataset of breast ultrasound images*. Data in Brief.

Preprocessing Pipeline

Ultrasound images suffer from speckle noise, scanning artifacts, and irrelevant background regions. To address these challenges, a custom preprocessing pipeline was implemented.

 Preprocessing Steps

1. Artifact & Text Removal

 Thresholding to detect scanner text and markers
 Morphological operations
 Image inpainting to reconstruct affected regions

2. Speckle Noise Reduction

 Median filtering to suppress ultrasound-specific noise

3. ROI Cropping (Mask-Based)

 Tumor regions extracted using ground-truth segmentation masks
 Bounding box cropping with padding to preserve surrounding context

4. Image Standardization

 Resize to 224 × 224 
 Pixel normalization

5. Data Augmentation (Training Only)

  Rotation
  Width/height shifts
  Zoom
  Horizontal flipping

This preprocessing strategy improves feature clarity and enables models to focus on clinically relevant regions.

---

 Deep Learning Models

 1. Transfer Learning (CNN-Based Models)

To overcome the limited dataset size, transfer learning was applied using ImageNet-pretrained backbones:

 DenseNet121
 DenseNet201
 ResNet50
 EfficientNetB0

Training Strategy

 Pretrained backbone frozen initially
 Custom classification head added
Class-weighted loss to handle data imbalance
 Progressive fine-tuning with reduced learning rates
 Batch Normalization layers frozen during fine-tuning

---

 2. Vision Transformer (ViT)

In addition to CNNs, a Vision Transformer (ViT) model was implemented from scratch:

 Image patch extraction (16 × 16)
 Patch embedding with positional encoding
 Multi-head self-attention layers
 Transformer encoder blocks
Global average pooling classification head

The ViT model was trained end-to-end using a low learning rate.

---

Training Configuration

Optimizer: Adam
Loss Function: Categorical Cross-Entropy
Batch size: 16
Epochs: 30 (with early stopping)
Class imbalance handling: Class weights

Callbacks Used:

 Early stopping
 ReduceLROnPlateau
 Best checkpoint saving

 Results and Performance

All models were evaluated using the same preprocessing, augmentation, and training strategy to ensure a fair comparison.

 Best Model

DenseNet201 achieved the highest validation performance.

Best validation accuracy (before fine-tuning):90.20%
Final fine-tuned validation accuracy: 90.20%
Final validation loss: 0.2782

 Fine-Tuning Strategy (DenseNet201)

Stage A – Partial Unfreezing

 Top layers unfrozen
 Batch Normalization frozen
 Learning rate: `1e-5`
  Validation accuracy: ~89%

Stage B – Full Unfreezing

 Entire backbone unfrozen
Batch Normalization frozen
Initial learning rate: `3e-6` (adaptively reduced)
Early stopping applied
Peak validation accuracy observed: 90.85%
 Final restored checkpoint accuracy: 90.20%

---

 Model Comparison

| Model              | Architecture            | Best Validation Accuracy |
| ------------------ | ----------------------- | ------------------------ |
| DenseNet121        | CNN (Transfer Learning) | < 90%                    |
| **DenseNet201**    | CNN (Transfer Learning) | 90.20%                   |
| ResNet50           | CNN (Transfer Learning) | < 90%                    |
| EfficientNetB0     | CNN (Transfer Learning) | < 90%                    |
| Vision Transformer | Transformer             | < DenseNet201            |

