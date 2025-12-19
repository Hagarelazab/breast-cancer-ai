# breast-cancer-ai
Deep learning project for breast cancer detection from medical images. For research and educational purposes only.

We use the Breast Ultrasound Images Dataset (BUSI) introduced by Al-Dhabyani et al. and published in Data in Brief (2020). The dataset consists of 780 breast ultrasound images in PNG format (~500×500 pixels), categorized into three classes: Normal (133 images), Benign (437 images), and Malignant (210 images).
The images were collected from Baheya Hospital for Early Detection and Treatment of Women’s Cancer, Cairo, Egypt, and are widely used as a benchmark dataset for breast ultrasound image analysis and classification.

Dataset Citation

Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. (2020). Dataset of breast ultrasound images. Data in Brief, 28, 104863.


The BUSI dataset is used for research and academic purposes only.
The dataset is not redistributed in this repository. Users should download
the dataset from the original source and cite the original publication.


 Method:
 Pipeline (Step-by-step)

1. Load Data
   Breast ultrasound images and corresponding labels
   (Normal, Benign, Malignant) are loaded from the BUSI dataset.

   2. Preprocessing
   Images are resized and normalized.
   
3.  Data Split
   The dataset is split into training, validation, and test sets
  

4.  Data Augmentation
   Data augmentation techniques such as horizontal/vertical flips,
   rotations, and zooming are applied to the training set only.

5. Model Architecture
   Deep learning models are constructed using CNN backbones such as
   DenseNet121, ResNet50, and EfficientNetB0.

6. Training
   Models are trained using early stopping to prevent overfitting.
   Class weighting is applied to address class imbalance, and the
   best-performing weights are saved.

7. Evaluation 
   Model performance is evaluated on the test set using accuracy,
   precision, recall, F1-score, and confusion matrix analysis.

8. Reporting
   Final results are summarized in tables and visualized using
   training and validation loss/accuracy curves.

