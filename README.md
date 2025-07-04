# COMSYS Hackathon: Gender Classification & Face Verification

Team Name: AlgoQueens

Team Member 1: Sakshee Dhormale
Team Member 2: Shreya Pawar


# Table of Contents: 

A. Introduction

B. Task A - Gender Classification

Innovation and Unique Approach, 
Resources & Links, 
Evaluation & Metrics Summary, 
Dependencies
     
C. Task B - Face Matching

Innovation and Unique Approach, 
Resources & Links, 
Evaluation & Metrics Summary, 
Dependencies
     
D. How to Test ( For Task A and Task B)

E. Future Improvements for both Tasks

# Task A- Gender Classification

 Gender Classification using MobileNetV2 Transfer Learning

In Task A, we implemented a Convolutional Neural Network (CNN) leveraging Transfer Learning with MobileNetV2 for gender classification on facial images. 
The model is designed to classify whether a given face image represents a male or a female, even under real-world variations such as different lighting conditions, poses, and expressions.
This approach is well-suited for scenarios with:

  1. Limited specific training images per gender.
  2. Noisy or varied image inputs from real-world scenarios.
  3. Classification-based identification applications.



# Innovation and Unique Approach

1. Transfer Learning with MobileNetV2: A pre-trained MobileNetV2 model (on ImageNet) serves as a powerful feature extractor. Its layers are fine-tuned during training to adapt to the specific nuances of gender classification.
2. Image Augmentation for Robustness: The model is trained on dynamically augmented images (e.g., flipped, rotated, brightened, zoomed, shifted) to significantly improve its ability to generalize and handle real-world variations in input data.
3. Class Weighting for Imbalance: Class weights are computed and applied during training to mitigate the effects of potential class imbalance in the dataset, ensuring the model learns effectively from both gender categories.
4. Early Stopping for Generalization: Training is automatically halted when validation loss stops improving over several epochs (`patience=7`), preventing overfitting and ensuring the best performing model weights are saved.
5. Self-Contained Script: The entire training and evaluation pipeline is encapsulated within a single, monolithic Python script for straightforward execution in notebook environments like Kaggle/Colab.


 Resource                         Link

 Notebook (IPYNB)               https://drive.google.com/file/d/1pBJuOpg1FanTwuK3tkU0XMcKayxUO2YU/view?usp=drive_link

 Model Diagram                  https://drive.google.com/file/d/1H3fktBTg184M-r0PEbWbqPA_P2y0cu3l/view?usp=drive_link

 Test Script                    https://drive.google.com/file/d/1ViA6O32ausF0L8-xSJcnn5tNpoY-zulR/view?usp=drive_link



# Evaluation Metrics Summary
      
1.Training Results

         Found 1926 images belonging to 2 classes.

        Individual Metrics for Training :

          Accuracy : 0.9528
          Precision : 0.9162
          Recall : 0.9458
          F1-score : 0.9299

        Confusion Matrix:
          [[ 368   26]
          [  65 1467]]

 
 2. Validation Results
 
          Found 422 images belonging to 2 classes.
         
        Individual Metrics for Validation:

          Accuracy: 0.9076
          Precision (weighted): 0.9059
          Recall (weighted): 0.9076
          F1-Score (weighted): 0.9047

        Validation Confusion Matrix:
        [[ 77  28]
         [ 11 306]]


 # Dependencies:

The following Python libraries are required to run this script:

 tensorflow 
 scikit-learn
 numpy
 Pillow (often an implicit dependency for image processing with TensorFlow)

To install all dependencies (if running in a custom environment):
pip install tensorflow scikit-learn numpy Pillow

 
# Task B - Face Matching
 
Face Verification with Siamese Network
 
In task B, we implemented a Siamese Neural Network for person verification on facial images. 
The model is designed to verify whether two face images belong to the same person (1) or different (0), even under distortions like blurriness, lighting changes, or resolution loss.
 
This approach is well-suited for scenarios with:
 
 1. Limited images per person
 2. Noisy/distorted inputs
 3. Verification-based identification

 
 
# Innovation and Unique Approach
 
1. Siamese Network Design: A single CNN is used on both a reference and a distorted image to produce L2-normalized embeddings that represent each image's identity.
2. Embedding Comparison: Each image is converted into an embedding (feature vector), and cosine similarity is used to compare them — made reliable through L2 normalization.
3. Smart Threshold Control: A similarity threshold is tuned to balance between identifying correct matches and rejecting uncertain or incorrect ones.
4. Image Augmentation for Robustness: The model was trained on augmented images (e.g., flipped, brightened, or noised) to improve its ability to handle real-world variations.
5. Early Stopping for Generalization: Training is stopped automatically when validation loss stops improving, preventing overfitting and saving the best version of the model.
 
  Resource                              Link

 Notebook (IPYNB)                  https://drive.google.com/file/d/1R5Y5e9DwazrlD_Z-crKM4kmuY_77JTrA/view?usp=drive_link

 Notebook (PDF Export)             https://drive.google.com/file/d/1wpHTt6n5c8_KRl3sarhDLPSnzj5Xgs8-/view?usp=drive_link

 Model Diagram                     https://drive.google.com/file/d/1NXAKq0uv1ww_ssJRH2-gu3BRe-XgKhjo/view?usp=drive_link

 Test Script                       https://drive.google.com/file/d/1f2GR16ehG3trHwxZv773fqmrZ6r1sc8k/view?usp=drive_link


 
 # Evaluation Metrics Summary
 
 1. Training Results

    Accuracy: 0.8954
    Precision: 0.9059
    Recall: 0.8825
    F1 Score: 0.8941
 
2. Validation Results
 
   Total Evaluated: 2954
   Accepted: 2805
   Rejected/Uncertain: 149
 
   Binary Match Metrics:
   Accuracy: 0.8270
   Precision: 1.0000
   Recall: 0.8270
   F1 Score: 0.9053

 
# Dependencies
 
  tensorflow >= 2.10
  scikit-learn
   matplotlib
   numpy
   Tqdm
 
To install all dependencies:
 pip install -r requirements.txt
 
# How to Test (Task A & Task B)
 
The test script for Task A - https://drive.google.com/file/d/1ViA6O32ausF0L8-xSJcnn5tNpoY-zulR/view?usp=drive_link
Model Task A - https://drive.google.com/file/d/1lwIBU5b9vah0am5l9F3q_fi6EdBmbkIJ/view?usp=drive_link (download)
The test script for Task B - https://drive.google.com/file/d/1f2GR16ehG3trHwxZv773fqmrZ6r1sc8k/view?usp=drive_link
Model Task B - https://drive.google.com/file/d/1o30A96Ly4OWOIFWAOZpqrqeKdJS0TB_w/view?usp=drive_link (download)

# Task A
 Edit the following 3 lines in `test_script.py`:

TEST_DATA_PATH_FOR_EVALUATION= "/kaggle/input/comys-hackathon5/Comys_Hackathon5/Task_A/val"

MODEL_PATH_FOR_EVALUATION = '/path/to/gender_classification_model.h5'

RESULTS_DIR = '/path/to/save/results'  


 # Task B
Edit the following 3 lines in `test_script.py`:

VAL_DIR = "/kaggle/input/dataset-taskb/Task_B_dataset/val"
FINAL_MODEL_PATH = "/kaggle/working/siamese_model_final.h5" (downloaded model path)

Model weights for Task  B are not given in the github repository due to file size limits. Please download them from the above links 

No other file required for test script , the code can run independently given path to model (after downloading) and dataset is provided

Add the path to your dataset in the above codes
download the given model for each task


# Future Improvements 

1. Model Optimization for Deployment
Use TensorFlow Lite or ONNX to compress and optimize the model for mobile or edge devices, making it usable in real-time low-power applications.


2. Add Multi-Angle & Multi-View Robustness
Train with more diverse facial angles and lighting conditions using advanced augmentation or synthetic data (e.g., GANs) to improve real-world performance.


3. Incorporate Face Alignment and Detection
Preprocess images using face alignment/detection (like MTCNN or MediaPipe) before embedding to reduce noise and improve matching accuracy.
