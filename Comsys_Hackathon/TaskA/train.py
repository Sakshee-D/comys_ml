file_content = """
import os
import sys
import numpy as np
# Import metrics for evaluation, including confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import class_weight # For handling imbalanced datasets
import tensorflow as tf
# Import Keras callbacks for training control
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator # For data loading
import math # Import math for ceil function


sys.path.append('.')

# Import model building utility
from model_utils import build_model


# Define constants for data paths and model parameters 
BASE_DATA_PATH = '/kaggle/input/comys-hackathon5/Comys_Hackathon5/Task_A'
TRAIN_DIR = os.path.join(BASE_DATA_PATH, 'train')
VAL_DIR = os.path.join(BASE_DATA_PATH, 'val')

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
TOTAL_EPOCHS = 20 # Maximum number of epochs, EarlyStopping will stop earlier if performance plateaus
MODEL_SAVE_PATH = '/kaggle/working/gender_classification_model.h5' # Path to save the best model
INITIAL_LEARNING_RATE = 1e-5 # Learning rate for the unfrozen MobileNetV2 fine-tuning


# Main function to train the model
def train_model():
    print("--- STARTING SINGLE-PHASE MODEL TRAINING PROCESS ---")

    # --- Data Loading and Preparation ---
    print("\\n[1/3] Preparing Data Generators...")
    # ImageDataGenerator for training data with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=20, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode='nearest'
    )
    # ImageDataGenerator for validation data 
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Create generators from directories
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=True
    )
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=False
    )
 # Get class labels (e.g., ['female', 'male'])
    class_labels = list(train_generator.class_indices.keys())
    print(f"Detected Classes: {class_labels}")

    # Calculate class weights to handle potential class imbalance
    print("Calculating class weights for imbalanced dataset...")
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Computed Class Weights: {class_weights_dict}")

    # --- Model Building and Training (Single Phase) ---
    print("\\n[2/3] Building and Training Model (Single Phase)...")
    # Build the model using the utility function
    model = build_model(IMG_HEIGHT, IMG_WIDTH, len(class_labels), initial_learning_rate=INITIAL_LEARNING_RATE)
    print("\\nModel Summary:")
    model.summary(print_fn=print) # Print model architecture summary

    # Define callbacks for training control
    callbacks = [
        EarlyStopping(patience=7, monitor='val_loss', restore_best_weights=True, verbose=1), # Stops training if validation loss doesn't improve for 7 epochs
        ModelCheckpoint(filepath=MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1), # Saves the model with the best validation loss
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-8) # Reduces learning rate if validation loss plateaus
    ]
 print(f"Starting single-phase training for up to {TOTAL_EPOCHS} epochs...")
    # Fit the model to the training data
    history = model.fit(
        train_generator,
        epochs=TOTAL_EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weights_dict # Apply class weights
    )
    print("Model training complete.")


    # --- Final Evaluation on Training Set ---
    print("\\n[3/3] Final Evaluation on Training Set...")
    # Load the best model saved by ModelCheckpoint for final evaluation
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    print(f"Loaded final best model for evaluation: {MODEL_SAVE_PATH}")

   
    # This ensures predictions are in the correct order for metric calculation
    final_train_eval_datagen = ImageDataGenerator(rescale=1./255)
    final_train_eval_generator = final_train_eval_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=False # CRITICAL: Do NOT shuffle for evaluation
    )
 num_train_samples = final_train_eval_generator.samples
    steps_per_epoch_eval = math.ceil(num_train_samples / BATCH_SIZE) # Calculate steps needed to cover all samples

    print("Collecting predictions for all training samples...")
    all_predictions = []
    all_true_labels = []

    # Iterate through the generator to get predictions and true labels
    for i in range(steps_per_epoch_eval):
        batch_X, batch_y = next(final_train_eval_generator)
        batch_predictions = model.predict(batch_X, verbose=0) # Get raw predictions
        all_predictions.extend(batch_predictions.flatten()) # Flatten and add to list
        all_true_labels.extend(batch_y.flatten()) # Flatten and add true labels

    # Convert predictions to binary labels (0 or 1) based on a 0.5 threshold
    train_pred_labels = (np.array(all_predictions) > 0.5).astype(int)
    train_true_labels = np.array(all_true_labels)


    # Print detailed training performance metrics
    print("\\n--- Training Set Performance Metrics ---")
    print(f"Accuracy: {accuracy_score(train_true_labels, train_pred_labels):.4f}")
    print(f"Precision (Macro Avg): {precision_score(train_true_labels, train_pred_labels, average='macro', zero_division=0):.4f}")
    print(f"Recall (Macro Avg): {recall_score(train_true_labels, train_pred_labels, average='macro', zero_division=0):.4f}")
    print(f"F1-score (Macro Avg): {f1_score(train_true_labels, train_pred_labels, average='macro', zero_division=0):.4f}")

    print("\\nDetailed Classification Report:")
    print(classification_report(train_true_labels, train_pred_labels, target_names=class_labels, zero_division=0))
    # Add Confusion Matrix for a clear view of true vs. predicted counts
    print("\\nConfusion Matrix:")
    cm = confusion_matrix(train_true_labels, train_pred_labels)
    print(cm)
 


    print("\\n--- TRAINING PROCESS COMPLETE ---")


# This block executes when train.py is run directly
if __name__ == '__main__':
    train_model()
"""


with open('train.py', 'w') as f:
    f.write(file_content.strip())

print("train.py has been updated to include the Confusion Matrix in the training metrics.")

print("Executing train.py...")
!python train.py
