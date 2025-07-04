import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import math

# Global Configuration 

BASE_DATA_PATH = '/kaggle/input/comys-hackathon5/Comys_Hackathon5/Task_A'
TRAIN_DIR = os.path.join(BASE_DATA_PATH, 'train')
VAL_DIR = os.path.join(BASE_DATA_PATH, 'val')

# >>> IMPORTANT: enter DESIRED TEST DATA PATH HERE <<<

TEST_DATA_PATH_FOR_EVALUATION = VAL_DIR 
# Example if We had a separate 'test' folder:
# TEST_DATA_PATH_FOR_EVALUATION = os.path.join(BASE_DATA_PATH, 'test')


IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
TOTAL_EPOCHS = 20
INITIAL_LEARNING_RATE = 1e-5
MODEL_SAVE_PATH = '/path/to/gender_classification_model.h5'
RESULTS_DIR = '/path/to/save/results' # Directory for saving metrics files


# Utility Functions 

def load_data(dataset_path, img_height, img_width, batch_size, is_training=False):
    """
    Loads and preprocesses image data using ImageDataGenerator.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    if is_training:
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        datagen = ImageDataGenerator(rescale=1./255)

    generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=is_training # Shuffle training data, do not shuffle validation/test
    )
    return generator

def build_model(img_height, img_width, num_classes=2, initial_learning_rate=1e-5):
    """
    Builds a CNN model using MobileNetV2 for transfer learning.
    """
    base_model = MobileNetV2(input_shape=(img_height, img_width, 3),
                             include_top=False,
                             weights='imagenet')

    base_model.trainable = True

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid') # 1 unit for binary classification
    ])

    model.compile(optimizer=Adam(learning_rate=initial_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# Evaluation Function 
# This function evaluates the model on a given path and returns its metrics.
def evaluate_model(data_path_for_evaluation, dataset_name="Evaluation", model_to_evaluate=None):
    """
    Evaluates a given model on a specified dataset path.
    Returns a dictionary containing all metrics and report string.
    """
    print(f"\n--- Evaluating on {dataset_name} Set ---")

    if model_to_evaluate is None:
        if not os.path.exists(MODEL_SAVE_PATH):
            print(f"Error: Model not found at {MODEL_SAVE_PATH}. Please ensure training completed successfully and model is saved.")
            return {}
        model_to_evaluate = tf.keras.models.load_model(MODEL_SAVE_PATH)
        print(f"Loaded model for evaluation: {MODEL_SAVE_PATH}")

    eval_generator = load_data(data_path_for_evaluation, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)

    if eval_generator.samples == 0:
        print(f"No images found in {data_path_for_evaluation} for evaluation. Cannot proceed.")
        return {}

    class_labels = list(eval_generator.class_indices.keys())
    
    eval_generator.reset() # Important to reset generator for consistent evaluation
    predictions = model_to_evaluate.predict(eval_generator, steps=eval_generator.samples // eval_generator.batch_size + (eval_generator.samples % eval_generator.batch_size != 0), verbose=0)

    pred_labels = (predictions > 0.5).astype(int).flatten()
    true_labels = eval_generator.classes[eval_generator.index_array][:len(pred_labels)]

    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, target_names=class_labels, zero_division=0)

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion Matrix': cm,
        'Report': report, # Store full report string
        'Class_Labels': class_labels
    }


#  Training Function 
def train_model_pipeline():
    """
    Trains the model and saves the best version.
    Returns the path to the best saved model.
    """
    print("--- STARTING SINGLE-PHASE MODEL TRAINING PROCESS ---")

    print("\n[1/3] Preparing Data Generators...")
    train_generator = load_data(TRAIN_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, is_training=True)
    val_generator = load_data(VAL_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)

    class_labels = list(train_generator.class_indices.keys())
    print(f"Detected Classes: {class_labels}")

    print("Calculating class weights for imbalanced dataset...")
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Computed Class Weights: {class_weights_dict}")

    print("\n[2/3] Building and Training Model (Single Phase)...")
    model = build_model(IMG_HEIGHT, IMG_WIDTH, len(class_labels), initial_learning_rate=INITIAL_LEARNING_RATE)
    print("\nModel Summary:")
    model.summary(print_fn=print)

    callbacks = [
        EarlyStopping(patience=7, monitor='val_loss', restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-8)
    ]

    print(f"Starting single-phase training for up to {TOTAL_EPOCHS} epochs...")
    history = model.fit(
        train_generator,
        epochs=TOTAL_EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weights_dict
    )
    print("Model training complete.")

    # ModelCheckpoint automatically saves the best model, so we just return its path
    return MODEL_SAVE_PATH


# Main Execution Block 
if __name__ == '__main__':
    # Ensure working directories exist
    os.makedirs('/kaggle/working/models', exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("--- Starting Full Task A Pipeline Execution (Monolithic, Single Cell) ---")

    # 1. Train the model and get the path to the best saved model
    best_model_path = train_model_pipeline()

    # 2. Load the best model
    if not os.path.exists(best_model_path):
        print(f"Error: Best model not found at {best_model_path}. Exiting.")
        # In a single cell, SystemExit isn't ideal, just print and allow cell to finish
        exit(1) 
    
    loaded_model = tf.keras.models.load_model(best_model_path)
    print(f"Successfully loaded best model from {best_model_path}")

    # 3. Perform final evaluation on the specified test data path
    print(f"\n--- Performing Final Evaluation on Test Data: {TEST_DATA_PATH_FOR_EVALUATION} ---")
    final_test_metrics = evaluate_model(TEST_DATA_PATH_FOR_EVALUATION, dataset_name="Examiner's Test Set", model_to_evaluate=loaded_model)

    # 4. Print Final Test Set Evaluation Metrics for Examiner
    print("\n--- Final Test Set Evaluation Metrics for Examiner ---")
    if final_test_metrics:
        # Print the classification report directly for proper formatting
        print("Test Classification Report:")
        print(final_test_metrics['Report']) 

        # Print the confusion matrix directly for proper formatting
        print("\nTest Confusion Matrix:")
        print(final_test_metrics['Confusion Matrix'])

        # Print individual metrics
        print("\nIndividual Metrics for Test:")
        print(f"Accuracy: {final_test_metrics['Accuracy']:.4f}")
        print(f"Precision (weighted): {final_test_metrics['Precision']:.4f}")
        print(f"Recall (weighted): {final_test_metrics['Recall']:.4f}")
        print(f"F1-Score (weighted): {final_test_metrics['F1-Score']:.4f}")
        print("--- Test Evaluation Complete ---")

        # Prepare content for saving to file
        # Using repr() for numpy array to ensure it's written as a string.
        file_output_content = (
            "Test Classification Report:\n"
            f"{final_test_metrics['Report']}\n"
            "\nTest Confusion Matrix:\n"
            f"{repr(final_test_metrics['Confusion Matrix'])}\n" 
            "\nIndividual Metrics for Test:\n"
            f"Accuracy: {final_test_metrics['Accuracy']:.4f}\n"
            f"Precision (weighted): {final_test_metrics['Precision']:.4f}\n"
            f"Recall (weighted): {final_test_metrics['Recall']:.4f}\n"
            f"F1-Score (weighted): {final_test_metrics['F1-Score']:.4f}\n"
            "--- Test Evaluation Complete ---\n"
        )

        file_path = os.path.join(RESULTS_DIR, 'taskA_metrics_test.txt')
        with open(file_path, 'w') as f:
            f.write(file_output_content)
        print(f"Final Test Set metrics saved to {file_path}")

    else:
        print("Failed to obtain final evaluation results for the test set.")

    print("\n--- Full Task A Pipeline Execution COMPLETE ---")

