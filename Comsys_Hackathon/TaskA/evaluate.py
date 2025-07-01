file_content = """
import os
import sys
import numpy as np
import tensorflow as tf
# Import all necessary metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


sys.path.append('.')

# Import data loading utility
from data_utils import load_data

# Define constants for data paths and model paths
BASE_DATA_PATH = '/kaggle/input/comys-hackathon5/Comys_Hackathon5/Task_A'
VAL_DIR = os.path.join(BASE_DATA_PATH, 'val') # Default path for evaluation
MODEL_PATH = '/kaggle/working/gender_classification_model.h5' # Path where the trained model is saved

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

# Main function to evaluate the model
def evaluate_model(dataset_path, dataset_name="Validation"):
    \"\"\"
    Evaluates the trained model on a specified dataset (validation or test).

    Args:
 dataset_path (str): Path to the dataset directory to evaluate.
        dataset_name (str): Name of the dataset for logging (e.g., "Validation", "Test").
    \"\"\"
    print(f"\\n--- Evaluating on {dataset_name} Set ---")

    # Check if the model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please run train.py first.")
        return # Exit if model is not found

    print(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH) # Load the saved model
    print("Model loaded successfully.")

    print(f"Loading {dataset_name} data from: {dataset_path}")
    # Load data for evaluation (shuffle=False is crucial here)
    eval_generator = load_data(dataset_path, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)

    # Get class labels from the generator
    class_labels = list(eval_generator.class_indices.keys())
    print(f"Detected classes for evaluation: {class_labels}")

    print(f"Running evaluation on {dataset_name} set...")
    # Evaluate model on the generator 
    loss, accuracy = model.evaluate(eval_generator, verbose=0)
    print(f"{dataset_name} Loss: {loss:.4f}")
    print(f"{dataset_name} Accuracy: {accuracy:.4f}")
 eval_generator.reset()
    # Get raw predictions from the model
    predictions = model.predict(eval_generator)
    num_classes = len(class_labels)
    # Convert raw predictions to binary labels (0 or 1)
    pred_labels = np.argmax(predictions, axis=1) if num_classes > 2 else (predictions > 0.5).astype(int).flatten()
  
    true_labels = eval_generator.classes[eval_generator.index_array] # Correct way to get ordered true labels

    print(f"\\n{dataset_name} Classification Report:")
    # Print a detailed classification report including precision, recall, f1-score per class
    print(classification_report(true_labels, pred_labels, target_names=class_labels))

    print(f"\\n{dataset_name} Confusion Matrix:")
    # Print the confusion matrix
    print(confusion_matrix(true_labels, pred_labels))

    # Calculate and print individual metrics for weighted average
    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, average='weighted', zero_division=0) # 'weighted' handles imbalance
    rec = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

    print(f"\\nIndividual Metrics for {dataset_name}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (weighted): {prec:.4f}")
    print(f"Recall (weighted): {rec:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")
    print(f"--- {dataset_name} Evaluation Complete ---")


# This block executes when evaluate.py is run directly
if __name__ == '__main__':
    evaluate_model(VAL_DIR, "Validation") # By default, evaluate on the validation set
"""


with open('evaluate.py', 'w') as f:
    f.write(file_content.strip())

print("evaluate.py created successfully.")

print("\nExecuting evaluate.py for Validation Set...")
!python evaluate.py