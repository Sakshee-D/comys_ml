file_content = """
import tensorflow as tf
from tensorflow.keras.preprocessing import image # For loading and processing images
import numpy as np
import os
import sys

# Add current directory to sys.path to allow imports like model_utils
sys.path.append('.')

# Define constants (should match those used in train.py and evaluate.py)
IMG_HEIGHT = 128
IMG_WIDTH = 128
MODEL_PATH = '/kaggle/working/gender_classification_model.h5' # Path to the saved model
CLASS_LABELS = ['female', 'male'] # Ensure these match your generator's labels (alphabetical order)

# Function to predict the class of a single image
def predict_single_image(image_path, model_path=MODEL_PATH, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, class_labels=CLASS_LABELS):
    \"\"\"
    Loads a trained model and predicts the class of a single image.

    Args:
        image_path (str): The path to the image file.
        model_path (str): The path to the saved Keras model (.h5 file).
        img_height (int): The height to resize the image to.
        img_width (int): The width to resize the image to.
        class_labels (list): A list of class names (e.g., ['female', 'male'])
 Returns:
        tuple: A tuple containing the predicted class name (str) and the raw
               prediction score (float).
               Returns (None, None) if the image or model cannot be loaded.
    \"\"\"
    # Check if image and model files exist
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None, None
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please train the model first.")
        return None, None

    try:
        # Load the trained model
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, None

    try:
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(img_height, img_width)) # Load image and resize
        img_array = image.img_to_array(img) # Convert image to NumPy array
        img_array = np.expand_dims(img_array, axis=0) # Add a batch dimension (model expects a batch of images)
        img_array /= 255.0 # Rescale pixel values to [0, 1] (must match training preprocessing)

        # Make prediction
        prediction = model.predict(img_array)[0][0] # Get the single scalar prediction for binary classification
 predicted_class_index = (prediction > 0.5).astype(int) # Convert probability to binary class index (0 or 1)
        predicted_class_name = class_labels[predicted_class_index] # Map index to class name

        return predicted_class_name, prediction

    except Exception as e:
        print(f"Error processing image {image_path} or making prediction: {e}")
        return None, None


if __name__ == '__main__':
    # Example usage:
    # IMPORTANT: Replace this with an actual path to an image in your dataset on Kaggle or locally
    # For a quick test, you can pick an image from the val or train folder
    # Example for Kaggle:
    # '/kaggle/input/comys-hackathon5/Comys_Hackathon5/Task_A/val/female/female_100.jpg'
    example_image_path = '/kaggle/input/comys-hackathon5/Comys_Hackathon5/Task_A/val/female/female_1.jpg' # REPLACE THIS WITH A REAL IMAGE PATH

    print(f"Attempting to predict for: {example_image_path}")
    predicted_gender, score = predict_single_image(example_image_path)

    if predicted_gender is not None:
        print(f"Predicted Gender: {predicted_gender} (Score: {score:.4f})")
    else:
        print("Prediction failed.")
"""
with open('predict_single.py', 'w') as f:
    f.write(file_content.strip())

print("predict_single.py created successfully.")
