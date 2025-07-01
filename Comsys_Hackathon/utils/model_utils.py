file_content = """
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2 # Import MobileNetV2 for transfer learning

# Function to build the CNN model with MobileNetV2
def build_model(img_height, img_width, num_classes=2, initial_learning_rate=1e-5):
    \"\"\"
    Builds a Convolutional Neural Network (CNN) model using MobileNetV2 for transfer learning.
    The base model is unfrozen from the start for single-phase training.

    Args:
        img_height (int): Height of the input images.
        img_width (int): Width of the input images.
        num_classes (int): Number of output classes (2 for binary classification).
        initial_learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        tf.keras.Model: Compiled Keras model.
    \"\"\"
   
    base_model = MobileNetV2(input_shape=(img_height, img_width, 3),
                             include_top=False, # Exclude the classification head
                             weights='imagenet') # Use pre-trained weights from ImageNet

    # Unfreeze the entire base model from the beginning for single-phase training.
    base_model.trainable = True
    # Create a new model by adding custom layers on top of the pre-trained base
    model = Sequential([
        base_model, # Add the pre-trained MobileNetV2 base model
        GlobalAveragePooling2D(), # Replaces Flatten, reducing features to a single vector per channel
        Dense(128, activation='relu'), # A dense layer with ReLU activation
        Dropout(0.5), # Dropout for regularization to prevent overfitting
        Dense(1, activation='sigmoid') # Final output layer for binary classification (sigmoid for probabilities)
    ])

    # Compile the model with a very low learning rate suitable for fine-tuning the entire network
    model.compile(optimizer=Adam(learning_rate=initial_learning_rate),
                  loss='binary_crossentropy', # Loss function for binary classification
                  metrics=['accuracy']) # Metric to monitor during training
    return model

# The original fine_tune_model function is not needed for single-phase training
# def fine_tune_model(...)
"""

# Write the content to model_utils.py file
with open('model_utils.py', 'w') as f:
    f.write(file_content.strip())

print("model_utils.py modified for single-phase training.")
