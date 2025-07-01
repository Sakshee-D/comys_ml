file_content = """
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to load and preprocess image data
def load_data(dataset_path, img_height, img_width, batch_size):
    \"\"\"
    Loads and preprocesses image data using ImageDataGenerator.

    Args:
        dataset_path (str): Path to the root dataset directory (e.g., 'train' or 'val').
        img_height (int): Target height of the images.
        img_width (int): Target width of the images.
        batch_size (int): Batch size for data generation.

    Returns:
        tf.keras.preprocessing.image.DirectoryIterator: Data generator.
    \"\"\"
    # Apply data augmentation only for the training set
    if 'train' in dataset_path:
        datagen = ImageDataGenerator(
            rescale=1./255,          # Normalize pixel values to [0, 1]
            rotation_range=20,       # Randomly rotate images by 20 degrees
            width_shift_range=0.2,   # Randomly shift images horizontally
            height_shift_range=0.2,  # Randomly shift images vertically
            shear_range=0.2,         # Apply shear transformation
            zoom_range=0.2,          # Randomly zoom into images
            horizontal_flip=True,    # Randomly flip images horizontally
            fill_mode='nearest'      # Strategy for filling in new pixels created by transformations
     )
    else:
        # Only rescale for validation and test sets (no augmentation)
        datagen = ImageDataGenerator(rescale=1./255)

    # Create a data generator from the directory
    generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_height, img_width), # Resize images to specified dimensions
        batch_size=batch_size,               # Batch size for feeding into the model
        class_mode='binary',                 # Since we have two classes (male/female)
        shuffle=True if 'train' in dataset_path else False # Shuffle only training data
    )
    return generator

if __name__ == '__main__':
    # Define base path for data (Kaggle specific path)
    base_data_path = '/kaggle/input/comys-hackathon5/Comys_Hackathon5/Task_A'
    train_path = os.path.join(base_data_path, 'train')
    val_path = os.path.join(base_data_path, 'val')

    # Define image dimensions and batch size
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    BATCH_SIZE = 32

    # Test loading training data
    print(f"Testing data loading from: {train_path}")
    train_generator = load_data(train_path, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)
 print(f"Found {train_generator.samples} training images belonging to {train_generator.num_classes} classes.")

    # Test loading validation data
    print(f"Testing data loading from: {val_path}")
    val_generator = load_data(val_path, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)
    print(f"Found {val_generator.samples} validation images belonging to {val_generator.num_classes} classes.")
"""

# Write the content to data_utils.py file
with open('data_utils.py', 'w') as f:
    f.write(file_content.strip())

print("data_utils.py created successfully.")
