import tensorflow as tf

def preprocess(file_path):
    # Read and decode image
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img

def load_images_from_folder(folder_path):
    import os
    paths = []
    for img in os.listdir(folder_path):
        full_path = os.path.join(folder_path, img)
        if full_path.endswith('.jpg') or full_path.endswith('.png'):
            paths.append(full_path)
    return paths
