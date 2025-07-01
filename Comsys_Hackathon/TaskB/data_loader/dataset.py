import os
import random
import tensorflow as tf
from utils.image_utils import preprocess  # Custom preprocessing function for images

# Getting all valid image file paths from a folder
def get_image_files(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(os.path.join(folder, f))
    ]

# Generating balanced positive and negative image pairs
def generate_balanced_pairs(data_path, max_pairs_per_identity=50):
    identities = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    identity_to_images = {}

    # Gather all image paths per identity
    for identity in identities:
        folder = os.path.join(data_path, identity)
        images = [
            os.path.join(root, f)
            for root, _, files in os.walk(folder)
            for f in files
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if len(images) >= 2:
            identity_to_images[identity] = images

    pairs = []
    labels = []
    all_identities = list(identity_to_images.keys())

    for identity in identity_to_images:
        imgs = identity_to_images[identity]
        random.shuffle(imgs)

        # Positive pairs (same identity), ensuring some sunny images are included. As got low accuracy sunny images while validating earlier
        pos_pairs = []
        sunny_imgs = [img for img in imgs if "sunny" in img.lower()]
        other_imgs = [img for img in imgs if "sunny" not in img.lower()]

        # Forced some sunny vs other image pairs
        for s_img in sunny_imgs:
            paired = random.choice(other_imgs) if other_imgs else random.choice(sunny_imgs)
            pos_pairs.append((s_img, paired))

        # Added additional positive pairs randomly
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                pos_pairs.append((imgs[i], imgs[j]))
        random.shuffle(pos_pairs)
        pos_pairs = pos_pairs[:max_pairs_per_identity]

        pairs.extend(pos_pairs)
        labels.extend([1] * len(pos_pairs))

        # Negative pairs (different identities)
        neg_pairs = []
        other_ids = [id2 for id2 in all_identities if id2 != identity]
        for _ in range(len(pos_pairs)):
            id2 = random.choice(other_ids)
            img1 = random.choice(imgs)
            img2 = random.choice(identity_to_images[id2])
            neg_pairs.append((img1, img2))
        pairs.extend(neg_pairs)
        labels.extend([0] * len(neg_pairs))

    return pairs, labels

# Custom augmentation for sunny and general images
def augment_img(img, filename):
    is_sunny = tf.strings.regex_full_match(tf.strings.lower(filename), ".*sunny.*")

    def sunny_aug():
        img_bright = tf.image.adjust_brightness(img, delta=0.2)
        img_hue = tf.image.adjust_hue(img_bright, delta=0.05)
        img_sat = tf.image.adjust_saturation(img_hue, 1.3)
        img_noise = img_sat + tf.random.normal(tf.shape(img_sat), mean=0.0, stddev=0.02)
        return tf.clip_by_value(img_noise, 0.0, 1.0)

    img = tf.cond(is_sunny, sunny_aug, lambda: img)

    # Apply generic augmentations
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=1.0, upper=1.2)
    img = tf.image.random_saturation(img, lower=1.0, upper=1.2)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    return tf.clip_by_value(img, 0.0, 1.0)

# Created a tf.data.Dataset of image pairs for training or validation
def make_dataset(pairs, labels, batch_size=16, augment=False):
    def preprocess_pair(path1, path2, label):
        img1 = preprocess(path1)
        img2 = preprocess(path2)

        if augment:
            img1 = augment_img(img1, path1)
            img2 = augment_img(img2, path2)

        return (img1, img2, label)

    paths1, paths2 = zip(*pairs)
    dataset = tf.data.Dataset.from_tensor_slices((list(paths1), list(paths2), labels))
    dataset = dataset.map(lambda x, y, z: preprocess_pair(x, y, z))
    return dataset.batch(batch_size).prefetch(8)
