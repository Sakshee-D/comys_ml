import os
import sys
sys.path.append(os.path.abspath("/kaggle/input/taskb-sourcecodes/final_taskB"))
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import plot_model
from IPython.display import Image, display

from models.siamese_model import make_siamese_model
from config import config

# Configuration
IMG_SIZE = (config["IMG_SIZE"], config["IMG_SIZE"])
VAL_DIR = config["VAL_DIR"]
FINAL_MODEL_PATH = config["FINAL_MODEL_PATH"]
THRESHOLD = config["THRESHOLD"]

# to preprocess a single image (resize, normalize)
def preprocess(path):
    img = load_img(path, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0
    return tf.convert_to_tensor(img, dtype=tf.float32)

# to get L2-normalized embedding from model
def get_normalized_embedding(model, image_tensor):
    emb = model(tf.expand_dims(image_tensor, axis=0))
    return tf.math.l2_normalize(emb, axis=-1)[0]

# to load final trained model and return embedding layer
def load_embedding_model(model_path):
    siamese = make_siamese_model()
    siamese.load_weights(model_path)
    return siamese.get_layer("embedding_model")

# to build identity database: average embeddings per person
def build_identity_db(val_dir, embedding_model):
    identity_db = {}
    for person in os.listdir(val_dir):
        person_dir = os.path.join(val_dir, person)
        if not os.path.isdir(person_dir):
            continue

        distortion_path = os.path.join(person_dir, "distortion")
        if not os.path.exists(distortion_path):
            continue

        reference_images = [
            f for f in os.listdir(person_dir)
            if f != "distortion" and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if len(reference_images) < 1:
            continue

        embeddings = []
        for img_name in reference_images:
            img_path = os.path.join(person_dir, img_name)
            try:
                img = preprocess(img_path)
                emb = get_normalized_embedding(embedding_model, img)
                embeddings.append(emb)
            except Exception:
                continue

        if embeddings:
            avg_embedding = tf.reduce_mean(tf.stack(embeddings), axis=0)
            identity_db[person] = tf.math.l2_normalize(avg_embedding, axis=-1)

    print(f"\nGenerated embeddings for {len(identity_db)} identities.")
    return identity_db

# to evaluate model performance on distorted images
def evaluate_model(val_dir, embedding_model, identity_db):
    y_true = []
    y_pred = []
    total_evaluated, accepted = 0, 0

    for person in tqdm(os.listdir(val_dir), desc="Evaluating"):
        dist_dir = os.path.join(val_dir, person, "distortion")
        if not os.path.isdir(dist_dir):
            continue

        for img_name in os.listdir(dist_dir):
            total_evaluated += 1
            img_path = os.path.join(dist_dir, img_name)
            img = preprocess(img_path)
            query_emb = embedding_model(tf.expand_dims(img, axis=0))
            query_emb = tf.nn.l2_normalize(query_emb, axis=-1)[0]

            best_score = -1
            best_match = None

            for candidate, db_emb in identity_db.items():
                db_emb = tf.nn.l2_normalize(db_emb, axis=-1)
                score = tf.reduce_sum(query_emb * db_emb).numpy()
                if score > best_score:
                    best_score = score
                    best_match = candidate

            prediction = best_match if best_score >= THRESHOLD else "Unknown"
            if prediction != "Unknown":
                accepted += 1

            y_true.append(person)
            y_pred.append(prediction)

    rejected = total_evaluated - accepted
    print(f"\nTotal Evaluated: {total_evaluated}")
    print(f"Accepted: {accepted}")
    print(f"Rejected/Uncertain: {rejected}")
    return y_true, y_pred, total_evaluated, accepted, rejected

# to compute binary classification metrics: correct vs incorrect match
def compute_metrics(y_true, y_pred):
    binary_true = [1] * len(y_true)
    binary_pred = [1 if y_t == y_p else 0 for y_t, y_p in zip(y_true, y_pred)]

    acc = accuracy_score(binary_true, binary_pred)
    prec = precision_score(binary_true, binary_pred, zero_division=0)
    rec = recall_score(binary_true, binary_pred, zero_division=0)
    f1 = f1_score(binary_true, binary_pred, zero_division=0)

    print("\nBinary Match Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    return binary_true, binary_pred

# NEW: Plot bar graph for Total Evaluated, Accepted, Rejected
def plot_evaluation_summary(total, accepted, rejected):
    plt.figure(figsize=(6, 4))
    labels = ["Total Evaluated", "Accepted", "Rejected/Uncertain"]
    values = [total, accepted, rejected]
    colors = ["steelblue", "seagreen", "tomato"]

    plt.bar(labels, values, color=colors)
    plt.title("Model Evaluation Summary")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# Main execution flow
# Visualize model architecture
siamese_model = make_siamese_model()
siamese_model.load_weights(FINAL_MODEL_PATH)
plot_model(siamese_model, to_file="siamese_model_diagram.png", show_shapes=True, show_layer_names=True)
display(Image("siamese_model_diagram.png"))

embedding_model = siamese_model.get_layer("embedding_model")
identity_db = build_identity_db(VAL_DIR, embedding_model)
y_true, y_pred, total, accepted, rejected = evaluate_model(VAL_DIR, embedding_model, identity_db)
binary_true, binary_pred = compute_metrics(y_true, y_pred)
plot_evaluation_summary(total, accepted, rejected)
