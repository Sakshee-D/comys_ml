import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Layer



# Configuration
IMG_SIZE = (100,100)
VAL_DIR = "/kaggle/input/dataset-taskb/Task_B_dataset/val"
FINAL_MODEL_PATH = "/kaggle/working/siamese_model_final.h5"
THRESHOLD = 0.8

def make_embedding():
    inp = Input(shape=(100, 100, 3))  # Input image of size 100x100x3 (RGB)

    # Convolutional layers to extract features
    x = Conv2D(64, (10, 10), activation='relu')(inp)
    x = MaxPooling2D()(x)
    
    x = Conv2D(128, (7, 7), activation='relu')(x)
    x = MaxPooling2D()(x)
    
    x = Conv2D(128, (4, 4), activation='relu')(x)
    x = MaxPooling2D()(x)
    
    x = Conv2D(256, (4, 4), activation='relu')(x)
    x = Flatten()(x)  # Flatten into a 1D vector
    
    x = Dense(4096, activation='sigmoid')(x)  # Final dense layer with sigmoid activation

    return Model(inputs=inp, outputs=x, name="embedding_model")  # Return the embedding model

def make_siamese_model():
    # Two input images: one is the query, the other is the reference/validation
    input_img = Input(shape=(100, 100, 3), name='input_img')
    val_img = Input(shape=(100, 100, 3), name='val_img')

    # Shared embedding model applied to both inputs
    embedding = make_embedding()
    emb1 = embedding(input_img)
    emb2 = embedding(val_img)

    # Compute L1 distance between embeddings
    l1 = L1Dist()([emb1, emb2])

    # Pass the distance through a dense layer to get similarity score (0 to 1)
    out = Dense(1, activation='sigmoid')(l1)

    # Final Siamese model takes two images and returns similarity
    return Model(inputs=[input_img, val_img], outputs=out)


# to preprocess a single image(resize, normalize)
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

    print(f"\nTotal Evaluated: {total_evaluated}")
    print(f"Accepted: {accepted}")
    print(f"Rejected/Uncertain: {total_evaluated - accepted}")
    return y_true, y_pred, total_evaluated, accepted

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

def plot_evaluation_summary(total, accepted):
    rejected = total - accepted
    labels = ['Total Evaluated', 'Accepted', 'Rejected/Uncertain']
    values = [total, accepted, rejected]
    colors = ['steelblue', 'mediumseagreen', 'salmon']

    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=values, palette=colors)
    plt.title('Evaluation Summary')
    plt.ylabel('Count')
    plt.ylim(0, total + 100)
    for i, v in enumerate(values):
        plt.text(i, v + 10, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.show()

# Main execution flow
embedding_model = load_embedding_model(FINAL_MODEL_PATH)
identity_db = build_identity_db(VAL_DIR, embedding_model)
y_true, y_pred, total_evaluated, accepted = evaluate_model(VAL_DIR, embedding_model, identity_db)
binary_true, binary_pred = compute_metrics(y_true, y_pred)
plot_evaluation_summary(total_evaluated, accepted)

