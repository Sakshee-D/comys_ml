import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.siamese_model import make_siamese_model
from data_loader.dataset import generate_balanced_pairs, make_dataset
from config import config

# Load and prepare data
pairs, labels = generate_balanced_pairs(config["DATASET_PATH"], config["MAX_PAIRS_PER_IDENTITY"])
labels = np.array(labels)

# Display pair distribution
positive_count = np.sum(labels == 1)
negative_count = np.sum(labels == 0)
print(f"\n Total Pairs: {len(labels)}")
print(f" Positive Pairs (same person): {positive_count}")
print(f" Negative Pairs (different people): {negative_count}")

plt.bar(['Positive Pairs', 'Negative Pairs'], [positive_count, negative_count], color=['green', 'red'])
plt.title('Pair Distribution in Training Data')
plt.ylabel('Number of Pairs')
plt.show()

# Split into training and validation sets
train_pairs, val_pairs, train_labels, val_labels = train_test_split(
    pairs, labels, test_size=0.2, stratify=labels, random_state=42
)

train_dataset = make_dataset(train_pairs, train_labels, batch_size=config["BATCH_SIZE"], augment=True)
val_dataset = make_dataset(val_pairs, val_labels, batch_size=config["BATCH_SIZE"], augment=False)

# Model Building - Initializes the Siamese neural network, loss function, and optimizer for training.

model = make_siamese_model()
loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(1e-4)

# Early stopping configuration 
patience = config["PATIENCE"]
min_delta = 0.001
best_val_loss = float("inf")
patience_counter = 0
max_batches_per_epoch = config["MAX_BATCHES_EPOCH"]

# Converting Python code to a fast, optimized TensorFlow graph for better training performance
@tf.function
def train_step(img1, img2, label):
    with tf.GradientTape() as tape:
        preds = model([img1, img2], training=True)
        loss = loss_fn(label, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, preds

@tf.function
def val_step(img1, img2, label):
    preds = model([img1, img2], training=False)
    loss = loss_fn(label, preds)
    return loss, preds

# Training loop
#Iteratively trains the model on batches and evaluates on validation data
#-Applies early stopping incase validation loss doesn't improve after a few epochs.
for epoch in range(config["EPOCHS"]):
    print(f"\nEpoch {epoch + 1}")
    train_loss, val_loss = 0.0, 0.0
    train_batches, val_batches = 0, 0
    all_train_preds, all_train_labels = [], []

    for step, (img1, img2, label) in enumerate(train_dataset):
        if step >= max_batches_per_epoch:
            break
        loss, preds = train_step(img1, img2, label)
        train_loss += loss.numpy()
        train_batches += 1
        all_train_preds.extend(tf.round(preds).numpy().flatten())
        all_train_labels.extend(label.numpy().flatten())

        if (step + 1) % 10 == 0:
            print(f"  Train Batch {step + 1}: Loss = {loss.numpy():.4f}")

    avg_train_loss = train_loss / train_batches

    all_val_preds, all_val_labels = [], []
    for img1, img2, label in val_dataset:
        v_loss, preds = val_step(img1, img2, label)
        val_loss += v_loss.numpy()
        val_batches += 1
        all_val_preds.extend(tf.round(preds).numpy().flatten())
        all_val_labels.extend(label.numpy().flatten())

    avg_val_loss = val_loss / val_batches

    print(f"Epoch {epoch + 1} Completed. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    if best_val_loss - avg_val_loss > min_delta:
        best_val_loss = avg_val_loss
        patience_counter = 0
        model.save(config["BEST_MODEL_PATH"])
        print("  Validation loss improved. Model saved.")
    else:
        patience_counter += 1
        print(f"  No improvement. Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print(" Early stopping triggered.")
            break

# Save final model
model.save(config["FINAL_MODEL_PATH"])
print(" Final model saved.")

# Evaluation on validation set
print("\n Final Evaluation on Validation Set:")
accuracy = accuracy_score(all_val_labels, all_val_preds)
precision = precision_score(all_val_labels, all_val_preds, zero_division=0)
recall = recall_score(all_val_labels, all_val_preds, zero_division=0)
f1 = f1_score(all_val_labels, all_val_preds, zero_division=0)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
