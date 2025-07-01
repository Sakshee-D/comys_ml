from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Layer
import tensorflow as tf

# Custom layer to compute L1 (absolute) distance between two embeddings
class L1Dist(Layer):
    def call(self, emb1, emb2):
        return tf.math.abs(emb1 - emb2)  # Element-wise absolute difference


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
    l1 = L1Dist()(emb1, emb2)

    # Pass the distance through a dense layer to get similarity score (0 to 1)
    out = Dense(1, activation='sigmoid')(l1)

    # Final Siamese model takes two images and returns similarity
    return Model(inputs=[input_img, val_img], outputs=out)


