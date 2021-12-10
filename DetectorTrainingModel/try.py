import os

from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps

from tensorflow import keras
import numpy as np

from tensorflow.keras import layers

input_dir = "SintelTrailer_degraded/"
target_dir = "SintelTrailer_BinaryMask/"
img_size = (360, 640)
num_classes = 2
batch_size = 32

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".png")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

print("Number of samples:", len(input_img_paths))


class SintelTrailer(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size_, img_size_, input_img_paths_, target_img_paths_):
        self.batch_size = batch_size_
        self.img_size = img_size_
        self.input_img_paths = input_img_paths_
        self.target_img_paths = target_img_paths_

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i:i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i:i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            x[j] = np.expand_dims(img, 2)
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
        return x, y


def get_model(img_size_, num_classes_):
    inputs = keras.Input(shape=img_size_ + (1,))

    """ [First half of the network: downsampling inputs] """

    # Entry block
    x1 = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)

    x2 = layers.Conv2D(64, 2, padding="same", activation="relu")(x1)

    x3 = layers.Conv2D(128, 2, padding="same", activation="relu")(x2)

    encoder = layers.concatenate([x1, x2, x3])
    encoder = layers.Conv2D(64, 1, padding="same")(encoder)
    encoder = layers.MaxPooling2D(pool_size=(2, 2))(encoder)

    y1 = layers.UpSampling2D(2)(encoder)
    y1 = layers.Conv2DTranspose(32, 3, padding="same", activation="relu")(y1)

    y2 = layers.Conv2DTranspose(32, 3, padding="same", activation="relu")(y1)

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes_, 3, activation="softmax", padding="same")(y2)

    # Define the model
    model_ = keras.Model(inputs, outputs)
    return model_


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = get_model(img_size, num_classes)
model.summary()

import random

# Split our img paths into a training and a validation set
val_samples = 1000
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data Sequences for each split
train_gen = SintelTrailer(
    batch_size, img_size, train_input_img_paths, train_target_img_paths
)
val_gen = SintelTrailer(batch_size, img_size, val_input_img_paths, val_target_img_paths)

# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.
model.compile(optimizer="adam", loss="categorical_crossentropy")

callbacks = [
    keras.callbacks.ModelCheckpoint("SintelTrailer.h5", save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.
epochs = 15
model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
# from keras.models import load_model
# model = load_model('SintelTrailer.h5')
# Generate predictions for all images in the validation set
val_gen = SintelTrailer(batch_size, img_size, val_input_img_paths, val_target_img_paths)
val_preds = model.predict(val_gen)


def display_mask(i):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    return img


# Display results for validation image #10
i = 1

# Display input image
# img1 = Image(filename=val_input_img_paths[i])
print(f'Length: {len(val_input_img_paths)}')
print(val_input_img_paths[i])
# Display ground-truth target mask
img2 = PIL.ImageOps.autocontrast(load_img(val_target_img_paths[i]))

# Display mask predicted by our model
img3 = display_mask(i)  # Note that the model only sees inputs at 150x150.
img2.save('original.png')
img3.save('predict.png')

