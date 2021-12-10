import os
import sys
import PIL
import random
import numpy as np
from PIL import ImageOps
from IPython.display import Image, display

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img

# Define fold path
DEFRADED_DIR = 'SintelTrailer_degraded/'
MASK_DIR = 'SintelTrailer_BinaryMask/'

input_img_paths = ''
target_img_paths = ''

image_size = (160, 160)
num_classes = 2
batch_size = 32


def load_dataset():
    global input_img_paths, target_img_paths
    print('Info: Reading dataset...')
    input_img_paths = sorted(
        [
            os.path.join(DEFRADED_DIR, fname)
            for fname in os.listdir(DEFRADED_DIR)
            if fname.endswith(".png")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(MASK_DIR, fname)
            for fname in os.listdir(MASK_DIR)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )
    print(f'Got dataset: {0} degraded images, {1} masks'.format(len(input_img_paths), len(target_img_paths)))


class SintelTrailerDataLoader(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size_, image_size_, input_img_paths_, target_img_paths_):
        self.batch_size = batch_size_
        self.image_size = image_size_
        self.input_img_paths = input_img_paths_
        self.target_img_paths = target_img_paths_

    def __len__(self):
        return len(self.input_img_paths)

    def __getimage__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i:1 + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i:1 + self.batch_size]
        x = np.zeros((self.batch_size,) + self.image_size + (1,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.image_size, color_mode="grayscale")
            x[j] = np.expand_dims(img, 2)
        y = np.zeros((self.batch_size,) + self.image_size + (1,), dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.image_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
        return x, y


def get_model(img_size, num_class):
    inputs = keras.Input(shape=img_size + (1,))

    """[First half of the network: downsampling inputs]"""

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    '''[Second half of the network: upsampling inputs]'''

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_class, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


def validation_split():
    global batch_size, image_size, input_img_paths, target_img_paths
    val_samples = 200
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)
    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]
    # Instantiate data Sequences for each split
    train_gen = SintelTrailerDataLoader(
        batch_size, image_size, train_input_img_paths, train_target_img_paths
    )
    val_gen = SintelTrailerDataLoader(batch_size, image_size, val_input_img_paths, val_target_img_paths)
    return train_gen, val_gen, val_input_img_paths, val_target_img_paths


def display_mask(index, pred_mask):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(pred_mask[index], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    display(img)


def training(trainset, valset, if_reuse, modelpath):
    """Build model"""
    global image_size, num_classes
    if if_reuse:
        model = keras.models.load_model("modelpath")
    else:
        model = get_model(image_size, num_classes)
        model.summary()
        model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
        callbacks = [
            keras.callbacks.ModelCheckpoint("SintekTrailer.h5", save_best_only=True)
        ]
        epochs = 15
        model.fit(trainset, epochs=epochs, validation_data=valset, callbacks=callbacks)

    _, test_set, test_input_path, test_target_path = validation_split()

    pred = model.predict(test_set)
    _index = random.randint(len(pred))
    display(Image(filename=test_input_path[_index]))
    display(PIL.ImageOps.autocontrast(load_img(test_target_path[_index])))
    display_mask(_index, pred)


def main(args):
    load_dataset()
    train_set, val_set, _, _ = validation_split()

    '''Free up RAM in case the model definition cells were run multiple times'''
    # keras.backend.clear_session()
    pre_trained_model_path = 'SintekTrailer.h5'
    # training(train_set, val_set, False, pre_trained_model_path)
    model = get_model(image_size, num_classes)
    model.summary()
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
    callbacks = [
        keras.callbacks.ModelCheckpoint("SintekTrailer.h5", save_best_only=True)
    ]
    epochs = 15
    model.fit(train_set, epochs=epochs, validation_data=val_set, callbacks=callbacks)


if __name__ == '__main__':
    main(sys.argv)
