import os
import sys
import PIL
import random
import numpy as np
from shutil import rmtree
from datetime import date
import matplotlib.pyplot as plt
import cv2 as cv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img

os.environ['DML_VISIBLE_DEVICES'] = '0'

# Define fold path
input_dir = "M:/MAI_dataset/tempSamples/degraded/"
target_dir = "M:/MAI_dataset/tempSamples/mask/"
valid_img_dir = 'M:/MAI_dataset/tempSamples/valid_set/frame/'
valid_mask_dir = "M:/MAI_dataset/tempSamples/valid_set/mask/"
test_img_dir = 'M:/MAI_dataset/tempSamples/test_set/cropped/'
test_mask_dir = "M:/MAI_dataset/tempSamples/mask/"
img_size = (180, 320)  # (273, 640)(360, 640)
num_classes = 2
batch_size = 2

date = date.today().strftime("%m-%d")
result_dir = f'M:/MAI_dataset/TrainedModels/{date}'
if os.path.isdir(result_dir) is False:
    os.mkdir(result_dir)


def load_dataset_path(input_dir_, target_dir_):
    print('Info: Reading dataset...')
    input_img_paths = sorted(
        [
            os.path.join(input_dir_, fname)
            for fname in os.listdir(input_dir_)
            if fname.endswith(".png")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(target_dir_, fname)
            for fname in os.listdir(target_dir_)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )
    print(f'Got dataset: {len(input_img_paths)} degraded images, {len(target_img_paths)} masks')
    return input_img_paths, target_img_paths


class ImageLoading(keras.utils.Sequence):
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
        x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")  # uint8
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            x[j] = np.expand_dims(img, 2)
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
        # print(x[300, 100], y[300, 100])
        return x, y


def get_model(img_size_, num_classes_):
    inputs = keras.Input(shape=img_size_ + (1,))  # img_size_ = [180, 320, 1]

    """ [First half of the network: downsampling inputs] """

    # Entry block
    x1 = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x1 = layers.BatchNormalization()(x1)

    x2 = layers.Conv2D(64, 2, padding="same", activation="relu")(x1)
    x2 = layers.BatchNormalization()(x2)

    x3 = layers.Conv2D(128, 2, padding="same", activation="relu")(x2)
    x3 = layers.BatchNormalization()(x3)

    encoder = layers.concatenate([x1, x2, x3])
    encoder = layers.Conv2D(64, 1, padding="same")(encoder)
    encoder = layers.MaxPooling2D(pool_size=(2, 2))(encoder)

    y1 = layers.UpSampling2D(2)(encoder)
    y1 = layers.Conv2DTranspose(32, 3, padding="same", activation="relu")(y1)
    y1 = layers.BatchNormalization()(y1)

    y2 = layers.Conv2DTranspose(32, 3, padding="same", activation="relu")(y1)
    y2 = layers.BatchNormalization()(y2)

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes_, 3, activation="softmax", padding="same")(y2)

    # Define the model
    model_ = keras.Model(inputs, outputs)
    return model_


def validation_split(input_img_paths, target_img_paths):
    global batch_size, img_size
    num_of_samples = len(input_img_paths)
    if num_of_samples <= 100:
        val_samples = num_of_samples
    else:
        val_samples = int(num_of_samples * 0.3)
    # random.Random(num_of_samples).shuffle(input_img_paths)
    # random.Random(num_of_samples).shuffle(target_img_paths)
    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]
    # Instantiate data Sequences for each split
    train_gen = ImageLoading(batch_size, img_size, train_input_img_paths, train_target_img_paths)
    val_gen = ImageLoading(batch_size, img_size, val_input_img_paths, val_target_img_paths)
    return train_gen, val_gen, val_input_img_paths, val_target_img_paths


def training(train_gen, val_gen, num_classes_, img_size_, use_pretrained, result_attempt_dir, test_gen):
    """Build model"""
    global result_dir
    model_path = f'{result_attempt_dir}/generalDegradedDetection.h5'
    if use_pretrained:
        model_path = 'M:/MAI_dataset/TrainedModels/02-14/Attempt 2/generalDegradedDetection.h5'
        print(f'INFO: Using pre-trained model from: {model_path}')
        model = keras.models.load_model(model_path, compile=False)
        test_preds = model.predict(test_gen)
        return test_preds
    else:
        # Build model
        model = get_model(img_size_, num_classes)
        model.summary()
        optimizer = tf.contrib.opt.AdamWOptimizer(learning_rate=0.01, weight_decay=0.0001)
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=["accuracy"]
                      )
        callbacks = [
            keras.callbacks.ModelCheckpoint(model_path, save_best_only=True)
        ]
        epochs = 50
        history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

        # list all data in history
        print(history.history.keys())

        # summarize history for accuracy
        fig1 = plt.figure(figsize=(8, 6))
        plt.title("Training history - Accuracy", fontsize=20)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f'{result_attempt_dir}/val_acc_plot.png', )

        # summarize history for loss
        fig2 = plt.figure(figsize=(8, 6))
        plt.title("Training history - Loss", fontsize=20)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.savefig(f'{result_attempt_dir}/val_loss_plot.png', )
        plt.show()

        test_preds = model.predict(test_gen)
        return test_preds


def convert_array_to_imgs(result_attempt_dir, input_degraded_img_path, ground_truth_mask_path, test_preds, if_truemask):
    """Quick utility to display a model's prediction."""

    if os.path.isdir(f'{result_attempt_dir}/mask_predictions'):
        rmtree(f'{result_attempt_dir}/mask_predictions')
    if os.path.isdir(f'{result_attempt_dir}/degraded'):
        rmtree(f'{result_attempt_dir}/degraded')
    os.mkdir(f'{result_attempt_dir}/mask_predictions')
    os.mkdir(f'{result_attempt_dir}/degraded')

    index = 0
    print(f'INFO: Predictions saved in the following path: {result_attempt_dir}/degraded/')
    for (degraded_img_path, mask_ori_path) in zip(input_degraded_img_path, ground_truth_mask_path):
        degraded_img = cv.imread(degraded_img_path)
        degraded_img = cv.resize(degraded_img, [320, 180])

        __mask = np.argmax(test_preds[index], axis=-1)
        __mask = np.expand_dims(__mask, axis=-1) * 255
        __mask = __mask * 255

        cv.imwrite(f'{result_attempt_dir}/degraded/img{(index + 1):03d}.png', degraded_img)
        cv.imwrite(f'{result_attempt_dir}/mask_predictions/pred{(index + 1):03d}.png', __mask)
        if if_truemask:
            mask_ori = cv.imread(mask_ori_path) * 255
            cv.imwrite(f'{result_attempt_dir}/mask_predictions/truth{(index + 1):03d}.png', mask_ori)
            cv.imshow(f'Mask_ori', cv.resize(mask_ori, [720, 360]))
            cv.moveWindow(f'Mask_ori', 2560, 360)

        mask_preds = cv.imread(f'{result_attempt_dir}/mask_predictions/pred{(index + 1):03d}.png', cv.IMREAD_GRAYSCALE)
        cv.imshow(f'Degraded frame', cv.resize(degraded_img, [1440, 720]))
        cv.imshow(f'Mask_preds', cv.resize(mask_preds, [1440, 720]))
        cv.moveWindow(f'Degraded frame', 2560, 0)
        cv.moveWindow(f'Mask_preds', 2560, 720)
        cv.waitKey(1)

        index += 1


def main(args):
    keras.backend.clear_session()

    attempts = 1

    input_img_paths, target_img_paths = load_dataset_path(input_dir, target_dir)
    valid_img, valid_mask = load_dataset_path(valid_img_dir, valid_mask_dir)
    test_img, test_mask = load_dataset_path(test_img_dir, test_mask_dir)

    train_gen, _, _, _ = validation_split(input_img_paths, target_img_paths)
    _, val_gen, _, _ = validation_split(input_img_paths, target_img_paths)
    _, test_gen, test_input_img_path, test_target_img_path = validation_split(test_img, test_mask)

    '''Free up RAM in case the model definition cells were run multiple times'''
    keras.backend.clear_session()
    if_reuse = input("Use pre-trained model? (Y/N)")
    if if_reuse.lower() == 'y':
        if_reuse = True
        result_attempt_dir = f'{result_dir}/Attempt {attempts}'
        if os.path.isdir(f'{result_attempt_dir}') is False:
            os.mkdir(f'{result_attempt_dir}')
    else:
        if_reuse = False
        if os.path.isdir(f'{result_dir}/Attempt {attempts}') is False:
            os.mkdir(f'{result_dir}/Attempt {attempts}')
            result_attempt_dir = f'{result_dir}/Attempt {attempts}'
        else:
            while os.path.isdir(f'{result_dir}/Attempt {attempts}') is True:
                if len(os.listdir(f'{result_dir}/Attempt {attempts}')) <= 1:
                    rmtree(f'{result_dir}/Attempt {attempts}')
                    break
                attempts += 1
            os.mkdir(f'{result_dir}/Attempt {attempts}')
            result_attempt_dir = f'{result_dir}/Attempt {attempts}'

    test_preds = training(train_gen, val_gen, num_classes, img_size, if_reuse, result_attempt_dir, test_gen)

    convert_array_to_imgs(result_attempt_dir, test_input_img_path, test_target_img_path, test_preds, False)


if __name__ == '__main__':
    main(sys.argv)
