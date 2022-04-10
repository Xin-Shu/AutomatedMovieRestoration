import os
import sys

import cv2 as cv
import numpy as np
from tqdm import tqdm
from shutil import rmtree
from datetime import date
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

from CropsExtraction import crop_img
from main import ImageLoading

# Define paths
MODEL_PATH = 'M:/MAI_dataset/TrainedModels/04-09/Attempt 1/generalDegradedDetection.h5'

# Define 'Carrier'
Carrier_InPath = 'M:/MAI_dataset/Sequence_lines_1/Carrier/'
Carrier_OutPath = 'M:/MAI_dataset/Sequence_lines_1/Carrier_testset/'
Carrier_PredPath = 'M:/MAI_dataset/Sequence_lines_1/Carrier_pred/'
Carrier_InSize = (1920, 1080)
Carrier_type = 'tif'
Carrier_numFrames = 100

# Define 'Cinecitta'
Cinecitta_InPath = 'M:/MAI_dataset/Sequence_lines_1/Cinecitta/'
Cinecitta_OutPath = 'M:/MAI_dataset/Sequence_lines_1/Cinecitta_testset/'
Cinecitta_PredPath = 'M:/MAI_dataset/Sequence_lines_1/Cinecitta_pred/'
Cinecitta_InSize = (1828, 1332)
Cinecitta_type = 'bmp'
Cinecitta_numFrames = 51

# Define 'scratchTest'
scratchTest_InPath = 'M:/MAI_dataset/Sequence_lines_1/scratchTest/'
scratchTest_OutPath = 'M:/MAI_dataset/Sequence_lines_1/scratchTest_testset/'
scratchTest_PredPath = 'M:/MAI_dataset/Sequence_lines_1/scratchTest_pred/'
scratchTest_InSize = (720, 486)
scratchTest_type = 'tif'
scratchTest_numFrames = 100

# General information
OUTSIZE = (320, 180)
BATCH_SIZE = 8
BIGIMG_THRE = 10   # Maximum value should be the height of the input image, and need to be integer multiples of 60.
TILE_THRE = 5       # Maximum value should be the height of the tiling image, which is 60 in this case


class ReloadImages(keras.utils.Sequence):

    def __init__(self, path, batchSize, imgSize):

        self.path = sorted([
                os.path.join(path, fname)
                for fname in os.listdir(path)
                if fname.endswith(".png")
            ])
        self.batchSize = batchSize
        self.imgSize = (imgSize[1], imgSize[0])

    def __len__(self):
        return len(self.path) // self.batchSize

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batchSize
        batch_input_img_paths = self.path[i:i + self.batchSize]
        x = np.zeros((self.batchSize,) + self.imgSize + (1,), dtype="uint8")  # uint8
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.imgSize, color_mode="grayscale")
            x[j] = np.expand_dims(img, 2)
        return x


class MakePredictions:

    def __init__(self, modelPath_, imgSeq, predOutPath, baseNamePath, outSize_, oriFrameFolder, oriSize, oriType):
        self.model = keras.models.load_model(modelPath_, compile=False)
        self.sequence = imgSeq
        self.outPath = predOutPath
        self.rawPath = f'{self.outPath}Raw Mask/'
        self.prediction = None
        self.outSize = outSize_
        self.oriPaths = sorted([
            os.path.join(oriFrameFolder, fname)
            for fname in os.listdir(oriFrameFolder)
            if fname.endswith(f".{oriType}")
        ])
        self.baseName = sorted([
                os.path.join(baseNamePath, fname)
                for fname in os.listdir(baseNamePath)
                if fname.endswith(".png")
            ])
        self.oriSize = oriSize
        self.oriType = oriType
        self.predict()
        self.assembleMask()

    def predict(self):
        self.prediction = self.model.predict(self.sequence)
        os.mkdir(self.rawPath)

        __index = 0
        for pred in tqdm(self.prediction, bar_format='{percentage:3.0f}%|{bar:100}{r_bar}'):
            __mask = np.argmax(pred, axis=-1)
            __mask = np.expand_dims(__mask, axis=-1)
            __mask = __mask * 255
            cv.imwrite(f'{self.rawPath}{os.path.basename(self.baseName[__index])}', __mask)

            __index += 1

    def assembleMask(self):

        maskAssembled = np.zeros([self.oriSize[1], self.oriSize[0]], dtype="uint8")
        maskWidth, maskHeight = self.outSize[0], int(self.outSize[1] / 3)

        maskFrameNum_mark = 1

        frameIN = self.oriPaths
        maskIN = sorted([
            os.path.join(self.rawPath, fname)
            for fname in os.listdir(self.rawPath)
            if fname.endswith(".png")]
        )

        for i in tqdm(range(0, len(maskIN)), bar_format='{percentage:3.0f}%|{bar:100}{r_bar}'):

            mask_ori = cv.imread(maskIN[i], cv.IMREAD_GRAYSCALE)
            mask_shape = mask_ori.shape  # e.g., (180, 320)

            sumMask = mask_ori.sum(axis=0)

            maskOUT = np.zeros([mask_shape[0] // 3, mask_shape[1]], dtype="uint8")

            # for col in range(0, len(sumMask)):
            #
            #     if sumMask[col] >= TILE_THRE:
            #         maskOUT[:, col] = 255
            #     else:
            #         maskOUT[:, col] = 0

            maskOUT = mask_ori[60:120, :]

            frameName = os.path.basename(maskIN[i])
            maskFrameNum, maskColNum, maskRowNum = int(frameName[5:8]), int(frameName[9:12]), int(frameName[13:16])
            topLeft_x, topLeft_y = maskWidth * (maskColNum - 1), maskHeight * (maskRowNum - 1)

            if maskAssembled[topLeft_y:topLeft_y + maskHeight, topLeft_x:topLeft_x + maskWidth].shape == maskOUT.shape:
                maskAssembled[topLeft_y:topLeft_y + maskHeight, topLeft_x:topLeft_x + maskWidth] = maskOUT

            else:
                tempMaskShape = maskAssembled[topLeft_y:topLeft_y + maskHeight, topLeft_x:topLeft_x + maskWidth].shape

                if tempMaskShape[0] != 0 and tempMaskShape[1] != 0:
                    maskAssembled[topLeft_y:topLeft_y + maskHeight, topLeft_x:topLeft_x + maskWidth] = \
                        maskOUT[-tempMaskShape[0]:, -tempMaskShape[1]:]

            if maskFrameNum_mark != maskFrameNum:
                frame_ori = cv.imread(frameIN[maskFrameNum_mark], cv.IMREAD_GRAYSCALE)
                cleanedOverlay = cv.cvtColor(frame_ori, cv.COLOR_GRAY2RGB)

                maskEnergy = maskAssembled.sum(axis=0) / 255
                # for col in range(0, len(maskEnergy)):
                #
                #     if maskEnergy[col] >= BIGIMG_THRE:
                #         maskAssembled[:, col] = 255
                #     else:
                #         maskAssembled[:, col] = 0

                cleanedOverlay[:, :, 1] = np.clip((cleanedOverlay[:, :, 1] - maskAssembled / 255 * 230), 0.0, 255.0)
                cv.imwrite(f'{self.outPath}/{os.path.basename(frameIN[maskFrameNum_mark])}', cleanedOverlay)
                cv.imwrite(f'{self.outPath}/{os.path.basename(maskIN[i])}', maskAssembled)
                maskFrameNum_mark = maskFrameNum


def main(inSize_, inPath_, outPath_, predPath_, numFrames_, filmType):

    reset = True
    if reset is True:
        if os.path.isdir(outPath_) or os.path.isdir(predPath_):
            rmtree(outPath_)
            os.mkdir(outPath_)
            rmtree(predPath_)
            os.mkdir(predPath_)
        crop_img(inPath_, outPath_, OUTSIZE, inSize_, filmType, numFrames_)

    __imgSequence = ReloadImages(outPath_, BATCH_SIZE, OUTSIZE)
    pred_ = MakePredictions(MODEL_PATH, __imgSequence, predPath_, outPath_, OUTSIZE,
                            inPath_, inSize_, filmType)


if __name__ == '__main__':
    # main(Carrier_InSize, Carrier_InPath, Carrier_OutPath, Carrier_PredPath, Carrier_numFrames, Carrier_type)
    # main(Cinecitta_InSize, Cinecitta_InPath, Cinecitta_OutPath,
    #      Cinecitta_PredPath, Cinecitta_numFrames, Cinecitta_type)
    main(scratchTest_InSize, scratchTest_InPath, scratchTest_OutPath,
         scratchTest_PredPath, scratchTest_numFrames, scratchTest_type)
