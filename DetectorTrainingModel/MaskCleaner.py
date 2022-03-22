import os
import sys

import cv2 as cv
import numpy as np
from tqdm import tqdm
from shutil import rmtree


def MaskCleaner(frameIN_PATH, maskIN_PATH, frameOUT_PATH, maskOUT_PATH, if_remake):

    if if_remake:
        rmtree(frameOUT_PATH)
        os.mkdir(frameOUT_PATH)
        rmtree(maskOUT_PATH)
        os.mkdir(maskOUT_PATH)

    if os.path.isdir(maskIN_PATH) is False or os.path.isdir(maskOUT_PATH) is False:
        import warnings
        errorMessage = f'Error: One of the following folders does not exist: {maskIN_PATH}; \n {maskOUT_PATH:>20}'
        warnings.simplefilter(errorMessage)
        exit()

    frameIN = sorted([
        os.path.join(frameIN_PATH, fname)
        for fname in os.listdir(frameIN_PATH)
        if fname.endswith(".png")]
    )

    maskIN = sorted([
        os.path.join(maskIN_PATH, fname)
        for fname in os.listdir(maskIN_PATH)
        if fname.endswith(".png")]
    )

    for i in tqdm(range(0, len(maskIN)), bar_format='{percentage:3.0f}%|{bar:100}{r_bar}'):

        frame_ori = cv.imread(frameIN[i], cv.IMREAD_GRAYSCALE)

        mask_ori = cv.imread(maskIN[i], cv.IMREAD_GRAYSCALE)
        mask_shape = mask_ori.shape                     # e.g., (180, 320)

        maskIN_1 = mask_ori[0:int(mask_shape[0] / 3), :]
        maskIN_2 = mask_ori[int(mask_shape[0] / 3):int(mask_shape[0] / 3 * 2), :]
        maskIN_3 = mask_ori[int(mask_shape[0] / 3 * 2):mask_shape[0], :]

        sum_1, sum_2, sum_3 = maskIN_1.sum(axis=0) / 255, maskIN_2.sum(axis=0) / 255, maskIN_3.sum(axis=0) / 255

        energy_of_current_frame = sum_1 * 1 + sum_2 * 4 + sum_3 * 1

        maskOUT = np.zeros([mask_shape[0] // 3, mask_shape[1]], dtype="uint8")
        threshold = 10

        for col in range(0, len(energy_of_current_frame)):

            if energy_of_current_frame[col] >= threshold:
                maskOUT[:, col] = 255
            else:
                maskOUT[:, col] = 0

        cleanedOverlay = cv.cvtColor(frame_ori, cv.COLOR_GRAY2RGB)
        # cleanedOverlay[:, :, 0] = np.clip((cleanedOverlay[:, :, 0] - maskOUT * 150), 0.0, 255.0)
        cleanedOverlay[int(mask_shape[0] / 3):int(mask_shape[0] / 3 * 2), :, 1] = \
            np.clip((cleanedOverlay[int(mask_shape[0] / 3):int(mask_shape[0] / 3 * 2), :, 1] - maskOUT / 255 * 150),
                    0.0, 255.0)
        # cleanedOverlay[:, :, 2] = np.clip((cleanedOverlay[:, :, 2] + maskOUT * 150), 0.0, 255.0)

        cv.imwrite(f'{frameOUT_PATH}/{os.path.basename(maskIN[i])}', cleanedOverlay)
        cv.imwrite(f'{maskOUT_PATH}/{os.path.basename(maskIN[i])}', maskOUT)


def main(args):

    date_ = input("Date of training results: ")
    attempt_ = input(f"Attempt number on {date_}: ")

    degraded_frame_folder = f'M:/MAI_dataset/TrainedModels/{date_}/Attempt {attempt_}/degraded/'
    predicted_mask_folder = f'M:/MAI_dataset/TrainedModels/{date_}/Attempt {attempt_}/mask/'
    frameOut_folder = f'M:/MAI_dataset/TrainedModels/{date_}/Attempt {attempt_}/cleaned_mask_over_frame/'
    maskOut_folder = f'M:/MAI_dataset/TrainedModels/{date_}/Attempt {attempt_}/cleaned_mask/'

    if os.path.isdir(maskOut_folder):
        rmtree(maskOut_folder)
        rmtree(frameOut_folder)
    os.mkdir(frameOut_folder)
    os.mkdir(maskOut_folder)
    MaskCleaner(degraded_frame_folder, predicted_mask_folder, frameOut_folder, maskOut_folder, 1)


if __name__ == '__main__':
    main(sys.argv)
