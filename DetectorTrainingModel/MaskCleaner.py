import os
import sys

import cv2 as cv
import numpy as np
from tqdm import tqdm
from shutil import rmtree


def MaskCleaner(maskIN_PATH, maskOUT_PATH, if_remake):

    if if_remake:
        rmtree(maskOUT_PATH)
        os.mkdir(maskOUT_PATH)

    if os.path.isdir(maskIN_PATH) is False or os.path.isdir(maskOUT_PATH) is False:
        import warnings
        errorMessage = f'Error: One of the following folders does not exist: {maskIN_PATH}; \n {maskOUT_PATH:>20}'
        warnings.simplefilter(errorMessage)
        exit()

    maskIN = sorted([
        os.path.join(maskIN_PATH, fname)
        for fname in os.listdir(maskIN_PATH)
        if fname.endswith(".png")]
    )

    for i in tqdm(range(0, len(maskIN)), bar_format='{percentage:3.0f}%|{bar:100}{r_bar}'):

        mask_ori = cv.imread(maskIN[i], cv.IMREAD_GRAYSCALE)
        mask_shape = mask_ori.shape                     # e.g., (180, 320)

        maskIN_1 = mask_ori[0:int(mask_shape[0] / 3), :]
        maskIN_2 = mask_ori[int(mask_shape[0] / 3):int(mask_shape[0] / 3 * 2), :]
        maskIN_3 = mask_ori[int(mask_shape[0] / 3 * 2):mask_shape[0], :]

        sum_1, sum_2, sum_3 = maskIN_1.sum(axis=0) / 255, maskIN_2.sum(axis=0) / 255, maskIN_3.sum(axis=0) / 255

        energy_of_current_frame = sum_1 * 1 + sum_2 * 4 + sum_3 * 1
        print(energy_of_current_frame)

        maskOUT = np.zeros(mask_shape, dtype="uint8")
        threshold = 10

        for col in range(0, len(energy_of_current_frame)):

            if energy_of_current_frame[col] >= threshold:
                maskOUT[:, col] = 255
            else:
                maskOUT[:, col] = 0

            cv.imwrite(f'{maskOUT_PATH}/{os.path.basename(maskIN[i])}', maskOUT)
            

def main(args):
    # date_ = input("Date of training results: ")
    # attempt_ = input(f"Attempt number on {date_}: ")
    date_ = '03-20'
    attempt_ = '4'

    degraded_frame_folder = f'M:/MAI_dataset/TrainedModels/{date_}/Attempt {attempt_}/degraded/'
    predicted_mask_folder = f'M:/MAI_dataset/TrainedModels/{date_}/Attempt {attempt_}/mask/'
    out_folder = f'M:/MAI_dataset/TrainedModels/{date_}/Attempt {attempt_}/cleaned_mask/'

    if os.path.isdir(out_folder):
        rmtree(out_folder)
    os.mkdir(out_folder)
    MaskCleaner(predicted_mask_folder, out_folder, 1)


if __name__ == '__main__':
    main(sys.argv)
