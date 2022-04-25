import os
import sys

import cv2 as cv
import numpy as np
from tqdm import tqdm
from shutil import rmtree

from CropsExtraction import ori_size
from DatasetPerparation import output_size


def MaskCleaner(frameIN_PATH, maskIN_PATH, frameOUT_PATH, maskOUT_PATH, maskEntireOUT_PATH, if_remake):

    if if_remake:
        rmtree(frameOUT_PATH)
        os.mkdir(frameOUT_PATH)
        rmtree(maskOUT_PATH)
        os.mkdir(maskOUT_PATH)
        rmtree(maskEntireOUT_PATH)
        os.mkdir(maskEntireOUT_PATH)

    if os.path.isdir(maskIN_PATH) is False or os.path.isdir(maskOUT_PATH) is False:
        import warnings
        errorMessage = f'Error: One of the following folders does not exist: {maskIN_PATH}; \n {maskOUT_PATH:>20}'
        warnings.simplefilter(errorMessage)
        exit()

    frameIN = sorted([
        os.path.join(frameIN_PATH, fname)
        for fname in os.listdir(frameIN_PATH)
        if fname.endswith(".bmp")]
    )

    maskIN = sorted([
        os.path.join(maskIN_PATH, fname)
        for fname in os.listdir(maskIN_PATH)
        if fname.endswith(".png")]
    )

    # Define a zero matrix that has the same shape as the degraded frame
    maskAssembled = np.zeros([ori_size[1], ori_size[0]], dtype="uint8")
    maskWidth, maskHeight = output_size[0], int(output_size[1] / 3),

    maskFrameNum_mark = 1

    # for i in tqdm(range(0, len(maskIN)), bar_format='{percentage:3.0f}%|{bar:100}{r_bar}'):
    for i in range(0, len(maskIN)):

        mask_ori = cv.imread(maskIN[i], cv.IMREAD_GRAYSCALE)
        mask_shape = mask_ori.shape                     # e.g., (180, 320)

        maskIN_1 = mask_ori[0:int(mask_shape[0] / 3), :]
        maskIN_2 = mask_ori[int(mask_shape[0] / 3):int(mask_shape[0] / 3 * 2), :]
        maskIN_3 = mask_ori[int(mask_shape[0] / 3 * 2):mask_shape[0], :]

        sum_1, sum_2, sum_3 = maskIN_1.sum(axis=0), maskIN_2.sum(axis=0), maskIN_3.sum(axis=0)

        energy_of_current_frame = sum_1 * 0.25 + sum_2 * 0.5 + sum_3 * 0.25

        maskOUT = np.zeros([mask_shape[0] // 3, mask_shape[1]], dtype="uint8")
        threshold = 200.0

        for col in range(0, len(energy_of_current_frame)):

            if energy_of_current_frame[col] >= threshold:
                maskOUT[:, col] = 255
            else:
                maskOUT[:, col] = 0

        # threshold = 128.0 / 2
        # maskOUT = maskIN_2 * 0.5 + (maskIN_1 + maskIN_3) * 0.25
        maskOUT[maskOUT > threshold] = 255
        maskOUT[maskOUT <= threshold] = 0

        frameName = os.path.basename(maskIN[i])
        maskFrameNum, maskColNum, maskRowNum = int(frameName[5:8]), int(frameName[9:12]), int(frameName[13:16])

        topLeft_x, topLeft_y = maskWidth * (maskColNum - 1), maskHeight * (maskRowNum - 1)
        # print(f'maskFrameNum: {maskFrameNum}, maskColNum: {maskColNum}, maskRowNum: {maskRowNum}')
        # print(f'topLeft_x: {topLeft_x}， topLeft_y： {topLeft_y}')

        if maskAssembled[topLeft_y:topLeft_y + maskHeight, topLeft_x:topLeft_x + maskWidth].shape == maskOUT.shape:

            maskAssembled[topLeft_y:topLeft_y + maskHeight, topLeft_x:topLeft_x + maskWidth] = maskOUT
        else:
            tempMaskShape = maskAssembled[topLeft_y:topLeft_y + maskHeight, topLeft_x:topLeft_x + maskWidth].shape
            maskAssembled[topLeft_y:topLeft_y + maskHeight, topLeft_x:topLeft_x + maskWidth] = \
                maskOUT[-tempMaskShape[0]:, -tempMaskShape[1]:]

        if maskFrameNum_mark != maskFrameNum:

            frame_ori = cv.imread(frameIN[maskFrameNum_mark], cv.IMREAD_GRAYSCALE)
            cleanedOverlay = cv.cvtColor(frame_ori, cv.COLOR_GRAY2RGB)

            # threshold_EntireMask = 15
            # maskEnergy = maskAssembled.sum(axis=0) / 255
            # for col in range(0, len(maskEnergy)):
            #
            #     if maskEnergy[col] >= threshold_EntireMask:
            #         maskAssembled[:, col] = 255
            #     else:
            #         maskAssembled[:, col] = 0

            cleanedOverlay[:, :, 1] = np.clip((cleanedOverlay[:, :, 1] - maskAssembled / 255 * 230), 0.0, 255.0)
            cv.imwrite(f'{maskEntireOUT_PATH}/{os.path.basename(frameIN[maskFrameNum_mark])}', cleanedOverlay)
            cv.imwrite(f'{maskEntireOUT_PATH}/{os.path.basename(maskIN[i])}', maskAssembled)
            maskFrameNum_mark = maskFrameNum

        # cleanedOverlay = cv.cvtColor(frame_ori, cv.COLOR_GRAY2RGB)
        # # cleanedOverlay[:, :, 0] = np.clip((cleanedOverlay[:, :, 0] - maskOUT * 150), 0.0, 255.0)
        # cleanedOverlay[int(mask_shape[0] / 3):int(mask_shape[0] / 3 * 2), :, 1] = \
        #     np.clip((cleanedOverlay[int(mask_shape[0] / 3):int(mask_shape[0] / 3 * 2), :, 1] - maskOUT / 255 * 150),
        #             0.0, 255.0)
        # # cleanedOverlay[:, :, 2] = np.clip((cleanedOverlay[:, :, 2] + maskOUT * 150), 0.0, 255.0)
        #
        # cv.imwrite(f'{frameOUT_PATH}/{os.path.basename(maskIN[i])}', cleanedOverlay)
        # cv.imwrite(f'{maskOUT_PATH}/{os.path.basename(maskIN[i])}', maskOUT)


def main(args):

    date_ = input("Date of training results: ")
    attempt_ = input(f"Attempt number on {date_}: ")
    # date_ = '03-23'
    # attempt_ = '1'

    degraded_frame_folder = f'M:/MAI_dataset/Cinecitta/Sequence_lines_1/'
    predicted_mask_folder = f'M:/MAI_dataset/TrainedModels/{date_}/Attempt {attempt_}/mask/'
    frameOut_folder = f'M:/MAI_dataset/TrainedModels/{date_}/Attempt {attempt_}/cleaned_mask_over_frame/'
    maskOut_folder = f'M:/MAI_dataset/TrainedModels/{date_}/Attempt {attempt_}/cleaned_mask/'
    maskEntireOut_folder = f'M:/MAI_dataset/TrainedModels/{date_}/Attempt {attempt_}/entire_cleaned_mask/'

    if os.path.isdir(maskOut_folder):
        rmtree(maskOut_folder)
        rmtree(frameOut_folder)
        rmtree(maskEntireOut_folder)
    os.mkdir(frameOut_folder)
    os.mkdir(maskOut_folder)
    os.mkdir(maskEntireOut_folder)
    MaskCleaner(degraded_frame_folder, predicted_mask_folder, frameOut_folder, maskOut_folder, maskEntireOut_folder, 1)


if __name__ == '__main__':
    main(sys.argv)
