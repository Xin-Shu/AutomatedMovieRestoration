import os
import sys
import datetime
import cv2 as cv
from tqdm import tqdm

'''
This script is for evaluation of artifically degraded frames that used to generate vertiall assembled dataset.
The evaluation methods include:
    Calculate the Peak Signal to Nosie Ratio (PSNR) of degraded frames with original frame.
'''

ORI_ST_FOLDER = 'M:/MAI_dataset/Origin_set/ST-sample/'
ORI_BBB_FOLDER = 'M:/MAI_dataset/Origin_set/BBB-sample/'
ORI_ED_FOLDER = 'M:/MAI_dataset/Origin_set/ED-sample/'
ORI_TOS_FOLDER = 'M:/MAI_dataset/Origin_set/TOS-sample/'

DEG_ST_FOLDER = 'M:/MAI_dataset/Degraded_set/SAMPLE-ST/frame/'
DEG_BBB_FOLDER = 'M:/MAI_dataset/Degraded_set/SAMPLE-BBB/frame/'
DEG_ED_FOLDER = 'M:/MAI_dataset/Degraded_set/SAMPLE-ED/frame/'
DEG_TOS_FOLDER = 'M:/MAI_dataset/Degraded_set/SAMPLE-TOS/frame/'


def getPSNR(originalFolderPath, DegradedFolderPath):

    if os.path.isdir(originalFolderPath) is False or os.path.isdir(DegradedFolderPath) is False:
        print(f'ERROR: (FATAL) The request folder path {originalFolderPath} or {DegradedFolderPath} does not exist.')
        print(f'ERROR: Quitting...')
        exit()

    oriFramePaths = sorted([
        os.path.join(originalFolderPath, fname)
        for fname in os.listdir(originalFolderPath)
        if fname.endswith(".png")]
    )

    degFramePaths = sorted([
        os.path.join(DegradedFolderPath, fname)
        for fname in os.listdir(DegradedFolderPath)
        if fname.endswith(".png")]
    )

    if len(oriFramePaths) != len(degFramePaths):

        print(f'WARNING: The number of original frames is different from degraded frames.'
              f'{len(oriFramePaths)} != {len(degFramePaths)}')

    psnrArr = []
    for i in tqdm(range(0, len(oriFramePaths) - 1), bar_format='{percentage:3.0f}%|{bar:100}{r_bar}'):

        oriFrame = cv.imread(oriFramePaths[i], cv.IMREAD_GRAYSCALE)
        degFrame = cv.imread(degFramePaths[i], cv.IMREAD_GRAYSCALE)
        psnrTemp = cv.PSNR(oriFrame, degFrame)
        psnrArr.append(psnrTemp)

    print(f'INFO: The average value of PSNR for {os.path.dirname(originalFolderPath)} sequence is '
          f'{sum(psnrArr) / len(psnrArr)} dB.')


def main():
    getPSNR(ORI_ST_FOLDER, DEG_ST_FOLDER)
    getPSNR(ORI_BBB_FOLDER, DEG_BBB_FOLDER)
    getPSNR(ORI_ED_FOLDER, DEG_ED_FOLDER)
    getPSNR(ORI_TOS_FOLDER, DEG_TOS_FOLDER)


if __name__ == '__main__':
    main()
