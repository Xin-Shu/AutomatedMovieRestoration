import os
import sys

import cv2 as cv
import numpy as np
from tqdm import tqdm
from shutil import rmtree


def MaskCleaner():
    pass

def main(args):
    date_ = input("Date of training results: ")
    attempt_ = input(f"Attempt number on {date_}: ")
    degraded_frame_folder = f'M:/MAI_dataset/TrainedModels/{date_}/Attempt {attempt_}/degraded/'
    predicted_mask_folder = f'M:/MAI_dataset/TrainedModels/{date_}/Attempt {attempt_}/mask/'
    out_folder = f'M:/MAI_dataset/TrainedModels/{date_}/Attempt {attempt_}/cleaned_mask/'

    if os.path.isdir(out_folder):
        rmtree(out_folder)
        os.mkdir(out_folder)


if __name__ == '__main__':
    main(sys.argv)
