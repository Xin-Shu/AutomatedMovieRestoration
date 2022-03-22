"""
This

FOR USAGE:
    type in date and attempt number
"""
import os
import sys
import cv2 as cv
import numpy as np
from tqdm import tqdm
from shutil import rmtree

in_size = (320, 180)
display_size = (720, 405)


def draw_mask_over_degraded_frame(frame_folder, mask_folder, out_folder_):

    if os.path.isdir(frame_folder) is False or os.path.isdir(mask_folder) is False:
        import warnings
        errorMessage = f'Error: One of the following folders does not exist: {frame_folder}; \n {mask_folder}'
        warnings.simplefilter(errorMessage)
        exit()

    frame_path = sorted([
        os.path.join(frame_folder, fname)
        for fname in os.listdir(frame_folder)
        if fname.endswith(".png")]
    )
    mask_path = sorted([
        os.path.join(mask_folder, fname)
        for fname in os.listdir(mask_folder)
        if fname.endswith(".png")]
    )

    if len(frame_path) != len(mask_path):
        print(f'Warning: the number of files are not the same, {len(frame_path)} frames while {len(mask_path)} masks.')

    for i in tqdm(reversed(range(0, len(mask_path))), bar_format='{percentage:3.0f}%|{bar:100}{r_bar}'):
        frame = cv.imread(frame_path[i], cv.IMREAD_GRAYSCALE)
        mask = cv.imread(mask_path[i], cv.IMREAD_GRAYSCALE)
        frame = cv.resize(frame, in_size)
        mask = cv.resize(mask, in_size)

        mask_over_ori = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
        mask_over_ori[:, :, 0] = np.clip((mask_over_ori[:, :, 0] - mask / 255 * 150), 0.0, 255.0)
        mask_over_ori[:, :, 1] = np.clip((mask_over_ori[:, :, 1] - mask / 255 * 150), 0.0, 255.0)
        mask_over_ori[:, :, 2] = np.clip((mask_over_ori[:, :, 2] + mask / 255 * 150), 0.0, 255.0)

        # cv.imshow('frame', cv.resize(frame, display_size))
        # cv.imshow('mask', cv.resize(mask, display_size))
        # cv.imshow('mask_over_ori', cv.resize(mask_over_ori, display_size))
        # cv.moveWindow('frame', int(- display_size[0] * 2.5 + 10), 10)
        # cv.moveWindow('mask', int(- display_size[0] * 1.5 + 10), 10)
        # cv.moveWindow('mask_over_ori', int(- display_size[0] * 2.5 + 10), 10 + display_size[1])

        cv.imwrite(f'{out_folder_}/{os.path.basename(frame_path[i])}', mask_over_ori)

        # cv.waitKey(1)


def main(args):
    date_ = input("Date of training results: ")
    attempt_ = input(f"Attempt number on {date_}: ")
    degraded_frame_folder = f'M:/MAI_dataset/TrainedModels/{date_}/Attempt {attempt_}/degraded/'
    predicted_mask_folder = f'M:/MAI_dataset/TrainedModels/{date_}/Attempt {attempt_}/mask/'
    out_folder = f'M:/MAI_dataset/TrainedModels/{date_}/Attempt {attempt_}/mask_over_frame/'
    if os.path.isdir(out_folder):
        rmtree(out_folder)
    os.mkdir(out_folder)
    draw_mask_over_degraded_frame(degraded_frame_folder, predicted_mask_folder, out_folder)
    # print(os.path.isdir(degraded_frame_folder))
    # print(os.path.isdir(predicted_mask_folder))


if __name__ == '__main__':
    main(sys.argv)


