import os
import sys
import glob

import cv2 as cv
import numpy as np
from tqdm import tqdm
from shutil import rmtree
''' 
This script is for extracting small crops from a large image, e.g. 1280×720 and higher resolutions.
The aim of adding this script is to make better predictions over large input images for vertical scratch detections,
    since the trained DNN takes 256x144 images as input.
'''
'''
The output frames are in resolution 320 * 180 and 8 * 10 cropped frames are extracted
This means the extracted area of the original frame is 1792,

'''

ori_size = (1828, 1332)
out_size = (320, 180)
input_path = 'M:/MAI_dataset/Sequence_lines_1/Cinecitta'
output_path = 'M:/MAI_dataset/tempSamples/test_set/frame/'
FILETYPE = 'bmp'
numFrames = 51

# Define 'scratchTest'
# input_path = 'M:/MAI_dataset/Sequence_lines_1/Carrier/'
# scratchTest_OutPath = 'M:/MAI_dataset/Sequence_lines_1/scrattifchTest_testset/'
# scratchTest_PredPath = 'M:/MAI_dataset/Sequence_lines_1/scratchTest_pred/'
# ori_size = (1920, 1080)
# FILETYPE = 'tif'
# numFrames = 20


def crop_img(input_path_, output_path_, out_size_, ori_size_, fileType, numFra, ifDarker):
    width, height = out_size_[0], int(out_size_[1] / 3)
    input_img_paths = glob.glob(f'{input_path_}/*.{fileType}')
    count_frame = 0

    if numFra > len(input_img_paths) - 1:
        num = len(input_img_paths) - 1
    else:
        num = numFra

    tilingCounter = 0
    for index in tqdm(range(1, num), bar_format='{percentage:3.0f}%|{bar:100}{r_bar}'):
        count_frame += 1
        input_img1 = cv.imread(input_img_paths[index - 1], cv.IMREAD_UNCHANGED)
        input_img2 = cv.imread(input_img_paths[index], cv.IMREAD_UNCHANGED)
        input_img3 = cv.imread(input_img_paths[index + 1], cv.IMREAD_UNCHANGED)
        if ifDarker:
            input_img1 = 255 - input_img1
            input_img2 = 255 - input_img2
            input_img3 = 255 - input_img3

        input_img1 = input_img1[:, :, 0]
        input_img2 = input_img2[:, :, 0]
        input_img3 = input_img3[:, :, 0]

        oriHeight, oriWidth = input_img1.shape[0], input_img1.shape[1]
        i, j = 0, 0
        while height * j <= oriHeight:
            i = 0
            while width * i <= oriWidth:

                topleft_x, topleft_y = i * width, j * height

                if topleft_x + width > oriWidth - 1:
                    topleft_x = oriWidth - 1 - width
                if topleft_y + height > oriHeight - 1:
                    topleft_y = oriHeight - 1 - height

                crop1 = input_img1[topleft_y:topleft_y + height, topleft_x:topleft_x + width]
                crop2 = input_img2[topleft_y:topleft_y + height, topleft_x:topleft_x + width]
                crop3 = input_img3[topleft_y:topleft_y + height, topleft_x:topleft_x + width]
                crop_vertical_3 = np.concatenate((crop1, crop2, crop3), axis=0)
                cv.imwrite(f'{output_path_}frame{count_frame:03d}-{(i + 1):03d}-{(j + 1):03d}.png', crop_vertical_3)

                i += 1
                tilingCounter += 1

            j += 1

    print(f'INFO: Total number of frames generated: {tilingCounter}.')


def main(args):
    reset = True
    if reset is True:
        if os.path.isdir(output_path):
            rmtree(output_path)
            os.mkdir(output_path)
        crop_img(input_path, output_path, out_size, ori_size, FILETYPE, numFrames)


if __name__ == '__main__':
    main(sys.argv)
