import os
import glob

import cv2 as cv
import numpy as np
from tqdm import tqdm
from shutil import rmtree
''' 
This script is for extracting small crops from a large image, e.g. 1280×720 and higher resolutions.
The aim of adding this script is to make better predictions over large input images for vertical scratch detections,
    since the trained DNN takes 320×180 images as input.
'''

adjust_size = (1920, 1080)
out_size = (320, 180)
input_path = 'M:/MAI_dataset/Sequence_lines_1'
output_path = 'M:/MAI_dataset/tempSamples/test_set/frame/'


def crop_img(input_path_, output_path_):
    input_img_paths = glob.glob(f'{input_path_}/*.bmp')
    count_frame = 0
    for index in tqdm(range(1, len(input_img_paths) - 1), bar_format='{percentage:3.0f}%|{bar:100}{r_bar}'):
        count_frame += 1
        input_img1 = cv.imread(input_img_paths[index - 1])
        input_img1 = cv.resize(input_img1, adjust_size)
        input_img2 = cv.imread(input_img_paths[index])
        input_img2 = cv.resize(input_img2, adjust_size)
        input_img3 = cv.imread(input_img_paths[index + 1])
        input_img3 = cv.resize(input_img3, adjust_size)

        for i in range(0, 6):
            for j in range(0, 6):

                topleft_x, topleft_y = i * 320, j * 180

                crop1 = input_img1[topleft_y:topleft_y + 180, topleft_x:topleft_x + 320]
                crop2 = input_img2[topleft_y:topleft_y + 180, topleft_x:topleft_x + 320]
                crop3 = input_img3[topleft_y:topleft_y + 180, topleft_x:topleft_x + 320]
                crop_vertical_3 = np.concatenate((crop1, crop2, crop3), axis=0)
                crop_vertical_3 = cv.resize(crop_vertical_3, out_size)
                cv.imwrite(f'{output_path_}frame{count_frame}-{i + 1}-{j + 1}.png', crop_vertical_3)


if __name__ == '__main__':
    reset = True
    if reset is True:
        if os.path.isdir(output_path):
            rmtree(output_path)
            os.mkdir(output_path)
        crop_img(input_path, output_path)
