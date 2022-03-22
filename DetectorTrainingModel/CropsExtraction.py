import os
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
The output frames are in resolution 256 * 144 and 8 * 10 cropped frames are extracted
This means the extracted area of the original frame is 1792,

'''

ori_size = (1828, 1332)
out_size = (320, 180)
input_path = 'M:/MAI_dataset/Sequence_lines_1'
output_path = 'M:/MAI_dataset/tempSamples/test_set/frame/'


def crop_img(input_path_, output_path_, out_size_, ori_size_):
    width, height = out_size_[0], out_size_[1]
    input_img_paths = glob.glob(f'{input_path_}/*.bmp')
    count_frame = 0
    for index in tqdm(range(1, len(input_img_paths) - 1), bar_format='{percentage:3.0f}%|{bar:100}{r_bar}'):
        count_frame += 1
        input_img1 = cv.imread(input_img_paths[index - 1])
        input_img2 = cv.imread(input_img_paths[index])
        input_img3 = cv.imread(input_img_paths[index + 1])

        i, j = 0, 0
        while height * (j + 1) <= ori_size_[1]:
            i = 0
            while width * (i + 1) <= ori_size_[0]:

                topleft_x, topleft_y = i * width, j * height

                if topleft_x + width > ori_size_[0] - 1:
                    topleft_x = ori_size_[0] - 1 - width
                if topleft_y + height > ori_size_[1] - 1:
                    topleft_y = ori_size_[1] - 1 - height

                crop1 = input_img1[topleft_y:topleft_y + height, topleft_x:topleft_x + width]
                crop2 = input_img2[topleft_y:topleft_y + height, topleft_x:topleft_x + width]
                crop3 = input_img3[topleft_y:topleft_y + height, topleft_x:topleft_x + width]
                crop_vertical_3 = np.concatenate((crop1, crop2, crop3), axis=0)
                crop_vertical_3 = cv.resize(crop_vertical_3, out_size)
                cv.imwrite(f'{output_path_}frame{count_frame}-{i + 1}-{j + 1}.png', crop_vertical_3)

                i += 1

            j += 1


if __name__ == '__main__':
    reset = True
    if reset is True:
        if os.path.isdir(output_path):
            rmtree(output_path)
            os.mkdir(output_path)
        crop_img(input_path, output_path, out_size, ori_size)
