import os
import time
import glob

import cv2 as cv
from shutil import rmtree
''' 
This script is for extracting small crops from a large image, e.g. 1280×720 and higher resolutions.
The aim of adding this script is to make better predictions over large input images for vertical scratch detections,
    since the trained DNN takes 320×180 images as input.
'''

adjust_size = (1920, 1080)
input_path = 'M:/MAI_dataset/Sequence_lines_1'
output_path = 'M:/MAI_dataset/tempSamples/test_set/cropped/'


def crop_img(input_path_, output_path_):
    input_img_paths = glob.glob(f'{input_path_}/*.bmp')
    count_frame = 0
    time_now = time.time()
    for p in input_img_paths:
        count_frame += 1
        input_img = cv.imread(p)
        input_img = cv.resize(input_img, adjust_size)
        for i in range(0, 6):
            for j in range(0, 6):
                topleft_x, topleft_y = i * 320, j * 180
                crop = input_img[topleft_y:topleft_y+180, topleft_x:topleft_x + 320]
                cv.imwrite(f'{output_path_}frame-{count_frame}-{i + 1}-{j + 1}.png', crop)
        if (count_frame % 10) == 0 or count_frame == len(input_img_paths):
            print(f'Finished cropping frame {count_frame} out of {len(input_img_paths)} frames'
                  f': {(time.time() - time_now):04f} sec')
            time_now = time.time()


if __name__ == '__main__':
    reset = True
    if reset is True:
        if os.path.isdir(output_path):
            rmtree(output_path)
            os.mkdir(output_path)
        crop_img(input_path, output_path)
