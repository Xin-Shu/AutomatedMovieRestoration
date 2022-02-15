import os
import sys
import time

import random
import cv2 as cv
import numpy as np
from tqdm import tqdm
from pyopencl.tools import get_test_platforms_and_devices

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'
cv.ocl.setUseOpenCL(True)
new_size = (320, 180)
fps = 120


def vertically_assembly(name):

    frame_folder = f'M:/MAI_dataset/Degraded_set/{name}/frame'
    mask_folder = f'M:/MAI_dataset/Degraded_set/{name}/mask'
    to_frame_folder = f'M:/MAI_dataset/Degraded_set/VA-{name}/frame'
    to_mask_folder = f'M:/MAI_dataset/Degraded_set/VA-{name}/mask'

    print(f'{name} if a folder: {to_frame_folder} {os.path.isdir(to_frame_folder)}')
    print(f'{name} if a folder: {to_mask_folder} {os.path.isdir(to_mask_folder)} \n')


def main(args):
    film_name = ['ST', 'BBB', 'ED', 'TOS']
    for name in film_name:
        vertically_assembly(name)
        print(f'INFO: Finished picture processing of film: {name}!')


if __name__ == '__main__':
    main(sys.argv)
