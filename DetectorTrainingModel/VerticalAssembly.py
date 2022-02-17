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
os.environ['DML_VISIBLE_DEVICES'] = '0'
cv.ocl.setUseOpenCL(True)
new_size = (320, 180)
fps = 120


def vertically_assembly(name):

    frame_folder = f'M:/MAI_dataset/Degraded_set/{name}/frame'
    mask_folder = f'M:/MAI_dataset/Degraded_set/{name}/mask'
    to_frame_folder = f'M:/MAI_dataset/Degraded_set/VA-{name}/frame'
    to_mask_folder = f'M:/MAI_dataset/Degraded_set/VA-{name}/mask'
    for path_temp in [frame_folder, mask_folder, to_frame_folder, to_mask_folder]:
        if os.path.isdir(path_temp) is False:
            import warnings
            errorMessage = f'Error: The following folder does not exist: {path_temp}',
            warnings.simplefilter(errorMessage)
            exit()

    frameFiles_path = sorted([
        os.path.join(frame_folder, fname)
        for fname in os.listdir(frame_folder)
        if fname.endswith(".png")]
    )
    maskFiles_path = sorted([
        os.path.join(mask_folder, fname)
        for fname in os.listdir(mask_folder)
        if fname.endswith(".png")]
    )
    print(f'INFO: Got dataset: {len(frameFiles_path)} degraded images, {len(maskFiles_path)} masks')

    for index in tqdm(range(1, len(frameFiles_path) - 1), bar_format='{percentage:3.0f}%|{bar:100}{r_bar}'):

        img1 = cv.imread(frameFiles_path[index - 1], cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(frameFiles_path[index], cv.IMREAD_GRAYSCALE)
        img3 = cv.imread(frameFiles_path[index + 1], cv.IMREAD_GRAYSCALE)
        frame_vertically_3 = np.concatenate((img1, img2, img3), axis=0)
        frame_vertically_3 = cv.resize(frame_vertically_3, [int(new_size[0] * 1.5), int(new_size[1] * 1.5)])
        cv.imwrite(f'{to_frame_folder}/frame-{index}.png', frame_vertically_3)
        # cv.imshow(f'frame_vertically_3', frame_vertically_3)
        # cv.moveWindow(f'frame_vertically_3', -1000 - 2 * int(new_size[0] * 1.5), 50)

        mask1 = cv.imread(maskFiles_path[index - 1], cv.IMREAD_GRAYSCALE)
        mask2 = cv.imread(maskFiles_path[index], cv.IMREAD_GRAYSCALE)
        mask3 = cv.imread(maskFiles_path[index + 1], cv.IMREAD_GRAYSCALE)
        mask_vertically_3 = np.concatenate((mask1, mask2, mask3), axis=0)
        mask_vertically_3 = cv.resize(mask_vertically_3, [int(new_size[0] * 1.5), int(new_size[1] * 1.5)])
        cv.imwrite(f'{to_mask_folder}/mask-{index}.png', mask_vertically_3)
        # cv.imshow(f'mask_vertically_3', mask_vertically_3 * 255)
        # cv.moveWindow(f'mask_vertically_3', -1000 - int(new_size[0] * 1.5), 50)

        # mask_over_ori = cv.cvtColor(frame_vertically_3, cv.COLOR_GRAY2RGB)
        # mask_over_ori[:, :, 0] = np.clip((mask_over_ori[:, :, 0] - mask_vertically_3 * 150), 0.0, 255.0)
        # mask_over_ori[:, :, 1] = np.clip((mask_over_ori[:, :, 1] - mask_vertically_3 * 150), 0.0, 255.0)
        # mask_over_ori[:, :, 2] = np.clip((mask_over_ori[:, :, 2] + mask_vertically_3 * 150), 0.0, 255.0)

        # cv.waitKey(20)


def main(args):
    film_name = ['ST', 'BBB', 'ED', 'TOS']
    for name in film_name:
        vertically_assembly(name)
        print(f'INFO: Finished picture processing of film: {name}!')


if __name__ == '__main__':
    main(sys.argv)
