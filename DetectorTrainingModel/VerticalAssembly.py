import os
import sys

import random
import cv2 as cv
import numpy as np
from tqdm import tqdm

from DatasetPerparation import output_size

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'
os.environ['DML_VISIBLE_DEVICES'] = '0'
cv.ocl.setUseOpenCL(True)
new_size = output_size
fps = 120


def vertically_assembly(name, reset):

    frame_folder = f'M:/MAI_dataset/Degraded_set/SAMPLE-{name}/frame'
    mask_folder = f'M:/MAI_dataset/Degraded_set/SAMPLE-{name}/mask'
    to_frame_folder = f'M:/MAI_dataset/Degraded_set/VA-{name}/frame'
    to_mask_folder = f'M:/MAI_dataset/Degraded_set/VA-{name}/mask'

    if reset:
        from shutil import rmtree
        rmtree(to_frame_folder)
        rmtree(to_mask_folder)
        os.mkdir(to_frame_folder)
        os.mkdir(to_mask_folder)

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
    print(f'\nINFO: Got dataset: {len(frameFiles_path)} degraded images, {len(maskFiles_path)} masks.\n')

    totalNum = 0
    for index in tqdm(range(1, len(frameFiles_path) - 1), bar_format='{percentage:3.0f}%|{bar:100}{r_bar}'):

        img1 = cv.imread(frameFiles_path[index - 1], cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(frameFiles_path[index], cv.IMREAD_GRAYSCALE)
        img3 = cv.imread(frameFiles_path[index + 1], cv.IMREAD_GRAYSCALE)

        mask1 = cv.imread(maskFiles_path[index - 1], cv.IMREAD_GRAYSCALE)
        mask2 = cv.imread(maskFiles_path[index], cv.IMREAD_GRAYSCALE)
        mask3 = cv.imread(maskFiles_path[index + 1], cv.IMREAD_GRAYSCALE)

        ori_size = img2.shape       # e.g., (545, 1280)
        scratches_x = [int(ori_size[1] / 5), int(ori_size[1] / 2), int(ori_size[1] / 1.2)]
        # print('scratches_x', scratches_x)

        i, j = 0, 0
        width, height = new_size[0], int(new_size[1] / 3)
        while height * (j + 1) <= ori_size[0]:

            topLeft_y = height * j
            i = 0
            for scratch_x in scratches_x:
                # print('scratch_x', scratch_x)

                topLeft_x = scratch_x - random.randint(int(new_size[0] * 0.1), int(new_size[0] * 0.9))

                if topLeft_x + width > ori_size[1]:
                    topLeft_x = ori_size[1] - width - 1
                if topLeft_x < 0:
                    topLeft_x = 0

                frameCrop_1 = img1[topLeft_y:topLeft_y + height, topLeft_x:topLeft_x + width]
                frameCrop_2 = img2[topLeft_y:topLeft_y + height, topLeft_x:topLeft_x + width]
                frameCrop_3 = img3[topLeft_y:topLeft_y + height, topLeft_x:topLeft_x + width]
                frameAssembly = np.concatenate((frameCrop_1, frameCrop_2, frameCrop_3), axis=0)

                # maskCrop_1 = mask1[topLeft_y:topLeft_y + height, topLeft_x:topLeft_x + width]
                maskCrop_2 = mask2[topLeft_y:topLeft_y + height, topLeft_x:topLeft_x + width]
                # maskCrop_3 = mask3[topLeft_y:topLeft_y + height, topLeft_x:topLeft_x + width]
                # maskAssembly = np.concatenate((maskCrop_1, maskCrop_2, maskCrop_3), axis=0)

                cv.imwrite(f'{to_frame_folder}/frame{index}-{i + 1}-{j + 1}.png', frameAssembly)
                cv.imwrite(f'{to_mask_folder}/mask{index}-{i + 1}-{j + 1}.png', maskCrop_2)

                i += 1
                totalNum += 1

            j += 1

    print(f'\nINFO: Got {totalNum} of assembled frames from {name}.')


def main(args):
    film_name = ['ST', 'BBB', 'ED', 'TOS']
    for name in film_name:
        vertically_assembly(name, 1)
        print(f'INFO: Finished picture processing of film: {name}!')


if __name__ == '__main__':
    main(sys.argv)
