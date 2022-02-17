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


def makeLineProfile(cols, pos, amplitude, damping, m, row, w):
    x = np.array(range(0, cols))
    dx = abs(x - (m * row + pos))
    profile = amplitude * (np.power(damping, dx)) * np.cos(3 * np.pi * dx / (2 * w))
    return profile


def degraded_module(name, resol):

    org_folder = f'M:/MAI_dataset/Origin_set/{resol}-png'
    degrade_folder = f'M:/MAI_dataset/Degraded_set/{name}/frame'
    mask_folder = f'M:/MAI_dataset/Degraded_set/{name}/mask'

    if os.path.isdir(org_folder) is False:
        import warnings
        errorMessage = f'Error: The following folder does not exist:\n%s', org_folder
        warnings.simplefilter(errorMessage)

    else:
        pngFiles = [file for file in os.listdir(org_folder) if file.endswith('.png')]
        # Processing images
        max_width = 5
        # colormap(gray(256));
        scratch_num_list = []
        line_pos_set, brightness_set = [50, 150, 270], 70
        count = 0
        time_now = time.time()
        print(f'\nProcessing filme [{name}], {len(pngFiles)} frames in total')
        for i in tqdm(range(1, len(pngFiles)), bar_format='{percentage:3.0f}%|{bar:100}{r_bar}'):
            count += 1
            frameName = org_folder + f'/{pngFiles[i]}'

            fullName = os.path.join(org_folder, frameName)
            frame_org = cv.imread(fullName)
            frame_resize = cv.resize(frame_org, new_size)
            gray_frame = cv.cvtColor(frame_resize, cv.COLOR_RGB2GRAY)  # 1 / 5 of 1080p
            rows, cols = gray_frame.shape
            binary_mask = np.zeros([rows, cols, 1], 'double')
            degrade = gray_frame
            degrade2 = degrade
            scratch_num = 1 + random.randint(0, 2)
            scratch_num_list.append(scratch_num)

            temp_brightness = brightness_set
            temp_line_pos_set = line_pos_set
            for line_num in range(1, scratch_num + 1):  # Add randomly up to 4 lines
                # line_pos = random.randint(max_width, cols - 2 * max_width) + max_width + 1
                line_pos = random.randint(-1, 1) + temp_line_pos_set[line_num - 1]
                if line_pos >= 320 or line_pos <= 0:
                    temp_line_pos_set = line_pos_set
                else:
                    temp_line_pos_set[line_num - 1] = line_pos
                w = 2 + random.randrange(1, max_width + 2, 2)
                a = (random.uniform(-2, 2) * np.sqrt(0.1) + 1) * temp_brightness + random.uniform(-1, 1)
                temp_brightness = a
                if temp_brightness >= 130 or temp_brightness <= 20:
                    temp_brightness = brightness_set
                if line_pos - w > 0:
                    left_boundary = line_pos - int(np.floor(w / 2))
                else:
                    left_boundary = 0
                if line_pos + w < cols:
                    right_boundary = line_pos + int(np.ceil(w / 2))
                else:
                    right_boundary = cols - 1
                if right_boundary == left_boundary:
                    right_boundary = left_boundary + 1
                scratch_width = right_boundary - left_boundary

                binary_mask[:, left_boundary + 1:right_boundary - 1] = 1
                slope = random.uniform(-1, 1) * 0.0001
                for n in range(0, rows):
                    profile = makeLineProfile(cols, line_pos, (a - 50), 0.25, slope, n, w)
                    temp = degrade2[n, left_boundary:right_boundary] + profile[left_boundary:right_boundary] * 0.3
                    np.place(temp, temp > 255.0, 255.0)
                    np.place(temp, temp < 0.0, 0.0)
                    degrade2[n, left_boundary:right_boundary] = temp

            degradedFullName = degrade_folder + f'/{count:05d}' + '.png'
            maskFullName = mask_folder + f'/{count:05d}' + '.png'
            cv.imwrite(degradedFullName, degrade2)
            cv.imwrite(maskFullName, binary_mask)

            # cv.imshow(f'Degraded frame', cv.resize(degrade2, [960, 540]))
            # cv.imshow(f'Mask', cv.resize(binary_mask, [960, 540]))
            # cv.moveWindow(f'Degraded frame', 1000, 0)
            # cv.moveWindow(f'Mask', 1000, 541)
            # cv.waitKey(1)


def main(args):
    film_name = [['ST', 'ST(cut)-720'], ['BBB', 'BBB-360'], ['ED', 'ED-360'], ['TOS', 'TOS-1080']]
    for name, resol in film_name:
        degraded_module(name, resol)
        print(f'INFO: Finished picture processing of film: {name}!')


if __name__ == '__main__':
    main(sys.argv)

