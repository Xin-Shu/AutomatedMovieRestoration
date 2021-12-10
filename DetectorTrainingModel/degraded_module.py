import os
import sys
import time
import cv2 as cv
import numpy as np

org_folder = 'M:\\MAI_dataset\\Origin_set\\ED-360-png'
degrade_folder = 'M:\\MAI_dataset\\Degraded_set\\ED\\frame'
mask_folder = 'M:\\MAI_dataset\\Degraded_set\\ED\\mask'


def makeLineProfile(cols, pos, amplitude, damping, m, row, w):
    x = np.array(range(0, cols))
    dx = abs(x - (m * row + pos))
    profile = amplitude * (np.power(damping, dx)) * np.cos(3 * np.pi * dx / (2 * w))

    return profile


def degraded_module():
    global org_folder, degrade_folder, mask_folder
    if os.path.isdir(org_folder) is False:
        import warnings
        errorMessage = f'Error: The following folder does not exist:\n%s', org_folder
        warnings.simplefilter(errorMessage)

    else:
        imgPattern = os.path.join(org_folder, '*.png')
        pngFiles = os.listdir(imgPattern)

        # Processing images
        max_width = 3
# colormap(gray(256));
#
        for i in range(1, len(pngFiles)):
            frameName = pngFiles[i].name
# if mod(i, 10) == 0
#     fprintf("Processing: %d of %d -- '%s'.\n", ...
#     i, length(pngFiles), frameName)
#     end
#     fullName = fullfile(org_folder, frameName);
#     frame_org = imread(fullName);
#     gray_frame = im2gray(imresize(frame_org, [180, 320])); % 1 / 5
#     of
#     1080
#     p
#     [rows, cols, chan] = size(gray_frame);
#     binary_mask = zeros(180, 320, 1, 'double');
#     degrade = gray_frame;
#     degrade2 = double(degrade);
#
#     for line_num = 1: (1 + floor(rand * 5)) % Add
#     randomly
#     up
#     to
#     5
#     lines
#     line_pos = floor(rand * (cols - 2 * max_width)) + max_width + 1;
#     w = round(1 + rand * max_width);
#     a = rand * 100;
#
#     if (line_pos - w > 1)
#         left_boundary = line_pos - w;
#     else
#         left_boundary = 1;
#     end
#     if (line_pos + w <= 320)
#         right_boundary = line_pos + w;
#     else
#         right_boundary = line_pos - w;
#     end
#     scratch_width = right_boundary - left_boundary + 1;
#
#     binary_mask(:, left_boundary: right_boundary)= ...
#     ones(180, scratch_width);
#
# slope = randi([-10, 10]) * 0.0005;
# for j = 1: rows
# profile = makeLineProfile(...
# cols, line_pos, (a - 50), 0.25, slope, j, w);
# degrade2(j, left_boundary: right_boundary) = ...
# double(degrade2(j, left_boundary: right_boundary)) ...
# + profile(:, left_boundary: right_boundary);
# end
# end
#
# % binary_mask = binary_mask > 0;
# degradedFullName = fullfile(degrade_folder, frameName);
# maskFullName = fullfile(mask_folder, frameName);
# imwrite(degrade2(1: 180, 1: 320), ...
# colormap(gray(256)), degradedFullName);
# imwrite(binary_mask(1: 180, 1: 320), maskFullName);
# % imshow(imresize(degrade2, [800, 1920]), colormap(gray(256)));
#
# end
#
# fprintf('%s\n', "INFO: Finished Images Processing!");
#
