import cv2 as cv
import numpy as np

path = '../figures/'

img1 = cv.imread(f'{path}knight00031.png', cv.IMREAD_UNCHANGED)
if img1.shape[2] == 1:
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2RGB)
img2 = cv.imread(f'{path}knight00031_bright.png', cv.IMREAD_UNCHANGED)
imgCompose = np.concatenate([img1, img2], axis=1)
height, width = imgCompose.shape[0], imgCompose.shape[1]

# cv.imshow('', cv.resize(imgCompose, [int(0.5 * width), int(0.5 * height)]))
# cv.waitKey(0)

cv.imwrite(f'{path}knight00031_comp.png', imgCompose)
