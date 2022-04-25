import os
import sys
import random

import cv2 as cv
import numpy as np
from tqdm import tqdm
from shutil import copyfile

maskFolder = 'M:/MAI_dataset/Sequence_lines_1/Knight/'
sr = cv.dnn_superres.DnnSuperResImpl_create()
modelPath = "EDSR_x2.pb"
sr.readModel(modelPath)
sr.setModel("edsr", 2)

path = 'C:/Users/82654/Desktop/MAI project/Dissertation/Eval_image/'
mask = cv.imread(f'{path}lines10005_withoutConcate.bmp', cv.IMREAD_GRAYSCALE)
maskEnergy = mask.sum(axis=0)
for i in range(0, len(maskEnergy)):
    if maskEnergy[i] >= 8*255:
        mask[:, i] = 255
mask = cv.resize(mask, [1828, 1332])
frame_ori = cv.imread(f'{path}/lines100005.png', cv.IMREAD_UNCHANGED)
cleanedOverlay = cv.cvtColor(frame_ori, cv.COLOR_GRAY2RGB)
cleanedOverlay[:, :, 1] = np.clip((cleanedOverlay[:, :, 1] - mask / 255 * 230), 0.0, 255.0)
cv.imwrite(f'{path}/cleanedOverlay.png', cleanedOverlay)
# pdmFiles = sorted([
#     os.path.join(maskFolder, fname)
#     for fname in os.listdir(maskFolder)
#     if fname.endswith('.png')
# ])
#
# for path in pdmFiles:
#     img = cv.imread(path, cv.IMREAD_UNCHANGED)
#     img = sr.upsample(img)
#     print(img.shape)
#
#     # os.remove(path)
#     cv.imwrite(f'{maskFolder}/Up_{os.path.basename(path)}', img)
# # path = f'{maskFolder}knight00031.png'
# path = f'D:/1.png'
# # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# frame_ori = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, 0:3]
# print(frame_ori.shape)
# # cleanedOverlay = frame_ori
# # maskAssembled = cv2.imread(f'{maskFolder}frame056-001-001.png', cv2.IMREAD_GRAYSCALE)
# # cleanedOverlay[:, :, 1] = np.clip((cleanedOverlay[:, :, 1] - maskAssembled / 255 * 230), 0.0, 255.0)
# # # os.remove(path)
# cv2.imwrite('D:/1.png', frame_ori)
# cv2.imwrite('D:/1_inverse.png', 255 - frame_ori)
