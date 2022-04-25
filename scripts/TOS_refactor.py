import os
import sys

import cv2 as cv
from shutil import copyfile
TOS_ORG = 'M:/MAI_dataset/Origin_set/TOS-1080-png/'
os.environ['DML_VISIBLE_DEVICES'] = '0'


def rename(path):
    if os.path.isdir(path) is False:
        import warnings
        errorMessage = f'Error: The following folder does not exist: {path}',
        warnings.simplefilter(errorMessage)

    frameFiles = sorted([
            os.path.join(path, fname)
            for fname in os.listdir(path)
            if fname.endswith(".png")]
    )

    totalFrames = len(frameFiles)

    for num in range(0, totalFrames):
        frame_from_path = frameFiles[num]
        actualNum = frame_from_path.replace('M:/MAI_dataset/Origin_set/TOS-1080-png/$filename', '')
        actualNum = int(actualNum.replace('.png', ''))
        if actualNum < 10:
            frame_num_str = '0000' + str(actualNum)
        if 10 <= actualNum < 100:
            frame_num_str = '000' + str(actualNum)
        if 100 <= actualNum < 1000:
            frame_num_str = '00' + str(actualNum)
        if 1000 <= actualNum < 10000:
            frame_num_str = '0' + str(actualNum)
        if actualNum >= 10000:
            frame_num_str = str(actualNum)
        frame_to_path = path + frame_num_str + '.png'
        copyfile(frame_from_path, frame_to_path)
        os.remove(frame_from_path)


def imdis(path):
    fps = 200
    if os.path.isdir(path) is False:
        import warnings
        errorMessage = f'Error: The following folder does not exist: {path}',
        warnings.simplefilter(errorMessage)

    frameFiles = sorted([
            os.path.join(path, fname)
            for fname in os.listdir(path)
            if fname.endswith(".png")]
    )
    for name in frameFiles:
        frame = cv.imread(name)
        cv.imshow('frame', frame)
        cv.waitKey(int(1000 / fps))


def main(args):
    # rename(TOS_ORG)
    imdis(TOS_ORG)


if __name__ == '__main__':
    main(sys.argv)
