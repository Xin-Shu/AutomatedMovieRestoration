import os
import sys
import random

import pyopencl as cl
from shutil import copyfile
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'

# Elephents Dream
ED_ORG = 'M:/MAI_dataset/Origin_set/ED-360-png/'
ED_DEGRADED = 'M:/MAI_dataset/Degraded_set/ED/frame/'
ED_MASK = 'M:/MAI_dataset/Degraded_set/ED/mask/'
# Big Buck Bunny
BBB_ORG = 'M:/MAI_dataset/Origin_set/BBB-360-png/'
BBB_DEGRADED = 'M:/MAI_dataset/Degraded_set/BBB/frame/'
BBB_MASK = 'M:/MAI_dataset/Degraded_set/BBB/mask/'
# Tear of Steel
TOS_ORG = 'M:/MAI_dataset/Origin_set/TOS-1080-png/'
TOS_DEGRADED = 'M:/MAI_dataset/Degraded_set/TOS/frame/'
TOS_MASK = 'M:/MAI_dataset/Degraded_set/TOS/mask/'
# Sintel Trailer
ST_ORG = 'M:/MAI_dataset/Origin_set/ST-720-png/'
ST_DEGRADED = 'M:/MAI_dataset/Degraded_set/ST/frame/'
ST_MASK = 'M:/MAI_dataset/Degraded_set/ST/mask/'

count = 0


def sampling_frames(degraded_path, mask_path, name):
    global count
    for path_temp in [degraded_path, mask_path]:
        if os.path.isdir(path_temp) is False:
            import warnings
            errorMessage = f'Error: The following folder does not exist: {path_temp}',
            warnings.simplefilter(errorMessage)
            break

    frameFiles = sorted([
            os.path.join(degraded_path, fname)
            for fname in os.listdir(degraded_path)
            if fname.endswith(".png")]
    )
    maskFiles = sorted([
            os.path.join(mask_path, fname)
            for fname in os.listdir(mask_path)
            if fname.endswith(".png")]
    )
    totalFrames = len(frameFiles)

    list_num_of_sample = random.sample(range(3000, totalFrames - 1000), 400)

    for num in list_num_of_sample:
        count += 1
        print(f'Processing: {count} out of 1200.')
        frame_from_path = frameFiles[num]
        frame_to_path = 'M:/MAI_dataset/tempSamples/degraded/' + 'frame-{:04}'.format(count) + '.png'
        copyfile(frame_from_path, frame_to_path)

        mask_from_path = maskFiles[num]
        mask_to_path = 'M:/MAI_dataset/tempSamples/mask/' + 'frame-{:04}'.format(count) + '.png'
        copyfile(mask_from_path, mask_to_path)


def main(args):
    reset = True
    if reset is True:
        from shutil import rmtree
        rmtree('M:/MAI_dataset/tempSamples/degraded/')
        os.mkdir('M:/MAI_dataset/tempSamples/degraded/')
        rmtree('M:/MAI_dataset/tempSamples/mask/')
        os.mkdir('M:/MAI_dataset/tempSamples/mask/')

    sampling_frames(ED_DEGRADED, ED_MASK, 'ED')
    sampling_frames(BBB_DEGRADED, BBB_MASK, 'BBB')
    sampling_frames(TOS_DEGRADED, TOS_MASK, 'TOS')
    # sampling_frames(ST_DEGRADED, ST_MASK, 'ST')


if __name__ == '__main__':
    main(sys.argv)
