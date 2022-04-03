import os
import sys
import random
from tqdm import tqdm
from shutil import copyfile

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'

# Elephents Dream
ED_DEGRADED = 'M:/MAI_dataset/Degraded_set/VA-ED/frame/'
ED_MASK = 'M:/MAI_dataset/Degraded_set/VA-ED/mask/'
# Big Buck Bunny
BBB_DEGRADED = 'M:/MAI_dataset/Degraded_set/VA-BBB/frame/'
BBB_MASK = 'M:/MAI_dataset/Degraded_set/VA-BBB/mask/'
# Tear of Steel
TOS_DEGRADED = 'M:/MAI_dataset/Degraded_set/VA-TOS/frame/'
TOS_MASK = 'M:/MAI_dataset/Degraded_set/VA-TOS/mask/'
# Sintel Trailer
ST_DEGRADED = 'M:/MAI_dataset/Degraded_set/VA-ST/frame/'
ST_MASK = 'M:/MAI_dataset/Degraded_set/VA-ST/mask/'

countTrain, countValid = 0, 0
numSamples = 300


def sampling_frames(degraded_path, mask_path, name):
    global countTrain, countValid
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

    indexTrainSample = random.sample(range(0, int(totalFrames * 0.75)), int(numSamples * 0.75))
    indexValidSample = random.sample(range(int(totalFrames * 0.75), totalFrames), int(numSamples * 0.25))

    print(f'\nProcessing training set [{name}]: '
          f'randomly pick {len(indexTrainSample)} from {totalFrames} frames.')
    for num in tqdm(indexTrainSample, bar_format='{percentage:3.0f}%|{bar:100}{r_bar}'):
        countTrain += 1
        frame_from_path = frameFiles[num]
        frame_to_path = 'M:/MAI_dataset/tempSamples/train_set/degraded/' + 'frame-{:04}'.format(countTrain) + '.png'
        copyfile(frame_from_path, frame_to_path)

        mask_from_path = maskFiles[num]
        mask_to_path = 'M:/MAI_dataset/tempSamples/train_set/mask/' + 'frame-{:04}'.format(countTrain) + '.png'
        copyfile(mask_from_path, mask_to_path)

    print(f'\nProcessing training set [{name}]: '
          f'randomly pick {len(indexValidSample)} from {totalFrames} frames.')
    for num in tqdm(indexValidSample, bar_format='{percentage:3.0f}%|{bar:100}{r_bar}'):
        countValid += 1
        frame_from_path = frameFiles[num]
        frame_to_path = 'M:/MAI_dataset/tempSamples/valid_set/frame/' + 'frame-{:04}'.format(countValid) + '.png'
        copyfile(frame_from_path, frame_to_path)

        mask_from_path = maskFiles[num]
        mask_to_path = 'M:/MAI_dataset/tempSamples/valid_set/mask/' + 'frame-{:04}'.format(countValid) + '.png'
        copyfile(mask_from_path, mask_to_path)


def main(args):
    reset = True
    if reset is True:
        from shutil import rmtree
        rmtree('M:/MAI_dataset/tempSamples/train_set/degraded/')
        os.mkdir('M:/MAI_dataset/tempSamples/train_set/degraded/')
        rmtree('M:/MAI_dataset/tempSamples/train_set/mask/')
        os.mkdir('M:/MAI_dataset/tempSamples/train_set/mask/')
        rmtree('M:/MAI_dataset/tempSamples/valid_set/')
        os.mkdir('M:/MAI_dataset/tempSamples/valid_set/')
        os.mkdir('M:/MAI_dataset/tempSamples/valid_set/frame')
        os.mkdir('M:/MAI_dataset/tempSamples/valid_set/mask')

    sampling_frames(ED_DEGRADED, ED_MASK, 'ED')
    sampling_frames(BBB_DEGRADED, BBB_MASK, 'BBB')
    sampling_frames(TOS_DEGRADED, TOS_MASK, 'TOS')
    sampling_frames(ST_DEGRADED, ST_MASK, 'ST')


if __name__ == '__main__':
    main(sys.argv)
