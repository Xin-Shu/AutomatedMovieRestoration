import os
import sys
import random
from tqdm import tqdm
from shutil import copyfile

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'

# # Elephents Dream
# ED_DEGRADED = 'M:/MAI_dataset/Degraded_set/ED/frame/'
# ED_MASK = 'M:/MAI_dataset/Degraded_set/ED/mask/'
# # Big Buck Bunny
# BBB_DEGRADED = 'M:/MAI_dataset/Degraded_set/BBB/frame/'
# BBB_MASK = 'M:/MAI_dataset/Degraded_set/BBB/mask/'
# # Tear of Steel
# TOS_DEGRADED = 'M:/MAI_dataset/Degraded_set/TOS/frame/'
# TOS_MASK = 'M:/MAI_dataset/Degraded_set/TOS/mask/'
# # Sintel Trailer
# ST_DEGRADED = 'M:/MAI_dataset/Degraded_set/ST/frame/'
# ST_MASK = 'M:/MAI_dataset/Degraded_set/ST/mask/'

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

    if name == 'ST':
        count_st = 0
        list_num_of_sample = random.sample(range(100, totalFrames - 200), 500)
        print(f'\nProcessing validation set [{name}]: '
              f'randomly pick {len(list_num_of_sample)} from {totalFrames} frames.')
        for num in tqdm(list_num_of_sample, bar_format='{percentage:3.0f}%|{bar:100}{r_bar}'):
            count_st += 1
            frame_from_path = frameFiles[num]
            frame_to_path = 'M:/MAI_dataset/tempSamples/valid_set/frame/' + 'frame-{:04}'.format(count_st) + '.png'
            copyfile(frame_from_path, frame_to_path)

            mask_from_path = maskFiles[num]
            mask_to_path = 'M:/MAI_dataset/tempSamples/valid_set/mask/' + 'frame-{:04}'.format(count_st) + '.png'
            copyfile(mask_from_path, mask_to_path)

    else:
        list_num_of_sample = random.sample(range(3000, totalFrames - 1000), 500)
        print(f'\nProcessing training set [{name}]: '
              f'randomly pick {len(list_num_of_sample)} from {totalFrames} frames.')
        for num in tqdm(list_num_of_sample, bar_format='{percentage:3.0f}%|{bar:100}{r_bar}'):
            count += 1
            frame_from_path = frameFiles[num]
            frame_to_path = 'M:/MAI_dataset/tempSamples/train_set/degraded/' + 'frame-{:04}'.format(count) + '.png'
            copyfile(frame_from_path, frame_to_path)

            mask_from_path = maskFiles[num]
            mask_to_path = 'M:/MAI_dataset/tempSamples/train_set/mask/' + 'frame-{:04}'.format(count) + '.png'
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
