import os
import sys
import random
from tqdm import tqdm
from shutil import copyfile, rmtree

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'

NUM_OF_PICKING = 500

ED_ORI = 'M:/MAI_dataset/Origin_set/ED-360-png/'        # Elephents Dream
BBB_ORI = 'M:/MAI_dataset/Origin_set/BBB-360-png/'      # Big Buck Bunny
TOS_ORI = 'M:/MAI_dataset/Origin_set/TOS-1080-png/'     # Tear of Steel
ST_ORI = 'M:/MAI_dataset/Origin_set/ST(cut)-720-png/'        # Sintel Trailer

ED_SAMPLE = 'M:/MAI_dataset/Origin_set/ED-sample/'      # Elephents Dream
BBB_SAMPLE = 'M:/MAI_dataset/Origin_set/BBB-sample/'    # Big Buck Bunny
TOS_SAMPLE = 'M:/MAI_dataset/Origin_set/TOS-sample/'    # Tear of Steel
ST_SAMPLE = 'M:/MAI_dataset/Origin_set/ST-sample/'      # Sintel Trailer


def OriginalFramesSampling(in_path, out_path, num, reset):

    if reset:
        rmtree(out_path)
        os.mkdir(out_path)

    frameFiles_path = sorted([
        os.path.join(in_path, fname)
        for fname in os.listdir(in_path)
        if fname.endswith(".png")]
    )

    totalFrames = len(frameFiles_path)
    if totalFrames > 10000:
        index_in = random.randint(int(totalFrames * 0.3), int(totalFrames * 0.7))
    else:
        index_in = 200 # random.randint(200, 1000)

    for i in tqdm(range(index_in, index_in + num), bar_format='{percentage:3.0f}%|{bar:100}{r_bar}'):

        frame_from_path = frameFiles_path[i]
        frame_to_path = f'{out_path}{os.path.basename(frameFiles_path[i])}'
        copyfile(frame_from_path, frame_to_path)


def main(args):
    reset = True

    OriginalFramesSampling(ED_ORI, ED_SAMPLE, NUM_OF_PICKING, reset)
    OriginalFramesSampling(BBB_ORI, BBB_SAMPLE, NUM_OF_PICKING, reset)
    OriginalFramesSampling(TOS_ORI, TOS_SAMPLE, NUM_OF_PICKING, reset)
    OriginalFramesSampling(ST_ORI, ST_SAMPLE, NUM_OF_PICKING, reset)


if __name__ == '__main__':
    main(sys.argv)
