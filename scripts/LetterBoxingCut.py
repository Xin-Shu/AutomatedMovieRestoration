import sys
import cv2 as cv
import time

input_path = 'M:/MAI_dataset/Origin_set/ST-720-png'
output_path = 'M:/MAI_dataset/Origin_set/ST(cut)-720-png'


def boxing_cutting(in_path, out_path, top_pixel, bot_pixel):
    import glob
    ori_paths = glob.glob(f'{in_path}/*.png')
    count = 0
    time_now = time.time()
    for p in ori_paths:
        count += 1
        if (count % 100) == 0:
            print(f'Processing: {count} of {len(ori_paths)}: {time.time() - time_now} sec')
            time_now = time.time()
        ori = cv.imread(p)
        crop_img = ori[top_pixel:bot_pixel, :]
        cv.imwrite(f'{out_path}/{count:05d}.png', crop_img)
        cv.imshow("cropped", crop_img)
        cv.waitKey(1)


def main(args):
    boxing_cutting(input_path, output_path, 88, 633)


if __name__ == '__main__':
    main(sys.argv)
