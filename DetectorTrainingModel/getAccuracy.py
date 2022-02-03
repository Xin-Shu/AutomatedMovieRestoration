import os
import sys
import cv2 as cv
import numpy as np
from sklearn.metrics import jaccard_score

result_attempt_dir = 'M:/MAI_dataset/TrainedModels/02-03/Attempt 1'


def get_average_accuracy(_dir):
    import glob
    if os.path.isdir(f'{_dir}/mask_predictions') is False:
        print('Error 1: The request directory does not exist.')
    else:
        __path = f'{_dir}/mask_predictions'
        pred_paths = glob.glob(f'{__path}/pred*.png')
        truth_paths = glob.glob(f'{__path}/truth*.png')
        accuracy_temp = []
        for p1, p2 in zip(pred_paths, truth_paths):
            pred_ = cv.imread(p1).flatten()
            truth_ = cv.imread(p2).flatten()
            accuracy = jaccard_score(truth_, pred_, average='macro')
            # accuracy = (pred_ == truth_).all(axis=(0, 2)).mean()
            accuracy_temp.append(accuracy)
        print(f'Average accuracy of these preditons was: {np.mean(accuracy_temp)}')


def main(args):
    get_average_accuracy(result_attempt_dir)


if __name__ == '__main__':
    main(sys.argv)
