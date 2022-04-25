import os
import sys
import cv2 as cv
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_curve


def get_average_accuracy(gtmFolderPath, pdmFolderPath):
    gtmFiles = sorted([
        os.path.join(gtmFolderPath, fname)
        for fname in os.listdir(gtmFolderPath)
        if fname.endswith('.png')
    ])
    pdmFiles = sorted([
        os.path.join(pdmFolderPath, fname)
        for fname in os.listdir(pdmFolderPath)
        if fname.startswith('frame')
    ])
    recall, precision, f1Array = [], [], []
    for i in range(0, len(pdmFiles)):
        # pred = (cv.imread(pdmFiles[i], cv.IMREAD_GRAYSCALE) / 255).astype(int).flatten()
        # ground = cv.resize(cv.imread(gtmFiles[i + 1], cv.IMREAD_GRAYSCALE), (320, 180)).astype(int).flatten()
        # cv.imshow('a', cv.imread(gtmFiles[i + 1], cv.IMREAD_GRAYSCALE))
        # cv.moveWindow('a', 100, 100)
        # cv.waitKey(10)
        predTemp = (cv.imread(pdmFiles[i], cv.IMREAD_GRAYSCALE) / 255).astype(int).flatten()
        groundTemp = (cv.imread(gtmFiles[i + 1], cv.IMREAD_GRAYSCALE) / 255).astype(int).flatten()
        pred = []
        ground = []
        for j in range(0, len(predTemp)):
            if predTemp[j] == 0:
                pass
            else:
                pred.append(predTemp[j])
                ground.append(groundTemp[j])

        if len(pred) != 0:
            result = precision_recall_fscore_support(ground, pred, average='weighted', zero_division=0)
            precision_temp, recall_temp = result[0], result[1]
            recall.append(recall_temp)
            precision.append(precision_temp)
            f1Array.append(f1_score(ground, pred, average='weighted', zero_division=0))
            print(f'INFO: Index: {i}, Pred: {len(pred)}, Ground: {len(ground)}, recall: {recall_temp:04f}, '
                  f'precision: {precision_temp:04f}, f1: {f1Array[-1]:04f}')
    print(f'Recall: {sum(recall) / len(recall)}')
    print(f'Precision: {sum(precision) / len(precision)}')
    print(f'F1-score: {sum(f1Array) / len(f1Array)}')
    return sum(f1Array) / len(f1Array)


def main(args):
    gtmPath = 'M:/MAI_dataset/ModelEval/ED/mask/'
    pdmPath = 'M:/MAI_dataset/ModelEval/ED/pred/'
    get_average_accuracy(gtmPath, pdmPath)


if __name__ == '__main__':
    main(sys.argv)
