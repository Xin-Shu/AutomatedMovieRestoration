import os
import sys
import cv2 as cv
from tqdm import tqdm


FOLDER_PATH = 'M:/MAI_dataset/Cinecitta/Sequence_lines_1'
FILM_NAME = 'Cinecitta'
fps = 30


class OriginalFilm:
    def __init__(self):
        self.video = []
        self.resolution = ()
        self.play_the_original_film(FOLDER_PATH)
        self.store_video(FILM_NAME)

    def play_the_original_film(self, input_path):
        frame_paths = sorted([
            os.path.join(input_path, fname)
            for fname in os.listdir(input_path)
            if fname.endswith('.bmp')]
        )
        self.resolution = cv.imread(frame_paths[0]).shape[0:2]
        print(self.resolution)
        for frame_path in tqdm(frame_paths, bar_format='{percentage:3.0f}%|{bar:100}{r_bar}'):
            frame = cv.imread(frame_path, cv.IMREAD_GRAYSCALE)
            self.video.append(frame)
            cv.waitKey(int(1000 / fps))
        print("INFO: Done forming video...")

    def store_video(self, film_name):
        print("INFO: Storing video...")
        try:
            fourcc = cv.VideoWriter_fourcc(*'H264')
            output_video = cv.VideoWriter(f'{FOLDER_PATH}/{film_name}.mp4', fourcc, 5, (1828, 1332))
            for i in range(len(self.video)):
                output_video.write(self.video[i])
            output_video.release()
            print("INFO: Done.")
        except NameError:
            print("ERROR: Failed to store played video. \n" + NameError)


def main(args):
    play_film = OriginalFilm()


if __name__ == '__main__':
    main(sys.argv)
