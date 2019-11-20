import argparse
import os
import cv2

from src.lib.datasets import VideoLoader


class video2frame():
    # 비디오 파일을 프레임 별로 나눠서 저장합니다.
    def __init__(self, **kwargs):
        self.input = kwargs['input']
        self.output = kwargs['output']
        self.sampling = 3               # 3 프레임당 하나 찍기

    def run(self):
        videos = VideoLoader(self.input, self.output)

        for video, output in videos:
            print(video, output)
            os.makedirs(output, exist_ok=True)

            capture = cv2.VideoCapture(video)
            cnt = 0
            save_cnt = 0
            while capture.isOpened():
                success, image = capture.read()
                if not success:
                    break

                cnt += 1

                if not cnt % self.sampling == 0:
                    continue

                save_cnt += 1

                cv2.imwrite("{}/frame{}.jpg".format(output, save_cnt), image)
                print("save : {}/frame{}.jpg".format(output, save_cnt))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument(
        '--input', type=str
    )

    parser.add_argument(
        '--output', type=str
    )

    FLAGS = parser.parse_args()

    # start process
    if not 'input' in FLAGS or not os.path.exists(FLAGS.input):
        print("입력 경로가 없습니다.")
    elif not 'output' in FLAGS:
        print("출력 경로가 없습니다.")
    else:
        # run
        video2frame(**vars(FLAGS)).run()
