import argparse
import os
from datetime import datetime

from src.lib.datasets import Folder
from src.lib.pascal_voc_writer import PascalVocWriter
from lib.yolo3.yolo import YOLO


class Label:
    # 샘플 데이터를 라벨링 하는 기능, yolo로 탐지합니다.
    def __init__(self, **kwargs):
        self.input = kwargs['input']
        self.output = kwargs['output']

        os.makedirs(self.output, exist_ok=True)

        # init ml models
        self.yolo = YOLO()  # detect 용 (yolo)

    def detect(self, model, path, dt_thr=0.7):
        # 객체 검출
        print(path, " 생성중...")
        return model.detect_image(path, dt_thr=dt_thr)

    def run(self):
        start_time = datetime.now()

        conf_thres = 0.7 # 라벨링은 70% 이상이면 탐지합니다.
        for i, video in enumerate(Folder(self.input)[:]):
            dt_objects, img_path, frames = self.detect(self.yolo, video, conf_thres)

            print("탐지 객체 수 : ", len(dt_objects))

            xml_path = sorted(x.replace('.jpg', '.xml') for x in [x.replace(self.input, self.output) for x in img_path])

            for img, xml, objects in zip(img_path, xml_path, dt_objects):
                writer = PascalVocWriter(image_path=img)
                for obj in objects:
                    x, y = obj[2]
                    w, h = obj[3]
                    writer.addBndBox(x, y, x + w, y + h, obj[0], obj[1])

                writer.save(target_file=xml)

        print("총 걸린 시간: {0}".format(datetime.now() - start_time))
        print("...success!!")


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
        Label(**vars(FLAGS)).run()


