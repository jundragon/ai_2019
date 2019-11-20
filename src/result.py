import argparse
from glob import glob
import os
import pickle
from operator import itemgetter

from src.lib.utils import save_dict2json


class Result:
    # 다른 분석 없이 중간 저장 파일 들로 최종 결과를 생성하는 코드
    # 분석 오류시 답 제출 할 때.... 비상용으로 생성하는 코드를 작성
    def __init__(self, **kwargs):
        self.input = kwargs['input']
        self.output = kwargs['output']
        self.count = kwargs['count']
        self.fname = kwargs['fname']

        os.makedirs(self.output, exist_ok=True)

    def generate_result(self):
        # load intermediate file (xxx.save)
        load_files = sorted(glob("%s/*.save" % self.input))

        total_answer = []
        for file in load_files:
            with open(file, "rb") as fp:
                data = pickle.load(fp)
                total_answer.append(data)

        # 비어있는 부분 채우기
        sorted_answer = sorted(total_answer, key=itemgetter('id'))

        result_answer = []
        for i in range(1, self.count+1):

            answer = {
                'id': i,
                'objects': [0, 0, 0, 0, 0, 0]
            }

            for item in sorted_answer:
                if item['id'] == i:
                    answer = item
                    break

            result_answer.append(answer)

        # 최종 저장
        result_dict = {
            "track1_results": result_answer
        }

        save_dict2json(result_dict, '{}/{}'.format(self.output, self.fname))

        print("...success!!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument(
        '--input', type=str
    )

    parser.add_argument(
        '--output', type=str
    )

    parser.add_argument(
        '--fname', type=str, default="t1_result.json"
    )

    parser.add_argument(
        '--count', type=int, default=500
    )

    FLAGS = parser.parse_args()

    # start process
    if not 'input' in FLAGS or not os.path.exists(FLAGS.input):
        print("입력 경로가 없습니다.")
    elif not 'output' in FLAGS:
        print("출력 경로가 없습니다.")
    else:
        # run
        Result(**vars(FLAGS)).generate_result()


