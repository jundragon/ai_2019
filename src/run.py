import argparse
import copy
import os
import pickle
from datetime import datetime
from glob import glob
from operator import itemgetter
import paramiko

import numpy as np
from PIL import Image, ImageDraw

from src.lib.datasets import Folder
from src.lib.sftp import get_ssh, ssh_execute, get_sftp, close_ssh, file_upload, close_sftp
from src.lib.utils import compare, uniq, save_dict2json, export_xml, export_txt
from lib.siammask.siammask import SM
from lib.yolo3.yolo import YOLO


class Task:

    def __init__(self, **kwargs):

        self.input = kwargs['input']
        self.output = kwargs['output']
        self.xml = kwargs['xml']
        self.txt = kwargs['txt']
        self.start = kwargs['start']
        self.step = kwargs['step']
        self.visual = kwargs['visual']
        self.count = kwargs['count']

        self.sftp = kwargs['sftp']
        self.sftp_home = kwargs['sftp_home']
        self.sftp_port = kwargs['sftp_port']
        self.sftp_id = kwargs['sftp_id']
        self.sftp_pw = kwargs['sftp_pw']

        os.makedirs(self.output, exist_ok=True)

        # init ml models
        self.yolo = YOLO(False)             # detect 용 (yolo)
        self.siammask = SM()                # track 용 (siammask)

        if self.visual is True:
            print("추적 결과 이미지를 출력합니다.")

    def detect(self, model, path, dt_thr=0.8):
        # 객체 검출
        return model.detect_image(path=path, step=self.step, dt_thr=dt_thr)

    def filter(self, model, tags, path, tr_thr=0.5):
        temp_tags = copy.deepcopy(tags)

        tr_step = 5

        for i, tag in enumerate(tags):
            # print("필터 중 {}/{}".format(i, len(temp_tags)))

            if tag not in temp_tags:
                continue

            #tr_objects, img = model.track_image(tag=tag, path=path, step=self.step)
            tr_objects, img = model.track_image(tag=tag, path=path, step=tr_step)

            if i+1 >= len(temp_tags):
                break

            remove_tags = []

            # 현재 트래킹 정보로 tag 를 필터링
            for idx, temp_tag in enumerate(temp_tags):
                if i >= idx or temp_tag[1][0] != tag[1][0]:
                    # 이전 tag 이고 label 이 다
                    continue

                if temp_tag[0] - tag[0] <= 0:
                    # 같은 frame 이 거나 이전 프레임
                    continue

                frame_diff = temp_tag[0] - tag[0]   # 프레임 차이

                find_idx = int(frame_diff * self.step / tr_step)

                # compare object
                if len(tr_objects) < find_idx:
                    # 추적 객체 없음
                    continue

                tr_object = tr_objects[find_idx-1]

                b_inter, parms = compare(temp_tag[1], tr_object[1])
                # if b_inter and parms['iou'] > tr_thr:
                if b_inter: # 겹치면 바로 제거 FIXME
                    remove_tags.append(temp_tag)
                    # print("remove tag ", parms['iou'], "\t", temp_tag, "\t", tr_object)

            # 안전하게 for문 밖에서 제거
            if len(remove_tags) > 0:
                for remove_tag in remove_tags:
                    temp_tags.remove(remove_tag)

        return temp_tags

    def tag(self, frames, dt_objects, tg_thr=0.2):

        tags = []           # 이름표
        rois = []           # 추적 대상

        # 검출 객체 태깅
        for i, (frame, objects) in enumerate(zip(frames, dt_objects)):

            if i == 0:
                # 탐지된 첫번째 프레임은 탐지 객체를 모두 등록
                rois.append(self.register(tags, frame, objects))
            else:
                cur_roi = copy.deepcopy(rois[i-1])
                compare_result = []
                next_roi = []

                # compare
                if len(cur_roi) > 0 and len(objects) > 0:
                    for obj in objects:
                        for roi in cur_roi:
                            if not obj[0] == roi[1][0]: # label 이 다르면 compare 하지 않는다.
                                continue

                            b_inter, parms = compare(obj, roi[1])

                            # if b_inter and parms['iou'] > tg_thr:
                            if b_inter:
                                compare_result.append([parms['iou'], obj, roi])

                if len(compare_result) > 0:
                    compare_result.sort(key=lambda el: el[0], reverse=True)
                    compare_result = uniq(compare_result)  # 중복 제거

                    # tag
                    for x in compare_result:
                        if x[2] in cur_roi:
                            next_roi.append([x[2][0], x[1]])  # 비교 결과에 현재 ROI 가 있으면 Update
                            del (cur_roi[cur_roi.index(x[2])])

                # register
                if len(compare_result) > 0:
                    next_roi.extend(
                        self.register(tags, frame,
                                      [x for x in objects if x not in np.array(compare_result)[:, 1].tolist()])
                    )
                else:
                    # 현재 비교대상 roi가 없으면 탐지된 객체를 모두 다음 roi 로 등록
                    next_roi.extend(
                        self.register(tags, frame, [x for x in objects])
                    )

                # 태그에 실패한 roi를 저장 (10프레임 이상 추적이 실패 했다면 저장 안함) FIXME
                # next_roi.extend(cur_roi)
                rois.append(next_roi)

        return tags, rois

    def register(self, tags, frame, objects):
        # 신규 객체 이름표 등록
        result = []
        for obj in objects:
            tags.append([frame, obj])         # 첫 등장 프레임과 라벨 정보로 태그 생성
            result.append([len(tags)-1, obj])    # tag_num, object
        return result

    def save_intermediate(self, video, output_dir, targets, tags, rois):
        # 분석 중간 파일을 저장합니다.
        # target 저장
        target_file = "{0}/{1}.{2}".format(output_dir, video, "target")
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        with open(target_file, 'wb') as fp:
            pickle.dump(targets, fp)

        # tags 저장
        tag_file = "{0}/{1}.{2}".format(output_dir, video, "tag")
        os.makedirs(os.path.dirname(tag_file), exist_ok=True)
        with open(tag_file, 'wb') as fp:
            pickle.dump(tags, fp)

        # rois 저장
        # roi_file = "{0}/{1}.{2}".format(output_dir, video, "roi")
        # os.makedirs(os.path.dirname(roi_file), exist_ok=True)
        # with open(roi_file, 'wb') as fp:
        #     pickle.dump(rois, fp)

    def draw_result(self, video, targets, tags):
        print("결과 그리기")
        total_imgs = sorted(glob("%s/*.jpg" % video))
        output_imgs = sorted(x.replace(self.input, self.output) for x in total_imgs)

        images = []
        img_idx = 0

        # target draw
        for i, target in enumerate(targets):
            if i == 0:
                # 새로 읽어오기
                if target[0] == 0:
                    img_idx = (target[0]+1) * self.step - self.step
                else:
                    img_idx = target[0] * self.step
                im = Image.open(total_imgs[img_idx])

            if (target[0] * self.step) != img_idx:
                # 저장
                images.append([im, output_imgs[img_idx]])

                # 새로 읽어오기
                if target[0] == 0:
                    img_idx = (target[0] + 1) * self.step - self.step
                else:
                    img_idx = target[0] * self.step
                im = Image.open(total_imgs[img_idx])

            color = "blue"
            if target not in tags:
                # 태그에 없는 건 빨강으로 그린다
                color = "red"

            draw = ImageDraw.Draw(im)
            x, y = target[1][2]
            w, h = target[1][3]
            draw.rectangle([(x, y), (x+w, y+h)], outline=color, width=5)

            if i == len(targets)-1:
                images.append([im, output_imgs[img_idx]])

        # save
        for img in images:
            os.makedirs(os.path.dirname(img[1]), exist_ok=True)
            img[0].save(img[1])

    def run_video(self, video):
        print("{0} 분석중...".format(video))
        start_time = datetime.now()

        dt_thr = 0.8  # detect 판정 민감도
        tg_thr = 0.2  # tag 판정 민감도
        fl_thr = 0.1    # filter 판정 민감도

        # detect object
        dt_objects, img_path, frames = self.detect(model=self.yolo, path=video, dt_thr=dt_thr)

        # tag object
        targets, rois = self.tag(frames, dt_objects, tg_thr=tg_thr)

        print("필터 전 검출 결과: ", len(targets))

        if len(targets) > 0 :
            # track object
            tags = self.filter(self.siammask, targets, video, tr_thr=fl_thr)

            print("필터 후 검출 결과: ", len(tags))
        else:
            tags = targets

        print("비디오 분석에 걸린 시간: {0}".format(datetime.now() - start_time))

        if self.visual is True:
            self.draw_result(video=video, targets=targets, tags=tags)

        # save intermediate
        # self.save_intermediate(video=video.split('/')[-1], output_dir=self.output, targets=targets, tags=tags, rois=rois)

        return dt_objects, tags, rois, img_path

    def run(self):

        start_time = datetime.now()

        if self.sftp is not '':
            ssh, sftp = self.connect()

        # make dummy answer
        temp_list = sorted(glob("{0}/*".format(self.input)))
        dummy_idx = int(temp_list[0].split('/')[-1].split('_')[-1])
        total_answer = [{'id': x, 'objects': [0, 0, 0, 0, 0, 0]} for x in range(dummy_idx, self.count+dummy_idx)]

        for i, video in enumerate(Folder(self.input)[self.start:]):
            # 분석 할 때 마다 중간 파일을 체크 후 load
            loadinfo_list = []
            for file in sorted(glob("{0}/*.save".format(self.output))):
                with open(file, 'rb') as fp:
                    idx = file.split('/')[-1].split('_')[-1].replace('.save', '')
                    data = pickle.load(fp)
                    total_answer[int(idx)] = data
                    # loadinfo_list.append(int(idx))

                    loadinfo_list.append(file.split('/')[-1].replace('.save', ''))

            if video.split('/')[-1] in loadinfo_list :
                continue

            file_id = video.split('/')[-1].split('_')[-1]

            # run video
            dt_objects, tags, rois, img_path = self.run_video(video=video)

            # xml export (VOC Format)
            if self.xml:
                print("xml 출력")
                export_xml(img_path, self.input, self.output, dt_objects)

            # txt export (YOLO Format)
            if self.txt:
                print("txt 출력")
                export_txt(img_path, self.input, self.output, dt_objects)

            # tag save
            if len(tags) > 0:
                tag_objects = np.array(tags)[:, 1].tolist()
                tag_objects = np.array(tag_objects)[:, 0].tolist()

                answer = {
                    "id": int(file_id),
                    "objects": [
                        tag_objects.count('person'),
                        tag_objects.count('fire extinguisher'),
                        tag_objects.count('fire hydrant'),
                        tag_objects.count('car'),
                        tag_objects.count('bicycle'),
                        tag_objects.count('motorbike')
                    ]
                }
            else:
                # 탐지 못함
                answer = {
                    "id": int(file_id),
                    "objects": [0, 0, 0, 0, 0, 0]
                }

            total_answer[i] = answer

            # 비디오마다 중간 저장
            target_file = "{0}/{1}.save".format(self.output, video.split('/')[-1])
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            with open(target_file, 'wb') as fp:
                pickle.dump(answer, fp)

            if self.sftp is not '':
                # file_upload(sftp, "/data/input/t1_gt.json", "/home/bbb.txt")
                # os.chmod(target_file, 755)
                # os.chown(target_file, 1000, 1000)
                file_upload(sftp, target_file, self.sftp_home+target_file)

            # 최종 저장
            result_dict = {
                "track1_results": total_answer
            }

            save_dict2json(result_dict, '{}/t1_result.json'.format(self.output))

        if self.sftp is not '':
            close_ssh(ssh)
            close_sftp(sftp)

        print("총 걸린 시간: {0}".format(datetime.now() - start_time))
        print("...success!!")

    def connect(self):
        # ssh = get_ssh(self.sftp, 22, "sftpid", "sftppw")
        ssh = get_ssh(self.sftp, self.sftp_port, self.sftp_id, self.sftp_pw)
        sftp = get_sftp(ssh)

        return ssh, sftp



if __name__ == "__main__":

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument('--input', type=str) # 입력 데이터 경로
    parser.add_argument('--output', type=str) # 출력 경로
    parser.add_argument('--xml', default=False, action='store_true') # label xml 출력 유무 (voc format: labelImg 파일과의 호환)
    parser.add_argument('--txt', default=False, action='store_true') # label txt 출력 유무 (yolo format: yolo 전이학습 및 evaluation 시에 사용)
    parser.add_argument('--start', type=int, default=0) # 분석 프레임 폴더(video) 시작점
    parser.add_argument('--step', type=int, default=10) # 분석 프레임 step (프레임 건너뛰기)
    parser.add_argument('--visual', default=False, action='store_true') # 이미지 출력 여부
    parser.add_argument('--count', type=int, default=50) # 분석 미완료 더미 데이터 생성 수 (대회용)
    # sftp 설정
    parser.add_argument('--sftp', type=str, default='') # sftp 목적지 ip address
    parser.add_argument('--sftp_home', type=str, default='/home/') # sftp 목적지의 home
    parser.add_argument('--sftp_port', type=int, default=22) # sftp 목적지 port
    parser.add_argument('--sftp_id', type=str, default='sftpid') # sftp 접속 id
    parser.add_argument('--sftp_pw', type=str, default='sftppw') # sftp 접숙 password

    FLAGS = parser.parse_args()

    # start process
    if not 'input' in FLAGS or not os.path.exists(FLAGS.input):
        print("입력 경로가 없습니다.")
    elif not 'output' in FLAGS:
        print("출력 경로가 없습니다.")
    else:
        # run
        Task(**vars(FLAGS)).run()
