import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader

from lib.datasets import YoloImage
from lib.yolo3.models import Darknet
from lib.yolo3.utils.utils import load_classes, non_max_suppression, rescale_boxes


class YOLO(object):

    def __init__(self, my_model=False):
        model_path = "./lib/yolo3/config/yolov3.cfg"
        weights_path = "./lib/yolo3/weights/yolov3.weights"
        class_path = "./lib/yolo3/data/coco.names"

        if my_model is True:
            print("my yolo 모델 로드...")
            model_path = "./lib/yolo3/config/yolov3-mine.cfg"   # 6 class classifications conf
            weights_path = "./checkpoints/yolov3_ckpt_1.pth"     # checkpoint weights (.pth)
            class_path = "./lib/yolo3/data/custom.names"          # 6 class classifications define
        else:
            print("yolo 모델 로드...")

        self._defaults = {
            'model_path': model_path,
            'weights_path': weights_path,
            'class_path': class_path,
            'n_cpu': 4,                     # number of cpu threads to use during batch generation
            'batch_size': 1,                # size of the batches
            'img_size': 416,                # size of each image dimension
            'img_shape': [1080, 1920],      # input image shape
            'conf_thres': 0.7,              # object confidence threshold
            'nms_thres': 0.4                # iou thresshold for non-maximum suppression
        }

        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device, "...ok")

        # Set up model
        self.model = Darknet(
            self._defaults['model_path'],
            img_size= self._defaults['img_size']
        ).to(device)

        if self._defaults['weights_path'].endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(self._defaults['weights_path'])
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(self._defaults['weights_path']))

        self.model.eval()   # Set in evaluation mode


    def detect_image(self, path, step, dt_thr=0.7):
        opt = self._defaults
        opt['conf_thres'] = dt_thr

        # dataset
        dataloader = DataLoader(
            YoloImage(folder_path=path, step=step, img_size=opt['img_size']),
            batch_size=opt['batch_size'],
            shuffle=False,
            num_workers=opt['n_cpu'],
        )

        classes = load_classes(opt['class_path'])  # Extracts class labels from file
        # target_classes = ['person', 'bicycle', 'car', 'motorbike', 'bus', 'truck', 'fire_hydrant']
        target_classes = ['person', 'fire extinguisher', 'fire hydrant', 'car', 'bicycle', 'motorbike']

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        # result
        result = []             # detect_object
        frames = []
        imgs = []               # Stores image paths
        img_detections = []     # Stores detections for each image index

        # toc = 0
        for batch_i, (img_paths, input_imgs) in enumerate(tqdm(dataloader, desc="Detecting objects")):
            # tic = cv2.getTickCount()

            i_shape = np.array(Image.open(''.join(img_paths))).shape[:2]

            # Configure input
            img = Variable(input_imgs.type(Tensor))

            # Get detections
            with torch.no_grad():
                detections = self.model(img)
                detections = non_max_suppression(detections, opt['conf_thres'], opt['nms_thres'])

            if detections[0] is None:
                continue

            # rescale_boxes(detections[0], opt['img_size'], opt['img_shape']) # Rescale boxes to original image
            rescale_boxes(detections[0], opt['img_size'], i_shape)  # Rescale boxes to original image

            dt_objects = []
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:

                w = x2 - x1
                h = y2 - y1

                if w < 32 or h < 32: # 단축 기준 32px 이하 결과는 무시
                    continue

                if classes[int(cls_pred)] in target_classes:
                    dt_objects.append(
                        [
                            classes[int(cls_pred)],
                            cls_conf.item(),
                            (x1, y1),
                            (w, h)
                        ]
                    )

            if len(dt_objects) > 0:
                # 탐지 결과가 있을 때만 저장
                result.append(dt_objects)
                imgs.extend(img_paths)
                img_detections.extend(detections)
                frames.append(batch_i)  # 탐지된 프레임 번호 저장

        #     toc += cv2.getTickCount() - tic
        #
        # toc /= cv2.getTickFrequency()
        # fps = batch_i / toc
        # print("Yolo Time: {:02.1f}s Speed: {:3.1f}fps".format(toc, fps))

        return result, imgs, frames
