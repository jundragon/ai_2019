import os
import random

import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from glob import glob
from PIL import Image
import numpy as np
import torch.nn.functional as F

from lib.utils import load_targetlabel
from lib.yolo3.utils.augmentations import horisontal_flip


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class SiammaskImage(Dataset):
    def __init__(self, folder_path, start, step=1):
        self.files = sorted(glob("%s/*.jp*" % folder_path))[start::step]

    def __getitem__(self, index):
        img_path = self.files[index]
        img = cv2.imread(img_path)

        return img_path, img

    def __len__(self):
        return len(self.files)


class YoloImage(Dataset):
    def __init__(self, folder_path, step=1, img_size=416):
        self.files = sorted(glob("%s/*.jp*" % folder_path))[::step]
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class DrawImage(Dataset):
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        # input load
        self.input_files = sorted(glob("%s/*.jp*" % input_dir))
        # output
        self.output_files = [x.replace(input_dir, output_dir) for x in self.input_files]

    def __getitem__(self, index):
        img_path = self.input_files[index % len(self.input_files)]
        img = cv2.imread(img_path)

        save_path = self.output_files[index % len(self.input_files)]

        return img, save_path

    def __len__(self):
        return len(self.files)


class Folder(Dataset):

    def __init__(self, input_dir):
        self.folders = sorted([x for x in glob("%s/*" % input_dir) if os.path.isdir(x)])

    def __getitem__(self, item):
        return self.folders[item]


class VideoLoader(Dataset):

    def __init__(self, input_dir, output_dir):
        self.files = sorted(glob("%s/*.mp4" % input_dir))
        self.output = output_dir

    def __getitem__(self, item):
        output = "{}/{}".format(self.output, self.files[item].split('/')[-1].split('.')[0])
        return self.files[item], output


class CocoDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index].rstrip()

        # Extract image as PyTorch tensor
        try:
            img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        except:
            # 실패하면 다음 이미지로
            return self.__getitem__((index+1) % len(self.img_files))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------
        # TODO: 커스텀 데이터 셋을 만들 때, 라벨을 여기서 정리합니다.

        label_path = self.label_files[index].rstrip()

        targets = None
        if os.path.exists(label_path):


            # FIXME: coco dataset 을 이용할 때만 사용합니다.
            # coco dataset 에서 6개 라벨 box 를 매칭
            #  my custom set              ||  coco set
            # ====================== || ======================
            # [0] person             ||  [0] person
            # [1] fire extinguisher  ||   없음
            # [2] fire hydrant       ||  [10] fire hydrant
            # [3] car                ||  [2] car, [5] bus, [7] truck
            # [4] bicycle            ||  [1] bicycle
            # [5] motorbike          ||  [3] motorbike

            if 'coco' in label_path:
                target_labels = [0, 1, 2, 3, 5, 7, 10]
                trans_labels = [0, 4, 3, 5, 3, 3, 2]

                labels = load_targetlabel(label_path=label_path, target_labels=target_labels, trans_labels= trans_labels)
            else:
                labels = np.loadtxt(label_path).reshape(-1, 5)


            if len(labels) <= 0:
                # 없으면 다음 이미지로
                return self.__getitem__((index + 1) % len(self.img_files))

            # boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5)) # TODO: 기존 코드 남겨둠
            boxes = torch.from_numpy(labels)

            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                try:
                    img, targets = horisontal_flip(img, targets)
                except:
                    # 없으면 다음 이미지로
                    return self.__getitem__((index + 1) % len(self.img_files))

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]

        if targets is None or len(targets) <= 0:
            return paths, imgs, targets

        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)


class CocoDataset_trans(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index].rstrip()

        # Extract image as PyTorch tensor
        try:
            img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        except:
            # 실패하면 다음 이미지로
            return self.__getitem__((index+1) % len(self.img_files))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------
        # TODO: 커스텀 데이터 셋을 만들 때, 라벨을 여기서 정리합니다.

        label_path = self.label_files[index].rstrip()

        targets = None
        if os.path.exists(label_path):


            # FIXME: coco dataset 을 이용할 때만 사용합니다.
            # coco dataset 에서 6개 라벨 box 를 매칭
            #  my custom set              ||  coco set
            # ====================== || ======================
            # [0] person             ||  [0] person
            # [1] fire extinguisher  ||   없음
            # [2] fire hydrant       ||  [10] fire hydrant
            # [3] car                ||  [2] car, [5] bus, [7] truck
            # [4] bicycle            ||  [1] bicycle
            # [5] motorbike          ||  [3] motorbike

            if 'coco' not in label_path:
                target_labels = [0, 4, 3, 5, 3, 3, 2]
                trans_labels = [0, 1, 2, 3, 5, 7, 10]

                labels = load_targetlabel(label_path=label_path, target_labels=target_labels, trans_labels= trans_labels)
            else:
                labels = np.loadtxt(label_path).reshape(-1, 5)


            if len(labels) <= 0:
                # 없으면 다음 이미지로
                return self.__getitem__((index + 1) % len(self.img_files))

            # boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5)) # TODO: 기존 코드 남겨둠
            boxes = torch.from_numpy(labels)

            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                try:
                    img, targets = horisontal_flip(img, targets)
                except:
                    # 없으면 다음 이미지로
                    return self.__getitem__((index + 1) % len(self.img_files))

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]

        if targets is None or len(targets) <= 0:
            return paths, imgs, targets

        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)