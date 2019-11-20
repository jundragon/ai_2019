import json
import math
import numpy as np

from PIL import ImageDraw, Image
from lib.pascal_voc_writer import PascalVocWriter
from lib.yolo_writer import YoloWriter


def uniq(list):
    # 리스트 중복 제거
    return [x for i, x in enumerate(list) if x not in list[:i]]

def calc_intersection(this, other):
    # [x, y, w, h]
    x1, y1, x2, y2 = max(this[0], other[0]), max(this[1], other[1]), \
                     min(this[0] + this[2], other[0] + other[2]), \
                     min(this[1] + this[3], other[1] + other[3])

    if x1 < x2 and y1 < y2:
        return True, (x1, y1, x2 - x1, y2 - y1)  # 겹침
    else:
        return False, (0, 0, 0, 0)  # 겹치지 않음


def calc_distance(this, other):
    # [x, y, w, h]

    # 센터 계산
    center_x1, center_y1, center_x2, center_y2 = this[0] + this[2] / 2, this[1] + this[3] / 2, \
                                                 other[0] + other[2] / 2, other[1] + other[3] / 2
    # 거리 계산 (피타고라스)
    return math.sqrt(math.pow(center_x1 - center_x2, 2) + math.pow(center_y1 - center_y2, 2))


def calc_iou(this, other):
    # [x, y, w, h]
    this_x = max(this[0], other[0])
    this_y = max(this[1], other[1])
    other_x = min(this[0] + this[2], other[0] + other[2])
    other_y = min(this[1] + this[3], other[1] + other[3])

    # 각각의 너비를 구함
    this_area = this[2] * this[3]
    other_area = other[2] * other[3]

    # 겹치는 영역의 너비를 구함
    inter_area = max(0, other_x - this_x) * max(0, other_y - this_y)

    # 합쳐진 영역의 너비를 구함
    union_area = float(this_area + other_area - inter_area)

    # 일치하는 정도의 비율을 구함
    iou = float(inter_area / union_area)

    return iou, inter_area, union_area


def compare(obj1, obj2):
    ob1_x, ob1_y = obj1[2]
    ob1_w, ob1_h = obj1[3]

    ob2_x, ob2_y = obj2[2]
    ob2_w, ob2_h = obj2[3]

    box1 = [ob1_x, ob1_y, ob1_w, ob1_h]
    box2 = [ob2_x, ob2_y, ob2_w, ob2_h]

    inter, interbox = calc_intersection(box1, box2)
    distance = calc_distance(box1, box2)
    iou, overlap, union = calc_iou(box1, box2)

    return inter, {
        'iou': iou,
        'overlap': overlap,
        'union': union,
        'distance': distance,
        'interbox': interbox
    }

def open_image(file):
    try:
        im = Image.open(file)
    except:
        print("이미지 로드 에러")
        return False

    return im

def draw_image(image, obj, color):
    draw = ImageDraw.Draw(image)

    left, top = obj[2]
    width, height = obj[3]

    draw.rectangle([(left, top), (left+width, top+height)], outline=color)
    del draw

    return image


def export_xml(img_path, input, output, objects):
    xml_path = sorted( x.replace('.jpg', '.xml') for x in [x.replace(input, output) for x in img_path])

    for img, xml, obj_list in zip(img_path, xml_path, objects):
        writer = PascalVocWriter(image_path=img)
        for obj in obj_list:
            x, y = obj[2]
            w, h = obj[3]
            writer.addBndBox(x, y, x+w, y+h, obj[0], obj[1])

        writer.save(target_file=xml)

def export_txt(img_path, input, output, objects):
    txt_path = sorted(x.replace('.jpg', '.txt') for x in [x.replace(input, output) for x in img_path])

    for img, txt, obj_list in zip(img_path, txt_path, objects):
        writer = YoloWriter(image_path=img)
        for obj in obj_list:
            x, y = obj[2]
            w, h = obj[3]
            writer.addBndBox(x, y, x + w, y + h, obj[0], obj[1])

        writer.save(
            classList=['person',
                       'fire extinguisher',
                       'fire hydrant',
                       'car',
                       'bicycle',
                       'motorbike'
                       ],
            target_file=txt
        )


# json
def save_dict2json(dict, file):
    with open(file, 'w') as fp:
        json.dump(dict, fp, indent=4, sort_keys=True)
    # print("'{}'에 저장됨".format(file))

def load_json2dict(file):
    print("json file -> python dictionary")
    with open(file, 'r') as fp:
        dict = json.load(fp)
    print("keys: ", dict.keys())
    print(dict)
    return dict

def load_targetlabel(label_path, target_labels, trans_labels):
    # coco set label 을 custom set label 로 변경

    items = np.loadtxt(label_path).reshape(-1, 5)
    target_items = [item for item in items if item[0] in target_labels]

    if len(target_items) <= 0 :
        return target_items

    for item in target_items:
        item[0] = trans_labels[target_labels.index(item[0])]

    return np.array(target_items)