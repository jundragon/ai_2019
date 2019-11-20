from __future__ import division

import argparse
from glob import glob

from torch.utils.data import DataLoader

from src.lib.datasets import CocoDataset
from lib.yolo3.models import *
from lib.yolo3.utils.parse_config import *
from lib.yolo3.utils.utils import *


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = CocoDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        if targets is None or len(targets) <= 0:
            continue

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    if len(sample_metrics) <= 0 :
        return 0,0,0,0,0

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="./lib/yolo3/config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="./lib/yolo3/config/custom.data", help="path to data config file")
    parser.add_argument("--models_path", type=str, default="/models/total", help="path to models file")
    parser.add_argument("--class_path", type=str, default="./lib/yolo3/data/custom.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, "...")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    for model_path in sorted(glob("%s/*.pth" % opt.models_path)):

        print("+++++++++++++++++++++++++++++++++++")
        print(model_path.split('/')[-1], " 의 성능 평가")

        # Initiate model
        model = Darknet(opt.model_def).to(device)
        if model_path.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(model_path)
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(model_path))

        print("Compute mAP...")

        precision, recall, AP, f1, ap_class = evaluate(
            model,
            path=valid_path,
            iou_thres=opt.iou_thres,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            img_size=opt.img_size,
            batch_size=8,
        )

        print("Average Precisions:")
        for i, c in enumerate(ap_class):
            print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

        print(f"mAP: {AP.mean()}")

        print("+++++++++++++++++++++++++++++++++++")
