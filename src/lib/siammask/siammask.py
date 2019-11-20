import json

import numpy as np
import torch

from lib.datasets import SiammaskImage
from lib.siammask.experiments.siammask.custom import Custom
from lib.siammask.test import siamese_init, siamese_track
from lib.siammask.utils.load_helper import load_pretrain


class SM(object):

    def __init__(self):
        self._defaults = {
            'resume': "./lib/siammask/experiments/siammask/SiamMask_DAVIS.pth",     # path to latest checkpoint (default: none)
            'config': "./lib/siammask/experiments/siammask/config_davis.json",      # hyper-parameter of SiamMask in json format
            'conf_thres': 0.5,                                                      # object confidence threshold
        }

        print("siammask 모델 로드...")

        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        # Setup Model
        self.cfg = json.load(open(self._defaults['config']))
        self.model = Custom(anchors=self.cfg['anchors'])
        self.model = load_pretrain(self.model, self._defaults['resume'])
        self.model.eval().to(device)   # Set in evaluation mode

    def track_image(self, tag, path, step):

        # result
        result = []     # track_object
        imgs = []       # Stores image paths

        # ROI
        label = tag[1][0]
        x, y = tag[1][2]
        w, h = tag[1][3]

        # Start Frame (등장 프레임)
        start = tag[0]
        dataloader = SiammaskImage(path, start, step=step)

        for f, (img_paths, input_imgs) in enumerate(dataloader):

            if f == 0:  # init
                target_pos = np.array([x + w / 2, y + h / 2])
                target_sz = np.array([w, h])
                state = siamese_init(input_imgs, target_pos, target_sz, self.model, self.cfg['hp'])  # init tracker
            elif f > 0:  # tracking
                state = siamese_track(state, input_imgs, mask_enable=True, refine_enable=True)  # track

                if state['score'].max() < self._defaults['conf_thres']:
                    break

                x = state['target_pos'][0] - state['target_sz'][0]/2
                y = state['target_pos'][1] - state['target_sz'][1]/2
                w = state['target_sz'][0]
                h = state['target_sz'][1]

                result.append(
                    [f, [label, state['score'].max(), (x, y), (w, h)]]
                )

                imgs.append(img_paths)

        return result, imgs
