#!/usr/bin/env python
# -*- coding: utf8 -*-
import codecs
import os

TXT_EXT = '.txt'

class YoloWriter:

    def __init__(self, image_path=None):
        self.image_path = image_path
        self.boxlist = []
        self.imgSize = [1080, 1920]

    def addBndBox(self, xmin, ymin, xmax, ymax, name, difficult):
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['name'] = name
        bndbox['difficult'] = difficult
        self.boxlist.append(bndbox)

    def BndBox2YoloLine(self, box, classList=[]):
        xmin = box['xmin']
        xmax = box['xmax']
        ymin = box['ymin']
        ymax = box['ymax']

        xcen = float((xmin + xmax)) / 2 / self.imgSize[1]
        ycen = float((ymin + ymax)) / 2 / self.imgSize[0]

        w = float((xmax - xmin)) / self.imgSize[1]
        h = float((ymax - ymin)) / self.imgSize[0]

        # PR387
        boxName = box['name']
        if boxName not in classList:
            classList.append(boxName)

        classIndex = classList.index(boxName)

        return classIndex, xcen, ycen, w, h

    def save(self, classList=[], target_file=None):
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        with codecs.open(target_file, 'w', encoding='utf8') as out_file:
            for box in self.boxlist:
                classIndex, xcen, ycen, w, h = self.BndBox2YoloLine(box, classList)
                # print (classIndex, xcen, ycen, w, h)
                out_file.write("%d %.6f %.6f %.6f %.6f\n" % (classIndex, xcen, ycen, w, h))