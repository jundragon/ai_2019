import codecs
import os
from lxml import etree
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement


class PascalVocWriter:

    def __init__(self, image_path=None):
        self.image_path = image_path
        self.boxlist = []

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding='utf-8').replace("  ".encode(), "\t".encode())

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.image_path is None:
            return None

        top = Element('annotation')

        folder = SubElement(top, 'folder')
        folder.text = self.image_path.split('/')[-2]

        filename = SubElement(top, 'filename')
        filename.text = self.image_path.split('/')[-1]

        image_path = SubElement(top, 'path')
        image_path.text = self.image_path

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = 'Unknown'

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(1920)
        height.text = str(1080)
        depth.text = str(3)


        segmented = SubElement(top, 'segmented')
        segmented.text = '0'
        return top

    def addBndBox(self, xmin, ymin, xmax, ymax, name, difficult):
        bndbox = {'xmin': int(float(xmin)), 'ymin': int(float(ymin)), 'xmax': int(float(xmax)), 'ymax': int(float(ymax))}
        bndbox['name'] = name
        bndbox['difficult'] = difficult
        self.boxlist.append(bndbox)

    def appendObjects(self, top):

        for each_object in self.boxlist:
            # <object>
            object_item = SubElement(top, 'object')

            # # <name>
            name = SubElement(object_item, 'name')
            name.text = each_object['name']

            # <pose>
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"

            # <truncated>
            truncated = SubElement(object_item, 'truncated')
            if int(float(each_object['ymax'])) == int(float(1080)) or (int(float(each_object['ymin']))== 1):
                truncated.text = "1" # max == height or min
            elif (int(float(each_object['xmax']))==int(float(1920))) or (int(float(each_object['xmin']))== 1):
                truncated.text = "1" # max == width or min
            else:
                truncated.text = "0"

            # <difficult>
            difficult = SubElement(object_item, 'difficult')
            difficult.text = str( bool(each_object['difficult']) & 1 )

            # <bndbox>
            bndbox = SubElement(object_item, 'bndbox')
            # <xmin>
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            # <ymin>
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            # <xmax>
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            # <ymax>
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])

    def save(self, target_file=None):
        root = self.genXML()
        self.appendObjects(root)

        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        with codecs.open(target_file, 'w', encoding='utf8') as out_file:
            result = self.prettify(root)
            out_file.write(result.decode('utf8'))
