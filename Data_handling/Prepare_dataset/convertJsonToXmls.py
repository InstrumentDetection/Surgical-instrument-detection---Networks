import cv2
import matplotlib.pyplot as plt
import json
import numpy as np
from PIL import Image
import os.path
from os import path
import argparse


import xml.etree.cElementTree as ET



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--json_path', required=True, action='store', default='.', help="Where to retrieve json file")
    parser.add_argument('-p', '--put_path', required=True, action='store', default='.', help="Where to retrieve json file")

    return parser.parse_args()

def convertJsonToXmls(resultPath):

    f = open(resultPath)
    data = json.load(f)
    for p in range(0,len(data)):

        imgName = data[p]['filename'].split('/')[-1]
        imgPath = data[p]['filename']
    
        imgOrig = cv2.imread(data[p]['filename'])
        img = np.copy(imgOrig)
        dh, dw, depth = img.shape
        print("imgPath",imgPath.split(".")[0])

        root = ET.Element("annotation")

        folder = ET.SubElement(root, "folder")
        folder.text = "testfolder"

        filename = ET.SubElement(root, "filename")
        filename.text = imgName

        paths = ET.SubElement(root, "path")
        paths.text = imgPath


        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = "Unknown"

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(dw)
        ET.SubElement(size, "height").text = str(dh)
        ET.SubElement(size, "depth").text = str(depth)


        segmented = ET.SubElement(root, "segmented")
        segmented.text = "0"


        for i in data[p]['objects']:
            
            name = i['name']
            print(i['name'])
            cords = i["relative_coordinates"]

            print(cords)
            x= cords["center_x"]
            y= cords["center_y"]
            w= cords["width"]
            h= cords["height"]
            

            # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
            # via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380
            nx = int(float(x)*dw - dw*(w/2))
            ny = int(float(y)*dh - dh*(h/2))
            nw = int(float(w)*dw)
            nh = int(float(h)*dh)

            xmin = nx
            ymin = ny
            xmax = nx + nw
            ymax = ny + nh

            objectx = ET.SubElement(root, "object")

            ET.SubElement(objectx, "name").text = name
            ET.SubElement(objectx, "pose").text = "Unspecified"
            ET.SubElement(objectx, "truncated").text = "0"
            ET.SubElement(objectx, "difficult").text = "0"

            bndbox = ET.SubElement(objectx, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(xmin)
            ET.SubElement(bndbox, "ymin").text = str(ymin)
            ET.SubElement(bndbox, "xmax").text = str(xmax)
            ET.SubElement(bndbox, "ymax").text = str(ymax)


            tree = ET.ElementTree(root)
            tree.write(imgPath.split(".")[0]+".xml")

    f.close()

def main():
    args = get_args()
    convertJsonToXmls(args.json_path,args.put_path)

if __name__ == '__main__':
    main()


