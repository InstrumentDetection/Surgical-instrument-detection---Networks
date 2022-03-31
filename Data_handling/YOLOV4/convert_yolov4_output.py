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

def convertJsonToXmls(resultPath,putPath):

    image_path = putPath
    if not path.exists(image_path):
        os.mkdir(image_path)
    if not image_path.endswith('/'):
        image_path = image_path + '/'

    f = open(resultPath)
    data = json.load(f)
    for p in range(0,len(data)):

        imgName = data[p]['filename'].split('/')[-1]
        imgPath = data[p]['filename']
        txtName = imgName.split(".")[0]+".txt"
        write_object  = open(image_path+txtName, "w+")

  


        for i in data[p]['objects']:
            
            name = i['class_id']
            print(i['name'])
            cords = i["relative_coordinates"]
            print(cords)

            x= cords["center_x"]
            y= cords["center_y"]
            if x <0:
                x = 0
            if y<0:
                y=0
            w= cords["width"]
            h= cords["height"]
            conf = i["confidence"]


            # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
            # via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380


            #write_object.write(str(name)+" " + str(round(x,4)) +" " + str(round(y,4)) +" "+str(round(w,4)) + " " + str(round(h,4))  +" "+str(round(conf,4))  +"\n" )

            write_object.write(str(name)+" " +str(round(conf,4))+" "+ str(round(x,4)) +" " + str(round(y,4)) +" "+str(round(w,4)) + " " + str(round(h,4))    +"\n" )

    f.close()

def main():
    #Input a json file to generate txt files for each image.
    # Remember conf_thresh = 0.25 for conf matrix and 0.005 for interference.
    args = get_args()
    convertJsonToXmls(args.json_path,args.put_path)

if __name__ == '__main__':
    main()


