import cv2
import matplotlib.pyplot as plt
import json
import numpy as np
from PIL import Image
import os.path
from os import path
import argparse
import shutil

import xml.etree.cElementTree as ET



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-json', '--json_path', required=True, action='store', default='.', help="Where to retrieve json file")

    return parser.parse_args()

def convertJsonToXmls(json_path):

    f = open(json_path)
    data = json.load(f)
    for p in range(0,len(data)):
        txtName = data[p]['filename'].split('.')[0]+".txt"

        print(txtName)
        fout = open(txtName, "r")
        txtLines = fout.readlines() # array of file lines
        print(txtLines)
        newStrings=[]

        ind = 0
        for i in data[p]['objects'] :
            
            conf = i["confidence"]
            print(ind)
            if(len(txtLines)>ind):
                newStrings.append(txtLines[ind].rstrip("\n")+" "+ str(conf) + "\n")
            # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
            # via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380

            ind +=1
        with open(txtName,'w') as secondfile:
            for line in newStrings:                
                secondfile.write(line)

    f.close()




def main():
    args = get_args()
    convertJsonToXmls(args.json_path)

if __name__ == '__main__':
    main()


