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

    return parser.parse_args()

def convertJsonToXmls(resultPath):

    f = open(resultPath)
    data = json.load(f)
    index =0
    print("size", len(data))
    dee =0
    newList = []
    for p in data:
        if(p["score"]<0.25):
            dee+=1
        else:
            newList.append(p)

        index +=1

    print("size", dee)
    f.close()
    with open(resultPath.split(".")[0] +"threshed.json", "w") as outfile:
        for i in newList:
            print(i)
            outfile.write(json.dumps(i, indent = 4))

def main():
    args = get_args()
    convertJsonToXmls(args.json_path)

if __name__ == '__main__':
    main()


