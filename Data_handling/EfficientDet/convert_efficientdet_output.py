import cv2
import matplotlib.pyplot as plt
import json
import numpy as np
from PIL import Image
import os.path
from os import path
import argparse
from os import path

import xml.etree.cElementTree as ET



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-resJson', '--json_path_res', required=True, action='store', default='.', help="Where to retrieve json file")
    parser.add_argument('-insJson', '--json_path_ins', required=True, action='store', default='.', help="Where to retrieve json file")
    parser.add_argument('-o', '--output_folder', required=True, action='store', default='.', help="Where to retrieve json file")
    parser.add_argument(
        '-conf', '--conf',
        default=0.25,
        type=float)
    return parser.parse_args()

def convertJsonToXmls(resultJson, instanceJson,output_folder,conf_thresh):
    if not path.isdir(output_folder):
        os.mkdir(output_folder)
    else:

        files_in_directory = os.listdir(output_folder)
        filtered_files = [file for file in files_in_directory if file.endswith(".txt")]
        for file in filtered_files:
	        path_to_file = os.path.join(output_folder, file)
	        os.remove(path_to_file)

    f = open(instanceJson)
    data = json.load(f)
    print(data["images"][0])
    f2 = open(resultJson)
    data2 = json.load(f2)
    print(data2[2])



    for i in data2:
        image_ind = i["image_id"]-1
        x_center = abs((i["bbox"][0]+(i["bbox"][2]/2)))/data["images"][image_ind]["width"]
        y_center = abs((i["bbox"][1]+(i["bbox"][3]/2)))/data["images"][image_ind]["height"]
        x_min = abs((i["bbox"][0]))/data["images"][image_ind]["width"]
        y_min = abs((i["bbox"][1]))/data["images"][image_ind]["height"]
        width= abs((i["bbox"][2]))/data["images"][image_ind]["width"]
        height = abs((i["bbox"][3]))/data["images"][image_ind]["height"]
        confidence = i["score"]
        classInd = i["category_id"] -1

        if confidence >= conf_thresh and classInd <= 6:
            temp = open(output_folder +data["images"][i["image_id"]-1]["file_name"].split(".")[0]+".txt", "a")
            temp.write(str(classInd) +" " + str(confidence)+" "+ str(x_center) + " "+ str(y_center) + " " +str(width) +" " + str(height) + "\n")

        #temp.write
        #print(data["images"][i["image_id"]-1]["file_name"].split(".")[0]+".txt")

    

def main():
    args = get_args()
    convertJsonToXmls(args.json_path_res,args.json_path_ins,args.output_folder,args.conf)

if __name__ == '__main__':
    main()


