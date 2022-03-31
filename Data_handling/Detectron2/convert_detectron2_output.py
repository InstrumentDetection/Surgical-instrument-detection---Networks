import cv2
import matplotlib.pyplot as plt
import json
import numpy as np
from PIL import Image
import os.path
from os import path
import argparse
import re


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--input_file', required=True, action='store', default='.', help="The network output, a txt-file")
    parser.add_argument('-o', '--output_folder', required=True, action='store', default=None, help="Where the extracted predictions will be places")
    return parser.parse_args()



def extract_predictions(input_file, output_folder):
    if not output_folder.endswith("/"):
        output_folder = output_folder + "/"

    with open(input_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            #Remove all new-lines, tabs and whitespaces
            line = re.sub(r"[\n\t\s]*", "", line)
            
            parts = line.split("!!!")
            img_path = parts[0].replace("image:", "")
            img_name = img_path.split("/")[-1].replace(".jpg","")
            comma_parts = parts[1].split(",")
            num_instances = int(comma_parts[0].split("num_instances=")[1])
            if num_instances > 0:
                image_height = int(comma_parts[1].split("image_height=")[1])
                image_width = int(comma_parts[2].split("image_width=")[1])
                predictions = parts[1].split("fields=[pred_boxes:Boxes(tensor(")[1].split(",device='cuda:0')),s")[0]
                predictions = predictions.replace("[","")
                predictions = predictions.replace("]", "")
                predictions = predictions.split(",")
                confidence = parts[1].split("scores:tensor([")[1].split("],device='cuda:0'),p")[0]
                confidence = confidence.split(",")
                #cleaned_confidence = [float(x) for x in confidence]
                classes = parts[1].split("pred_classes:tensor([")[1].split("],device='cuda:0')])")[0]
                classes = classes.split(",")
                #cleaned_classes = [int(x) for x in classes]

                #Convert bounding box values, currently they are in the format xmin, ymin, xmax, ymax
                #converting to normalized xcenter, ycenter, width, height
                #print("predictions: ", predictions)
                #print("confidence: ", confidence)
                #print("classes: ", classes)
                #print("\n")
                predictions = [float(x) for x in predictions]
                for i in range(num_instances):
                    xmin = predictions[4*i]
                    ymin = predictions[4*i + 1]
                    xmax = predictions[4*i + 2]
                    ymax = predictions[4*i + 3]
                    predictions[4*i] = ((xmin + xmax)/2)/image_width      #xcenter
                    predictions[4*i + 1] = ((ymin + ymax)/2)/image_height #ycenter
                    predictions[4*i + 2] = (xmax-xmin)/image_width        #width
                    predictions[4*i + 3] = (ymax-ymin)/image_height       #height
                



                with open(output_folder+img_name+".txt", "w+") as pred:

                    for i in range(num_instances):
                        pred.write(classes[i] + " ")
                        pred.write(confidence[i] + " ")
                        for j in range(4):
                            pred.write(str(predictions[4*i+j]) + " ")
                        pred.write("\n")
                        
            else:
                with open(output_folder+img_name+".txt", "w+") as pred:
                    pass






if __name__ == "__main__":
    args = get_args()
    extract_predictions(args.input_file, args.output_folder)