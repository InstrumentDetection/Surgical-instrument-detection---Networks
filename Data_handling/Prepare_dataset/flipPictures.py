from scipy import ndimage, misc

import numpy as np

import cv2
import argparse
import os

STEP = 1000
NAME_PREFIX = 'a'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--image_folder', required=True, action='store', default='.', help="folder")
    return parser.parse_args()

def create_imgs(in_folder):

    for file in os.listdir(in_folder):
        if(file.endswith(".jpg")):

            img = cv2.imread(in_folder + file)
            cv2.imshow('Testing', img)
            if img.shape[1]<img.shape[0]:
                img = ndimage.rotate(img, 90)
                cv2.imwrite(in_folder + file , img)
                print("changed", file)
def main():
    args = get_args()
    create_imgs(args.image_folder)

if __name__ == '__main__':
    main()