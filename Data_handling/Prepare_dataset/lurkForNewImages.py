import os.path
import time



# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
from os import listdir
from os.path import join
import argparse
import glob 
import partition_dataset as partition
import list_files as listFiles
import label_to_csv as lblCsv
import convertXmlStrucToCoco as convCoco
import shutil






def startLooking(img_dir,dest_dir):
    label_dir = img_dir
    desti_dir = dest_dir
    print(img_dir)
    if not label_dir.endswith('/'):
        label_dir = label_dir + '/'
    if not desti_dir.endswith('/'):
        desti_dir = desti_dir + '/'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    file_path = label_dir +"predictions.jpg"
    i =0
    while(1):

        if os.path.exists(file_path):
            i = i+1
            shutil.move(file_path,desti_dir+"pred"+str(i)+".jpg" )
        else:
            time.sleep(0.01)


    
    
def main():
    parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-f', '--image_folder', required=True, action='store', default='.', help="Folder where predictions come.")

    parser.add_argument(
        '-o', '--outputDir',
        help='Path to the output folder where images should be placed',
        type=str,
        default=None
    )
   
    args = parser.parse_args()

    if args.outputDir is None:
        args.outputDir = args.imageDir
    startLooking(args.image_folder,args.outputDir)

if __name__ == '__main__':
    main()
