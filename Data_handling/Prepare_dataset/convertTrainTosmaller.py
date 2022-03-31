# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
from os import listdir
from os.path import join
import argparse
import glob 
import Prepare_dataset.partition_dataset as partition
import list_files as listFiles
import label_to_csv as lblCsv
import convertXmlStrucToCoco as convCoco
import numpy as np
import random
classes = [ "diatermi", "forceps","scalpel","needle driver", "retractor", "hl tube", "saw"] # own data sets which classes which category to write, in the order



    


def convert_all(txt_dir,ratio,output_dir):
    f = open(txt_dir, "r") #Load file in any mode that's able to read, ie r, r+, w+ etc
    lines = f.readlines()
    if not output_dir.endswith('/'):
        output_dir = output_dir + '/'
    length= len(lines)
    indList=np.arange(0,length)
    numberSamp = round(ratio * length)
    print(numberSamp)
    sampleList=np.random.choice(length, size=numberSamp, replace=False)
    #sampleList  = random.sample(indList, numberSamp)
    write_object  = open(output_dir+"train_"+str(ratio)+".txt", "w+")
    for sample in sampleList:

        write_object.write(lines[sample])


    
    
def main():
    parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-f', '--txt_file', required=True, action='store', default='.', help="Folder where annotated images are.")

    parser.add_argument(
        '-r', '--ratio',
        help='Path to the output folder where the train and test dirs should be created. '
             'Defaults to the same directory as IMAGEDIR.',
        type=float,
        default=None
    )
    parser.add_argument(
        '-o', '--output',
        help='The file ending of your annotation file, e.g. txt or xml',
        type=str,
        default='txt'

    )
    args = parser.parse_args()

    



    convert_all(args.txt_file,args.ratio,args.output)

if __name__ == '__main__':
    main()
