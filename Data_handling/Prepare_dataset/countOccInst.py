""" usage: partition_dataset.py [-h] [-i IMAGEDIR] [-o OUTPUTDIR] [-r RATIO] [-x]
python3 partition_dataset.py -x -i '/home/serge/repos/TensorFlow/workspace/training_demo/images' -r 0.1 -o '/home/serge/repos/TensorFlow/workspace/training_demo/images'
Partition dataset of images into training and testing sets

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGEDIR, --imageDir IMAGEDIR
                        Path to the folder where the image dataset is stored. If not specified, the CWD will be used.
  -o OUTPUTDIR, --outputDir OUTPUTDIR
                        Path to the output folder where the train and test dirs should be created. Defaults to the same directory as IMAGEDIR.
  -r RATIO, --ratio RATIO
                        The ratio of the number of test images over the total number of images. The default is 0.1.

  -e, --ending          The file ending of your annotation file, e.g. txt or xml
"""
import os
import re
from shutil import copyfile
import argparse
import math
import random
import glob
import os.path
import shutil
#from xxlimited import Str
import xml.etree.ElementTree as ET


def iterate_dir(source):
    itemdict = {
    "diatermi": 0,
    "forceps": 0,
    "head": 0,
    "scalpel": 0,
    "needle driver": 0,
    "retractor": 0,
    "hl tube": 0,
    "saw": 0
    }

    if not source.endswith('/'):
        source = source + '/'


    image_ids =[os.path.basename(f) for f in glob.glob(source + '*.xml')]
    #print(image_ids)
    for file in image_ids:
        tree = ET.parse(source+file)
        root = tree.getroot()
        #print("obj: ",file)
        for obj in root.iter('object'):
            #print(obj.find('name').text)
            itemdict[obj.find('name').text] +=1
    return itemdict
           
            
            
        


def main():

    # Initiate argument parser
    parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--imageDir',
        help='Path to the folder where the image dataset is stored. If not specified, the CWD will be used.',
        type=str,
        default=os.getcwd()
    )
   
        
        
    args = parser.parse_args()

    
    # Now we are ready to start the iteration
    print(iterate_dir(args.imageDir))


if __name__ == '__main__':
    main()