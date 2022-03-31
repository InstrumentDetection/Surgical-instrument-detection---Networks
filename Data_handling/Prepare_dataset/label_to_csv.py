import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse
import numpy as np
import shutil
import pandas as pd
import os, sys, random
import xml.etree.ElementTree as ET
import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from PIL import Image

##Transforms xml file to csv format.

def xml_to_csv(path,out_path,train_or_test):

    input_path=path
    output_path = out_path
    if not input_path.endswith('/'):
        input_path = input_path + '/'
    if not output_path.endswith('/'):
        output_path = output_path + '/'

    data=pd.DataFrame(columns=['fileName','xmin','ymin','xmax','ymax','class'])

    os.getcwd()
    #read All files
    allfiles = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    #Read all pdf files in images and then in text and store that in temp folder 
    #Read all pdf files in images and then in text and store that in temp folder
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    image_ids =[os.path.basename(f) for f in glob.glob(input_path + '*.xml')]

    for file in image_ids:
        #print(file)

        tree = ET.parse(input_path+file)
        root = tree.getroot()
        fileName= input_path+root[1].text.split(".")[0]+".jpg"
        im = np.array(Image.open(fileName)) # pick a random image
        im = im[:,:,:3]
        for obj in root.iter('object'):
            cls_name = obj.find('name').text
            xml_box = obj.find('bndbox')
            xmin = xml_box.find('xmin').text
            ymin = xml_box.find('ymin').text
            xmax = xml_box.find('xmax').text
            ymax = xml_box.find('ymax').text
                        # Append rows in Empty Dataframe by adding dictionaries
            data = data.append({'fileName': fileName, 'xmin': xmin, 'ymin':ymin,'xmax':xmax,'ymax':ymax,'class':cls_name}, ignore_index=True)
            



    classes = [ "diatermi", "forceps","head","scalpel","needle driver", "retractor", "hl tube", "saw"] # own data sets which classes which category to write, in the order
    with open(output_path+'detectorClasses.csv', 'w') as f:
        for i, class_name in enumerate(classes):
            f.write(f'{class_name},{i}\n')         

    if not os.path.exists('snapshots'):
        os.mkdir('snapshots')

    data = data.iloc[:-1]
    data.to_csv(output_path+train_or_test+'Data.csv', index=None, header=False)
    print(data.head())


def main():

    # Initiate argument parser
    parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-f', '--folder',
        help='Path to the folder where the image dataset is stored. ',
        type=str,
        default=os.getcwd()
    )
    parser.add_argument(
        '-o', '--outputDir',
        help='Path to the output folder where the data csv file should be stored',
        type=str,
        default=None
    )
    parser.add_argument(
        '-t', '--train_or_test',
        help='Either test or train folder name',
        type=str,
        default=None
    )
    args = parser.parse_args()
    xml_to_csv(args.folder,args.outputDir,args.train_or_test)
 
 

    print('Successfully converted xml to csv.')

if __name__ == '__main__':
    main()