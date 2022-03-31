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
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--image_folder', required=True, action='store', default='.', help="folder")
    
    return parser.parse_args()

classes = [ "diatermi", "forceps","scalpel","needle driver", "retractor", "hl tube", "saw"] # own data sets which classes which category to write, in the order



def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
 
def convert_annotation(img_dir, label_dir, image_id):
    in_file = open(img_dir + '/%s.xml'%(image_id), encoding = 'utf-8')
    ##for obj in root.iter('object'):


    tree=ET.parse(in_file)
    root = tree.getroot()
    out_file = open(label_dir + '%s.txt'%(root[1].text.split(".",1)[0]), 'a')
    
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
 
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    


def convert_all(img_dir,cust_img_dir,dest_dir,network,ratio,ending):
    label_dir = img_dir
    desti_dir = dest_dir
    print(img_dir)
    if not cust_img_dir.endswith('/'):
        cust_img_dir = cust_img_dir + '/'
    if not label_dir.endswith('/'):
        label_dir = label_dir + '/'
    if not desti_dir.endswith('/'):
        desti_dir = desti_dir + '/'
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    
    image_ids =[os.path.basename(f) for f in glob.glob(label_dir + '*.JPG')]
    for i, image_id in enumerate(image_ids):
        os.rename(label_dir + image_id, (label_dir + image_id).split('.')[0] +".jpg" )


    if network == "yolov4":


        image_ids2 =[os.path.basename(f) for f in glob.glob(label_dir + '*.jpg')]
        for image_id in image_ids2:

            open(label_dir + '%s.txt'%(image_id.split(".",1)[0]), 'w')

        image_ids3 =[os.path.basename(f) for f in glob.glob(cust_img_dir + '*.jpg')]
        for image_id in image_ids3:

            open(cust_img_dir + '%s.txt'%(image_id.split(".",1)[0]), 'w')
        image_ids =[os.path.basename(f) for f in glob.glob(label_dir + '*.xml')]

        image_ids_office =[os.path.basename(f) for f in glob.glob(cust_img_dir + '*.xml')]
        print(image_ids_office)
        for image_id in image_ids:
            convert_annotation(img_dir, label_dir, image_id[:-4])
        for image_id in image_ids_office:
            convert_annotation(cust_img_dir, cust_img_dir, image_id[:-4])
        partition.iterate_dir(img_dir,cust_img_dir, dest_dir, ratio, ending)
        temp_dir = dest_dir
        if not temp_dir.endswith('/'):
            temp_dir = temp_dir + '/'


    
        listFiles.create_file(temp_dir,temp_dir+"test/", "test")
        listFiles.create_file(temp_dir,temp_dir+"train_only_surgery/", "train_only_surgery")
        listFiles.create_file(temp_dir,temp_dir+"train_all/", "train_all")

        listFiles.create_file(temp_dir,temp_dir+"val/", "val")



    
    
def main():
    parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-f', '--image_folder', required=True, action='store', default='.', help="Folder where annotated images are.")
    parser.add_argument('-cf', '--custom_image_folder', required=True, action='store', default='.', help="Folder where annotated images are.")

    parser.add_argument(
        '-o', '--outputDir',
        help='Path to the output folder where the train, validation  and test dirs should be created. '
             'Defaults to the same directory as IMAGEDIR.',
        type=str,
        default=None
    )

    args = parser.parse_args()

    if args.outputDir is None:
        args.outputDir = args.imageDir
    



    convert_all(args.image_folder,args.custom_image_folder,args.outputDir,"yolov4",0.1,"txt")

if __name__ == '__main__':
    main()
