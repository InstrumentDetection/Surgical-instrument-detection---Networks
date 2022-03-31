import os
import glob
import xml.etree.ElementTree as ET
import xmltodict
import json
from xml.dom import minidom
from collections import OrderedDict
import argparse
import glob 
import shutil

def XML2JSON(xml_path,out_path,train_or_test, network):
    if not xml_path.endswith('/'):
        xml_path = xml_path + '/'
    if not out_path.endswith('/'):
        out_path = out_path + '/'
    
    xmlFiles=glob.glob(os.path.join(xml_path, '*.xml'))
    attrDict = dict()

    attrDict["categories"]=[{"id":1,"name":"diatermi","supercategory":"none"},
                    {"id":2,"name":"forceps","supercategory":"none",},
                    {"id":3,"name":"scalpel","supercategory":"none",},
                {"id":4,"name":"needle driver","supercategory":"none"},
                {"id":5,"name":"retractor","supercategory":"none"},
                {"id":6,"name":"hl tube","supercategory":"none"},
                {"id":7,"name":"saw","supercategory":"none"}
                  ]
    images = list()
    annotations = list()
    image_id = 0
    id1 = 1
    for file in xmlFiles:    
        image_id = image_id + 1      
        annotation_path=file
        image = dict()
        doc = xmltodict.parse(open(annotation_path).read(), force_list=('object'))
        image['id'] = image_id
        image['file_name'] = str(doc['annotation']['filename']).split(".")[0] + ".jpg"
        image['height'] = int(doc['annotation']['size']['height'])
        image['width'] = int(doc['annotation']['size']['width'])
        
        print ("File Name: {} and image_id {}".format(file, image_id))
        images.append(image)
        
        if 'object' in doc['annotation']:
            for obj in doc['annotation']['object']:
                for value in attrDict["categories"]:
                    annotation = dict()          
                    if str(obj['name']) == value["name"]:
                        annotation["id"] = id1

                        annotation["image_id"] = image_id

                        x1 = int(obj["bndbox"]["xmin"])  - 1
                        y1 = int(obj["bndbox"]["ymin"]) - 1
                        x2 = int(obj["bndbox"]["xmax"]) - x1
                        y2 = int(obj["bndbox"]["ymax"]) - y1                         
                        annotation["bbox"] = [x1, y1, x2, y2]
                        annotation["segmentation"] = [[x1,y1,x1,(y1 + y2), (x1 + x2), (y1 + y2), (x1 + x2), y1]]

                        annotation["area"] = float(x2 * y2)
                        annotation["category_id"] = value["id"]
                        id1 +=1
                        annotation["iscrowd"] = 0

                        annotations.append(annotation)

        else:
            print("File: {} doesn't have any object".format(file))

        
    if network == "ssd":        

        attrDict["images"] = images    
        attrDict["annotations"] = annotations
        attrDict["type"] = "instances"
        print("hey")
        print(out_path +"COCO_dataset")
        if not os.path.exists(out_path +"COCO_dataset"):
            print("in gere")
            os.makedirs(out_path +"COCO_dataset")    
        if not os.path.exists(out_path +"COCO_dataset"+"/annotations"):
            os.makedirs(out_path +"COCO_dataset"+"/annotations")
        if not os.path.exists(out_path +"COCO_dataset"+"/train2017"):
            os.makedirs(out_path +"COCO_dataset"+"/train2017")
        if not os.path.exists(out_path +"COCO_dataset"+"/val2017"):
            os.makedirs(out_path +"COCO_dataset"+"/val2017")
        if not os.path.exists(out_path +"COCO_dataset"+"/test2017"):
            os.makedirs(out_path +"COCO_dataset"+"/test2017")

        if train_or_test =="train":

            jsonString = json.dumps(attrDict)
            with open(out_path +"COCO_dataset"+"/annotations"+"/"+"instances_train2017.json", "w") as f:
                f.write(jsonString)

            jpgFiles=glob.glob(os.path.join(xml_path, '*.jpg'))
            for jpgFile in jpgFiles:
                head, tail = os.path.split(jpgFile)
                shutil.copyfile(xml_path +tail, out_path +"COCO_dataset"+"/train2017/"+ tail)
        elif train_or_test =="val":

            jsonString = json.dumps(attrDict)
            with open(out_path +"COCO_dataset"+"/annotations"+"/"+"instances_val2017.json", "w") as f:
                f.write(jsonString)

            jpgFiles=glob.glob(os.path.join(xml_path, '*.jpg'))
            for jpgFile in jpgFiles:
                head, tail = os.path.split(jpgFile)
                shutil.copyfile(xml_path +tail, out_path +"COCO_dataset"+"/val2017/"+ tail)
        elif train_or_test =="test":

            jsonString = json.dumps(attrDict)
            with open(out_path +"COCO_dataset"+"/annotations"+"/"+"instances_test2017.json", "w") as f:
                f.write(jsonString)

            jpgFiles=glob.glob(os.path.join(xml_path, '*.jpg'))
            for jpgFile in jpgFiles:
                head, tail = os.path.split(jpgFile)
                shutil.copyfile(xml_path +tail, out_path +"COCO_dataset"+"/test2017/"+ tail)
    elif network == "retinanet":
        attrDict["images"] = images    
        attrDict["annotations"] = annotations
        attrDict["type"] = "instances"

        if train_or_test =="train":


            if not os.path.exists(out_path +"COCO_dataset_train"):
                os.makedirs(out_path +"COCO_dataset_train")
            if not os.path.exists(out_path +"COCO_dataset_train"+"/data"):
                os.makedirs(out_path +"COCO_dataset_train"+"/data")
            jsonString = json.dumps(attrDict)
            with open(out_path +"COCO_dataset_train"+"/"+"train.json", "w") as f:
                f.write(jsonString)

            jpgFiles=glob.glob(os.path.join(xml_path, '*.jpg'))
            for jpgFile in jpgFiles:
                head, tail = os.path.split(jpgFile)
                shutil.copyfile(xml_path +tail, out_path +"COCO_dataset_train"+"/data/"+ tail)
        elif train_or_test =="test":
            if not os.path.exists(out_path +"COCO_dataset_test"):
                os.makedirs(out_path +"COCO_dataset_test")
            if not os.path.exists(out_path +"COCO_dataset_test"+"/data"):
                os.makedirs(out_path +"COCO_dataset_test"+"/data")
            jsonString = json.dumps(attrDict)
            with open(out_path +"COCO_dataset_test"+"/"+"test.json", "w") as f:
                f.write(jsonString)

            jpgFiles=glob.glob(os.path.join(xml_path, '*.jpg'))
            for jpgFile in jpgFiles:
                head, tail = os.path.split(jpgFile)
                shutil.copyfile(xml_path +tail, out_path +"COCO_dataset_test"+"/data/"+ tail)




def main():
    parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-f', '--annotation_folder', required=True, action='store', default='.', help="Folder where annotated images are.")

    parser.add_argument(
        '-o', '--outputDir',
        help='Path to the output folder where the train and test dirs should be created. '
             'Defaults to the same directory as IMAGEDIR.',
        type=str,
        default=None
    )
    parser.add_argument(
        '-tr', '--train_or_test',
        help='Path to the output folder where the train and test dirs should be created. '
             'Defaults to the same directory as IMAGEDIR.',
        type=str,
        default=None
    )
    
    args = parser.parse_args()

    if args.outputDir is None:
        args.outputDir = args.annotation_folder
    



    XML2JSON(args.annotation_folder,args.outputDir,args.train_or_text,"ssd")

if __name__ == '__main__':
    main()
