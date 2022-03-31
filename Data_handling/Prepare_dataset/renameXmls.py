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
from xxlimited import Str
import xml.etree.ElementTree as ET


def iterate_dir(source,source_office, dest, ratio, ending):
    if not source.endswith('/'):
        source = source + '/'
    ending =ending.replace('.', '')
    source = source.replace('\\', '/')
    source_office = source_office.replace('\\', '/')

    dest = dest.replace('\\', '/')
    train_dir_surg = os.path.join(dest, 'train_only_surgery')
    train_dir_all = os.path.join(dest, 'train_all')

    test_dir = os.path.join(dest, 'test')
    val_dir = os.path.join(dest, 'val')

    if os.path.exists(test_dir):
    # removing the file using the os.remove() method
        shutil.rmtree(test_dir)
    if os.path.exists(val_dir):
    # removing the file using the os.remove() method
        shutil.rmtree(val_dir)
    if os.path.exists(train_dir_all):
    # removing the file using the os.remove() method
        shutil.rmtree(train_dir_all)
    if os.path.exists(train_dir_surg):
    # removing the file using the os.remove() method
        shutil.rmtree(train_dir_surg)
    if not os.path.exists(train_dir_all):
        os.makedirs(train_dir_all)
    if not os.path.exists(train_dir_surg):
        os.makedirs(train_dir_surg)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    xml_files = glob.glob(source + '*.{}'.format(ending))
    xml_files_office = glob.glob(source_office + '*.{}'.format(ending))

    num_images = len(xml_files)
    num_test_images = math.ceil(ratio*num_images)
    num_val_images = math.ceil(ratio*num_images)

    copy_xml = True
    image_ids =[os.path.basename(f) for f in glob.glob(source + '*.JPG')]
    image_ids_office =[os.path.basename(f) for f in glob.glob(source_office + '*.JPG')]

    for i, image_id in enumerate(image_ids):
        os.rename(source + image_id, (source + image_id).split('.')[0] +".jpg" )
    for i, image_id in enumerate(image_ids_office):
        os.rename(source_office + image_id, (source_office + image_id).split('.')[0] +".jpg" )
    if ending == "xml":
        
        for i in range(num_test_images):
            idx = random.randint(0, len(xml_files)-1)
            basename =os.path.basename(xml_files[idx])
            tree = ET.parse(xml_files[idx])
            root = tree.getroot()
            filename = basename + '.jpg'
            
            copyfile(os.path.join(source, root[1].text.split(".")[0]+".jpg"),
                        os.path.join(test_dir, root[1].text.split(".")[0]+".jpg"))
            xml_filename =  basename.split(".")[0] + '.{}'.format(ending)
            copyfile(os.path.join(source, xml_filename),
                            os.path.join(test_dir, xml_filename))


            xml_files.remove(xml_files[idx])
        for xml_file in xml_files:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                basename =os.path.basename(xml_file).split('.')[0]
                filename = basename + '.jpg'
            
                copyfile(os.path.join(source, root[1].text.split(".")[0]+".jpg"),
                        os.path.join(train_dir, root[1].text.split(".")[0]+".jpg"))
                xml_filename =  basename + '.{}'.format(ending)
                copyfile(os.path.join(source, xml_filename),
                            os.path.join(train_dir, xml_filename))
  
    elif ending == "txt":    
        for i in range(num_test_images):
            idx = random.randint(0, len(xml_files)-1)
            basename =os.path.basename(xml_files[idx]).split('.')[0]
            filename = basename + '.jpg'

            copyfile(os.path.join(source, basename +".xml"),
                    os.path.join(test_dir, basename +".xml"))

            copyfile(os.path.join(source, filename),
                    os.path.join(test_dir, filename))
            if copy_xml:
                xml_filename = basename + '.{}'.format(ending)
                copyfile(os.path.join(source, xml_filename),
                        os.path.join(test_dir,xml_filename))
            xml_files.remove(xml_files[idx])
        for i in range(num_val_images):
            idx = random.randint(0, len(xml_files)-1)
            basename =os.path.basename(xml_files[idx]).split('.')[0]
            filename = basename + '.jpg'


            copyfile(os.path.join(source, basename +".xml"),
                    os.path.join(test_dir, basename +".xml"))
            copyfile(os.path.join(source, filename),
                    os.path.join(val_dir, filename))
            if copy_xml:
                xml_filename = basename + '.{}'.format(ending)
                copyfile(os.path.join(source, xml_filename),
                        os.path.join(val_dir,xml_filename))
            xml_files.remove(xml_files[idx])
        for xml_file in xml_files:
            basename =os.path.basename(xml_file).split('.')[0]
            filename = basename + '.jpg'
            copyfile(os.path.join(source, basename +".xml"),
                    os.path.join(train_dir_all, basename +".xml"))
            copyfile(os.path.join(source, basename +".xml"),
                    os.path.join(train_dir_surg, basename +".xml"))
            copyfile(os.path.join(source, filename),
                    os.path.join(train_dir_all, filename))
            copyfile(os.path.join(source, filename),
                    os.path.join(train_dir_surg, filename))
            if copy_xml:
                xml_filename =  basename + '.{}'.format(ending)
                copyfile(os.path.join(source, xml_filename),
                        os.path.join(train_dir_surg, xml_filename))
                copyfile(os.path.join(source, xml_filename),
                        os.path.join(train_dir_all, xml_filename))
        for xml_file in xml_files_office:
            basename =os.path.basename(xml_file).split('.')[0]
            filename = basename + '.jpg'
            print("in here")
            copyfile(os.path.join(source_office, filename),
                    os.path.join(train_dir_all, filename))
            copyfile(os.path.join(source_office, basename +".xml"),
                    os.path.join(train_dir_all, basename +".xml"))
            if copy_xml:
                xml_filename =  basename + '.{}'.format(ending)
                copyfile(os.path.join(source_office, xml_filename),
                        os.path.join(train_dir_all, xml_filename))


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
    parser.add_argument(
        '-o', '--outputDir',
        help='Path to the output folder where the train and test dirs should be created. '
             'Defaults to the same directory as IMAGEDIR.',
        type=str,
        default=None
    )
    parser.add_argument(
        '-r', '--ratio',
        help='The ratio of the number of test images over the total number of images. The default is 0.1.',
        default=0.1,
        type=float)
    parser.add_argument(
        '-e', '--ending',
        help='The file ending of your annotation file, e.g. txt or xml',
        type=str,
        default='txt'
    )
        
        
    args = parser.parse_args()

    if args.outputDir is None:
        args.outputDir = args.imageDir
    
    # Now we are ready to start the iteration
    iterate_dir(args.imageDir, args.outputDir, args.ratio, args.ending)


if __name__ == '__main__':
    main()