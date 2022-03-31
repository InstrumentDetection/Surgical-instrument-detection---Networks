# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
from os import listdir
from os.path import join
import argparse
import glob 

    


def clear_all(img_dir):
    label_dir = img_dir
    if not label_dir.endswith('/'):
        label_dir = label_dir + '/'
    xml_ids =[os.path.basename(f) for f in glob.glob(label_dir + '*.xml')]


    # Absolute path of a file
    #old_name = r"E:\demos\files\reports\details.txt"
    #new_name = r"E:\demos\files\reports\new_details.txt"

    # Renaming the file
    #os.rename(old_name, new_name)
    for files in xml_ids:
        tree = ET.parse(label_dir+files)
        root = tree.getroot()
        fileName= label_dir+root[1].text.split(".")[0]+".jpg"
        if not root[1].text.split(".")[0] == files.split(".")[0]:
            os.rename(label_dir + files, label_dir + root[1].text.split(".")[0]+".xml" )
            print("renamed:",files)


def main():
    parser = argparse.ArgumentParser(description="Remove xml file if they dont have a related jpg file in the same folder.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-f', '--xml_folder', required=True, action='store', default='.', help="Folder where xml files are.")
    args = parser.parse_args()
    clear_all(args.xml_folder)

if __name__ == '__main__':
    main()
