import numpy as np
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--image_folder', required=True, action='store', default='.', help="folder")
    parser.add_argument('-l', '--add_letter',action='store', default='a', help="Letter to add")
    return parser.parse_args()

def change_name(in_folder,letter):

    for file in os.listdir(in_folder):
        if(file.endswith(".jpg")):
            os.rename(in_folder + file, in_folder+letter + file )

def main():
    args = get_args()
    change_name(args.image_folder,args.add_letter)

if __name__ == '__main__':
    main()