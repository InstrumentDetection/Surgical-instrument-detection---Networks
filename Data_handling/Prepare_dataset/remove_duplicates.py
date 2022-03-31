import glob
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--data_folder', required=True, action='store', default='.', help="folder")
    
    return parser.parse_args()

def remove_files(data_folder):

    if not data_folder.endswith('/'):
        data_folder = data_folder + '/'

    image_ids =[os.path.basename(f) for f in glob.glob(data_folder + '*.jpg')]
    image_ids = sorted(image_ids, key=lambda x: int(x[:len(x)-9]))
    frame_idx = -1
    #print(image_ids)
    for img_name in image_ids:
        current_frame_idx = int(img_name[:len(img_name)-9])
        if current_frame_idx == frame_idx:
            print("Removing file: ", data_folder + img_name)
            os.remove(data_folder + img_name)
        else:
            frame_idx = current_frame_idx
    

def main():
    args = get_args()
    remove_files(args.data_folder)

if __name__ == '__main__':
    main()

