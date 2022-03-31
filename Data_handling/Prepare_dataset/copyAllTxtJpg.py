import shutil 
import os 
import argparse
#Copy all .txt and -png files to specified directory

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--from_folder', required=True, action='store', default='.', help="folder")
    parser.add_argument('-t', '--to_folder', required=True, action='store', default='.', help="The folder to move all files.")
    
    return parser.parse_args()

def copy_files(from_folder, to_folder):




    # Check whether the specified path exists or not
    isExist = os.path.exists(to_folder)
    if not isExist:
    # Create a new directory because it does not exist 
        os.makedirs(to_folder)
        print("The new directory is created!")


    for file in os.listdir(from_folder):
        if file.endswith(".jpg"):
            print("here",from_folder + file)
            shutil.copy(from_folder + file,to_folder)
        if file.endswith(".txt"):
            shutil.copy(from_folder + file,to_folder)

def main():
    args = get_args()
    copy_files(args.from_folder, args.to_folder)

if __name__ == '__main__':
    main()

