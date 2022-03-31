import cv2
import matplotlib.pyplot as plt
import json
import numpy as np
from PIL import Image
import os.path
from os import path
import argparse
import subprocess
import os


import xml.etree.cElementTree as ET
from create_train_img import create_imgs
import list_files
import remove_duplicates



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--main_folder', required=True, action='store', default='.', help="Where everything will be created")
    parser.add_argument('-v', '--video', required=True, action='store', default=None, help="Where the video is located")
    parser.add_argument('-s', '--frame_step', required=True, action='store', type=int, default=None, help="Stepsize used when the images were generated")
    parser.add_argument('-i', '--instruments', required=True, action='store', type=str, default=None, help="Instruments of interest. Separate them wth comma, not space, ex 'diatermi,saw'")
    return parser.parse_args()


def print_info(instrument, video_start_time, video_end_time, chosen_step_size):
    print("\n")
    print("Instrument: ", instrument)
    print("video_start_time: ", video_start_time)
    print("video_end_time: ", video_end_time)
    print("chosen_step_size: ", chosen_step_size)

def generate_images(resultPath, video, frame_step, instruments_of_interest, out_folder):

    f = open(resultPath)
    data = json.load(f)

    time_between_frames = frame_step/25 # assuming video was 25 fps

    occurences = [{"diatermi":0, "forceps":0,"scalpel":0,"needle driver":0, "retractor":0, "hl tube":0, "saw":0} for x in range(len(data))]
    print("Assuming these classes are used: diatermi, forceps, scalpel, needle driver, retractor, hl tube, saw")
    print(instruments_of_interest)
    for p in range(0,len(data)):

        imgName = data[p]['filename'].split('/')[-1]
        imgPath = data[p]['filename']
    
        imgOrig = cv2.imread(data[p]['filename'])
        img = np.copy(imgOrig)
        dh, dw, depth = img.shape
        #print("imgPath",imgPath.split(".")[0])

        root = ET.Element("annotation")

        folder = ET.SubElement(root, "folder")
        folder.text = "testfolder"

        filename = ET.SubElement(root, "filename")
        filename.text = imgName

        paths = ET.SubElement(root, "path")
        paths.text = imgPath


        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = "Unknown"

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(dw)
        ET.SubElement(size, "height").text = str(dh)
        ET.SubElement(size, "depth").text = str(depth)


        segmented = ET.SubElement(root, "segmented")
        segmented.text = "0"


        for i in data[p]['objects']:
            
            name = i['name']
            occurences[p][name] += 1 # for each frame, store how many of each instrument there is
        
    
    minimum_step_size = 10 # 10=apprixmates one image every 0.5 seconds
    maximum_step_size = 250 # 250=apprixmates one image every 10 seconds
    wait_max_nbr_of_steps = 10
    steps_waited = 0
    start_idx = 0
    started = False

    """
    IDEA: Loop over every intrument of interest, one at a time. For each frame (each frame is represented by a
    dictionary that counts occurences for each instrument) we check
    if the instrument is present. If we find such a frame, save the index and continue the loop.
    When a number of images have passed with no detection, we assume it is no longer in use, so 
    we record the index where this happens. Then we want to extract images from this interval. First
    the step-size is calculated based on the lenght of the interval. Then we call the function that
    extracts the images.
    """
    for instrument in instruments_of_interest: 
        for image_idx, instrument_dict in enumerate(occurences):
            if instrument_dict[instrument] > 0:
                steps_waited = 0  
                if not started:
                    start_idx = image_idx
                    started = True
            elif started:
                steps_waited += 1
                if steps_waited == wait_max_nbr_of_steps:
                    started = False
                    steps_waited = 0
                    nbr_of_steps_with_instrument = image_idx - start_idx - wait_max_nbr_of_steps
                    begin_from_idx = max(0, start_idx-1)
                    end_at_index = image_idx - wait_max_nbr_of_steps + 1
                    ellapsed_time = (end_at_index-begin_from_idx)*time_between_frames
                    #this line could be written better
                    chosen_step_size = int(min(maximum_step_size, max(minimum_step_size, 3*ellapsed_time + minimum_step_size)))
                    #calc start and end time
                    video_start_time = int(begin_from_idx*time_between_frames/60) + (int(begin_from_idx*time_between_frames) % 60) / 100
                    video_end_time = int(end_at_index*time_between_frames/60) + (int(end_at_index*time_between_frames) % 60) / 100

                    print_info(instrument, video_start_time, video_end_time, chosen_step_size)
                    create_imgs(video, out_folder, chosen_step_size, video_start_time, video_end_time)

    
            


    f.close()

def main():
    args = get_args()
    main_folder = args.main_folder
    if not main_folder.endswith('/'):
        main_folder = main_folder + '/'
    base_images = main_folder + "base_images/"
    generated_images = main_folder + "generated_images/"
    os.makedirs(base_images, exist_ok=True)
    os.makedirs(generated_images, exist_ok=True)

    darknet = '/home/serge/exjobb/yolov4/darknet/darknet '
    data = ' /home/serge/exjobb/yolov4/4feb/data/obj.data '
    cfg = ' /home/serge/exjobb/yolov4/4feb/cfg/yolo-obj.cfg '
    weights = ' /home/serge/exjobb/yolov4/4feb/weights_backup/yolo-obj_best.weights '
    json_path = main_folder + "json_file.json"
    txt_file = main_folder + "test.txt"

    s = "Type yes to extract images from entire video. OBS: only if it hasn't already been done. Otherwise just press enter: "
    print()
    answer = input(s)
    if(answer == 'yes'):
        create_imgs(args.video, base_images, args.frame_step) # extract images over entire video and puts them in folder base_images
        print("Images from video created")
        list_files.create_file(main_folder, base_images, 'test') # generate a textfile with paths to all images
        print("textfiles pointing to images created")
        print("\n\n")
        print("Run this command in another terminal. It performs predcitions on the images and write to json-file:\n")
        print(darknet+' detector '+' test '+data+cfg+weights+' -thresh '+' 0.3 '+' -ext_output '+' -dont_show '+' -out ', json_path, ' < ', txt_file)
        #subprocess.run([darknet, 'detector', 'test', data, cfg, weights, '-thresh', '0.3', '-ext_output', '-dont_show', '-out', json_path, '<', txt_file])
        #print("predictions completed")
        print("\n\n")
        answer = ""
        while(answer != 'yes'):
            answer = input("Enter yes when the other program is finished: ")

    instruments = args.instruments.split(",")
    generate_images(json_path, args.video, args.frame_step, instruments, generated_images)
    print("Image generation completed")
    print("\nDuplicates can occur if multiple instruments are passed in")
    print("\nCHECKING FOR DUPLICATES\n")
    remove_duplicates.remove_files(generated_images)
    #print("\nDONT FORGETT TO REMOVE DUPLICATES IF THEY OCCUR. DUPLICATES HAVE THE SAME FRAME NUMBER\n")

if __name__ == '__main__':
    main()


