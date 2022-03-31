import numpy as np
import cv2
import argparse
import os
import random
import string
import moviepy.editor as mp
import math
#STEP = 1000
NAME_PREFIX = 'a'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', required=True, action='store', default='.', help="video")
    parser.add_argument('-o', '--out_folder', required=True, action='store', default='.', help="folder")
    parser.add_argument('-n', '--clip_name', required=True, action='store', default='.', help="clip name")

    return parser.parse_args()


def a(hour, min, sec):
    return hour*3600 + min*60 + sec

def create_clip(video, out_folder, clip_name):
    if not out_folder.endswith('/'):
        out_folder = out_folder + '/'

    video = mp.VideoFileClip(video)

    # delete video fragment from 00:30 to 01:00
    segments = [(1738,4391)]
    #9.22
    tracking_part1 = [(9*60+22, 9*60+52), (10*60, 10*60+10), (10*60+47, 11*60+5), (28*60+10, 28*60+20), (35*60, 35*60+20), 
                        (36*60, 36*60+29),
                        (42*60+37, 43*60+17), 
                        (45*60+50, 46*60+20), (52*60+45, 53*60+17), 
                        (73*60+13, 73*60+50),
                        (87*60+7, 87*60+18), (92*60+30, 92*60+40), (156*60+10, 156*60+38), (156*60+48, 157*60+32)]

    large_heart = [(a(0,48,30), a(0,49,29)), (a(0,50,22), a(0,50,49)), (a(0,53,20), a(0,53,30)), (a(0,54,30), a(0,54,46)),
                    (a(0,56,54), a(0,57,29)), (a(0,59,33),a(1,0,0)), (a(1,2,10),a(1,2,23)), (a(1,8,25),a(1,8,50)),
                    (a(1,15,43),a(1,16,8)), (a(1,18,15), a(1,18,25)), (a(1,30,18),a(1,30,44)), (a(1,32,29),a(1,32,40)),
                    (a(1,33,20), a(1,33,31)), (a(1,35,40),a(1,35,48)), (a(1,36,1),a(1,36,19)), (a(1,39,26),a(1,39,39)),
                    (a(1,48,21),a(1,48,45)), (a(1,54,23), a(1,54,42)), (a(2,55,30), a(2,55,46))]
    segments = large_heart

    clips = []  # list of all video fragments
    for start_seconds, end_seconds in segments:
        # crop a video clip and add it to list
        c = video.subclip(start_seconds, end_seconds)
        clips.append(c)

    final_clip = mp.concatenate_videoclips(clips)
    print("filepath to clip: ", out_folder + clip_name)
    final_clip.write_videofile(out_folder + clip_name)
    final_clip.close()

def main():
    args = get_args()
    create_clip(args.video, args.out_folder, args.clip_name)

if __name__ == '__main__':
    main()



for_glen_clip1_20210603 = [(0, 20),
                (1*60 +50, 2*60),
                (7*60+25, 7*60+30),
                (9*60+6, 9*60+10),
                (37*60+32, 37*60+37),
                (76*60+23, 76*60+47),
                (85*60+7, 85*60+13),
                (94*60+43,95*60+11),
                (104*60+33, 104*60+39)
                ]