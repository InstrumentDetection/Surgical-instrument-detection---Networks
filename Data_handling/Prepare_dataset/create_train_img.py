import numpy as np
import cv2
import argparse
import os
import random
import string
from moviepy.editor import VideoFileClip
import math
#STEP = 1000
NAME_PREFIX = 'a'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', required=True, action='store', default='.', help="video")
    parser.add_argument('-f', '--image_folder', required=True, action='store', default='.', help="folder")
    parser.add_argument('-s', '--steps', action='store', default=1000, type=int,help="folder")
    parser.add_argument('-start', '--start', action='store', default=0.0, type=float,help="Minute to start")
    parser.add_argument('-end', '--end', action='store', default=10000.0,type=float, help="Minute to end")

    return parser.parse_args()

def get_frame(frame_no, cap):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    return cap.read()

def create_imgs(video, out_folder,STEP, start=0.0, end=10000000.0):
    fracS, wholeS = math.modf(start)
    fracE, wholeE = math.modf(end)

    minStart = wholeS + (fracS*100)/60
    minEnd = wholeE + (fracE*100)/60

    clip = VideoFileClip(video)
    minVideo = clip.duration/60
    folder =out_folder
    cap = cv2.VideoCapture(video)
    tot  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if minVideo < end:
        end= minVideo
    if not os.path.exists(folder):
        os.makedirs(folder)
    randString = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    for k in range(round((minStart/minVideo)*tot), round((minEnd/minVideo)*tot), STEP):
        _, frame = get_frame(k, cap)
        lname = folder + '/{}{}.jpg'.format(k, randString)
        try:
            if frame.shape[1]<frame.shape[0]:
                frame = frame.flip(frame,1)
            cv2.imshow('Testing', frame)
            key = cv2.waitKey(1)
            cv2.imwrite(lname, frame)
            #print(lname)
        except:
            print("Error at the end of video, but it alright, probably")
            break
        

    cap.release()
    cv2.destroyAllWindows()
def main():
    args = get_args()
    create_imgs(args.video, args.image_folder,args.steps,args.start, args.end)

if __name__ == '__main__':
    main()
