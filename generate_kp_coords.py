import os
import glob
from skimage import io
import numpy as np
import dlib
import sys
import argparse
def parsers():

    parser = argparse.ArgumentParser(description='Obtain kp coordinates from frames.')
    parser.add_argument("-i", "--input", required=True,
        help="path to input image folder")
    parser.add_argument('--generate_frames', action='generate_frames',
                        help='generate frames from .mp4 on folder')
    parser.add_argument('--train_mode', action='create_train_folders',
                        help='')
    parser.add_argument('--test_mode', action='create_test_folders',
                        help='')





def generate_frames():
    #TODO: generate frames
    pass

def generate_keypoints():
    #TODO: generate keypoints
    pass

def create_folders():
    #TODO: create img and kp folders, with option to train or test
    pass

def findkp(path):
    fpath = path + "/*.jpg"
    frames_path = glob.glob(fpath)
    save_path = path + "_keypoints/"
    predictor = dlib.shape_predictor(
        os.path.abspath('predictor/shape_predictor_68_face_landmarks.dat'))
    detector = dlib.get_frontal_face_detector()

    for frames in frames_path:
        print(frames)

        #save_path = ("/Users/fireis/Documents/masters/she/Em20_Fala1_Neutra_SheilaFaermann_keypoints")
        print(save_path)
            

        img = io.imread(frames )
        dets = detector(img, 1)
        if len(dets) > 0:
            shape = predictor(img, dets[0])
            points = np.empty([68, 2], dtype=int)
            for b in range(68):
                points[b,0] = shape.part(b).x
                points[b,1] = shape.part(b).y

            save_name = os.path.join(save_path, frames[:-4] + '.txt')
            print(save_name)
            np.savetxt(save_name, points, fmt='%05d', delimiter=',')

def findksp():

    frames_path = glob.glob("/Users/fireis/Documents/masters/she/Em20_Fala1_Neutra_SheilaFaermann")

    predictor = dlib.shape_predictor(
        os.path.abspath('predictor/shape_predictor_68_face_landmarks.dat'))
    detector = dlib.get_frontal_face_detector()

    for frames in frames_path:
        save_path = (os.path.abspath(frames[:-1]+'_keypoints'))
        print(save_path)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        for frame in os.listdir(os.path.abspath(frames)):
            print(frames + "/" + frame)
            
            if(os.path.isfile(frames + "/" + frame) ):

                img = io.imread(frames + "/" + frame)
                dets = detector(img, 1)
                if len(dets) > 0:
                    shape = predictor(img, dets[0])
                    points = np.empty([68, 2], dtype=int)
                    for b in range(68):
                        points[b,0] = shape.part(b).x
                        points[b,1] = shape.part(b).y

                    save_name = os.path.join(save_path, frame[:-4] + '.txt')
                    print(save_name)
                    np.savetxt(save_name, points, fmt='%05d', delimiter=',')
def rename():
    import os, re
    from os.path import isfile, join, isdir
    paths = ["/Users/fireis/Documents/masters/she/Em20_Fala1_Neutra_SheilaFaermann"]

    for path in paths:
        files = os.listdir(path)
        print(files)
        for file in files:
            print(file)
            fpath = path + "/" + file
            n = int(re.findall("[0-9]+[(.jpg)]", file)[0].replace(".", ""))
            os.rename(fpath, path + "/{:05d}.jpg".format(n))
            print("{:05d}".format(n))

paths = ["/Users/fireis/Documents/masters/test_03/Em20_Fala1_CarolinaHolly"]
for path in paths:
    findkp(path)
