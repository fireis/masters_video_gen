import numpy as np
import argparse
import glob
import cv2
from scipy.optimize import curve_fit
from PIL import Image
from skimage import feature
from skimage import io


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="path to input image folder"
    )
    parser.add_argument(
        "-p",
        "--predictor",
        type=str,
        required=False,
        help="path to predictor .dat file",
        default="predictor/shape_predictor_68_face_landmarks.dat",
    )
    args = parser.parse_args()
    return args

def read_keypoints(txt_path):
    keypoints = np.loadtxt(txt_path, delimiter=",", dtype="int")
    return keypoints

def draw_keypoints(keypoints, image):
    
    for (x, y) in keypoints:
	    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

if __name__ == "__main__":
    args = arg_parser()
    print("Folder to operate: {}".format(args.input))
    keypoint_files = glob.glob(args.input+"*.txt")
  

    for keypoint_file in keypoint_files:
        out_path = keypoint_file.replace(".txt", "_drawn.jpg")
        image_path = keypoint_file.replace(".txt", ".jpg")
        img = io.imread(image_path)
        keypoints = read_keypoints(keypoint_file)
        draw_keypoints(keypoints, img)   
        cv2.imwrite(out_path, img[:, :, ::-1])





