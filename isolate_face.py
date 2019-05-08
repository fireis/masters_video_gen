import os
import glob
from skimage import io
import numpy as np
import dlib
import sys
import argparse
import cv2


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
    parser.add_argument(
        "-m",
        "--mode",
        default="kp")
    args = parser.parse_args()
    return args


def read_img(file_path):
    img = io.imread(file_path)
    return img


def find_keypoints(img, detector, predictor):
    dets = detector(img, 1)
    if len(dets) > 0:
        shape = predictor(img, dets[0])
        points = np.empty([68, 2], dtype=int)
        for b in range(68):
            points[b, 0] = shape.part(b).x
            points[b, 1] = shape.part(b).y
        return points
    else:
        return [0]


def proc_img(img_path, detector, predictor):
    # mask everything outside of left eyebrow, right eyebrow, face

    current_img = read_img(img_path)
    keypoints = find_keypoints(current_img, detector, predictor)

    face = keypoints[0:17]
    #eyebrows = keypoints[17:27]
    #eyebrows = eyebrows[::-1]  # Invert coords due to DLIB ordering
    #face_eyebrows = np.insert(face, 0, eyebrows, axis=0)

    # Get upper points from mirrorring (adapted from Vid2Vid)
    pts = keypoints[:17, :].astype(np.int32)
    baseline_y = (pts[0, 1] + pts[-1, 1]) / 2
    upper_pts = pts[1:-1, :].copy()
    upper_pts[:, 1] = baseline_y + (baseline_y - upper_pts[:, 1]) * 2 // 3
    head = np.insert(face, 0, upper_pts[::-1], axis=0)

    # Mask the face
    mask = np.array([head])
    image2 = np.zeros(current_img.shape)
    cv2.fillPoly(image2, [mask], 255)
    maskimage2 = cv2.inRange(image2, 1, 255)

    # Create new image masking only the face
    out = cv2.bitwise_and(current_img, current_img, mask=maskimage2)
    # TODO: Improve this string handling. This works but should be better
    #out_path = img_path.replace(img_path[-4:], "_masked" + img_path[-4:])
    out_path = img_path

    cv2.imwrite(out_path, out[:, :, ::-1])
    np.savetxt(out_path.replace(out_path[-3:],"txt"), keypoints, delimiter=",", fmt="%05d")

def get_keypoints(img_path, detector, predictor):
    current_img = read_img(img_path)    
    keypoints = find_keypoints(current_img, detector, predictor)
    txt_path = os.path.dirname(img_path) 
    vid_number = txt_path[-5:] + "/"
    txt_path = os.path.dirname(os.path.dirname(txt_path)) + "/train_keypoints/" + vid_number
    txt_path = txt_path + img_path.replace(img_path[-3:],"txt")[-9:]
    print("Saving: {}".format(txt_path))
    np.savetxt(txt_path, keypoints, delimiter=",", fmt="%05d")

if __name__ == "__main__":
    args = arg_parser()
    print("Folder to operate: {}".format(args.input))

    predictor = dlib.shape_predictor(os.path.abspath(args.predictor))
    detector = dlib.get_frontal_face_detector()
    imgs = [f for f in glob.glob(args.input + "**/*.jpg", recursive=True)]
    total_imgs = len(imgs)
    proc = 0
    if args.mode == "kp":
        for img in imgs:
            get_keypoints(img, detector, predictor)        
            print("Processed: {}% of the files".format(proc/total_imgs*100))
            proc += 1
    
    # TODO: read entire folder, not single files
    
