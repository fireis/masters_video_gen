from PIL import Image, ImageDraw
import argparse
import numpy as np
import glob
import face_alignment
from skimage import io
import os

# def arg_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "-i", "--input", type=str, required=True, help="path to input image folder"
#     )
#     parser.add_argument(
#         "-s",
#         "--size",
#         type=tuple,
#         required=False,
#         help="path to predictor .dat file",
#         default=(256, 256),
#     )
#     args = parser.parse_args()
#     return args


def read_img(img_path):
    return Image.open(img_path)


def read_keypoints(kp_path):
    return np.loadtxt(kp_path, delimiter=",")


def make_power_2(n, base=32.0):
    return int(round(n / base) * base)


def get_crop_coords(keypoints, size):
    min_y, max_y = keypoints[:, 1].min(), keypoints[:, 1].max()
    min_x, max_x = keypoints[:, 0].min(), keypoints[:, 0].max()
    offset = (max_x - min_x) // 2
    min_y = max(0, min_y - offset * 2)
    min_x = max(0, min_x - offset)
    max_x = min(size[0], max_x + offset)
    max_y = min(size[1], max_y + offset)
    return (int(min_y), int(max_y), int(min_x), int(max_x))


def resize_img(img, keypoints):

    # hardcoded to comply with generated models, 256 and 320 should be variables
    new_w = 256
    new_h = 256

    min_y, max_y, min_x, max_x = get_crop_coords(keypoints, img.size)
    img = img.crop((min_x, min_y, max_x, max_y))
    new_w, new_h = make_power_2(new_w), make_power_2(new_h)
    method = Image.BICUBIC
    img = img.resize((new_w, new_h), method)
    return img


def find_kp(path, method="fa", save_img=True, resize=True):
    fpath = path + "/*.jpg"
    frames_path = glob.glob(fpath)
    if method == "dlib":
        save_path_txt = path + "dlib_keypoints/"
        save_path_img = path + "dlib_img/"

        if not os.path.exists(save_path_txt):
            os.makedirs(save_path_txt)
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)

        predictor = dlib.shape_predictor(
            os.path.abspath('predictor/shape_predictor_68_face_landmarks.dat'))
        detector = dlib.get_frontal_face_detector()

    elif method=="fa":
        save_path_txt = path + "fa_keypoints/"
        save_path_img = path + "fa_img/"

        if not os.path.exists(save_path_txt):
            os.makedirs(save_path_txt)
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)

        detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    save_path_img_orig = path + "orig_img/"
    if not os.path.exists(save_path_img_orig):
        os.makedirs(save_path_img_orig)
    if not os.path.exists(save_path_img_orig):
        os.makedirs(save_path_img)

    for frames in frames_path:
        # print(frames)
        img = io.imread(frames)
        img_d = Image.new("RGB", (np.shape(img)[1], np.shape(img)[0]) )
        img_orig = Image.open(frames)

        # img = resize_img(img, keypoints)

        draw = ImageDraw.Draw(img_d)

        if method == "dlib":
            dets = detector(img, 1)
            if len(dets) > 0:
                shape = predictor(img, dets[0])

        elif method == "fa":
            shape = detector.get_landmarks(img)
            # print(shape)

        points = np.empty([68, 2], dtype=int)
        for b in range(68):
            # print(shape[0][b][0])
            points[b, 0] = shape[0][b][0]
            points[b, 1] = shape[0][b][1]
            if save_img:
                draw.ellipse((points[b, 0], points[b, 1], points[b, 0] + 10,points[b, 1] + 10), fill='white',
                             outline='white')

        save_name_txt = os.path.join(save_path_txt, frames[-9:-4] + '.txt')
        save_name_img =  os.path.join(save_path_img, frames[-9:])
        save_name_img_orig = os.path.join(save_path_img_orig, frames[-9:])
        # print(save_name_img)

        if resize:
            method = Image.BICUBIC
            img_d = img_d.resize((256,256), method)
            img_orig = img_orig.resize((256,256), method)

        np.savetxt(save_name_txt, points, fmt='%05d', delimiter=',')
        img_d.save(save_name_img)
        img_orig.save(save_name_img_orig)



if __name__ == "__main__":
    # This code uses sections from base prep of vid2vid
    # args = arg_parser()

    # for img_path in glob.glob(args.input + "*/*/*.jpg"):
    #     print("current file:{0}".format(img_path))
    #     keypoints_path = img_path.replace("img", "kp").replace(
    #         ".jpg", ".txt"
    #     )
    #     img = resize_img(img, keypoints)
    #
    #     img = read_img(img_path)
    #     keypoints = read_keypoints(keypoints_path)
    #
    #
    #
    #     img.save(img_path)
    find_kp("D:/masters/train/00002/")
    find_kp("D:/masters/train/00003/")
    find_kp("D:/masters/train/00004/")
    find_kp("D:/masters/train/00005/")
    find_kp("D:/masters/train/00006/")
    find_kp("D:/masters/train/00007/")
    find_kp("D:/masters/train/00008/")
