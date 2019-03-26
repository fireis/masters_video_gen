from PIL import Image
import argparse
import numpy as np
import glob


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="path to input image folder"
    )
    parser.add_argument(
        "-s",
        "--size",
        type=tuple,
        required=False,
        help="path to predictor .dat file",
        default=(256, 256),
    )
    args = parser.parse_args()
    return args


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
    new_w = int(round(256 / 4)) * 4
    new_h = int(round(320 / 4)) * 4

    min_y, max_y, min_x, max_x = get_crop_coords(keypoints, img.size)
    img = img.crop((min_x, min_y, max_x, max_y))
    new_w, new_h = make_power_2(new_w), make_power_2(new_h)
    method = Image.BICUBIC
    img = img.resize((new_w, new_h), method)
    return img


if __name__ == "__main__":
    # This code uses sections from base prep of vid2vid
    args = arg_parser()

    for img_path in glob.glob(args.input + "*/*/*.jpg"):
        print("current file:{0}".format(img_path))
        keypoints_path = img_path.replace("train_img", "train_keypoints").replace(
            ".jpg", ".txt"
        )
        img = read_img(img_path)
        keypoints = read_keypoints(keypoints_path)
        resized_img = resize_img(img, keypoints)
        resized_img.save(img_path)
