import numpy as np


def generate_rgb_agnostic(img, img_agnostic_parse_map):
    sum_img_agnostic_parse_map = np.sum(img_agnostic_parse_map, axis=2)
    sum_img_agnostic_parse_map[sum_img_agnostic_parse_map!=0] = 1
    img_rgb_agnostic = (sum_img_agnostic_parse_map.reshape(*sum_img_agnostic_parse_map.shape, 1)*img).astype(np.uint8)
    return img_rgb_agnostic


if __name__ == "__main__":
    import os
    import sys

    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parentdir)

    from utils.utils import read_img, write_img

    img = read_img("../../../zalando-hd-resized/train/image/00000_00.jpg")
    img_agnostic_parse_map = read_img("../../../zalando-hd-resized/train/image-parse-agnostic-v3.2/00000_00.png")

    rgb_agnostic = generate_rgb_agnostic(img, img_agnostic_parse_map)

    write_img(rgb_agnostic, "../data/test_flow/train/ia", "00000_00.jpg")


