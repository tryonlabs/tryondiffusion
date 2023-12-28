import numpy as np


def generate_rgb_agnostic(img, img_agnostic_parse_map):
    sum_img_agnostic_parse_map = np.sum(img_agnostic_parse_map, axis=2)
    sum_img_agnostic_parse_map[sum_img_agnostic_parse_map!=0] = 1
    img_rgb_agnostic = (sum_img_agnostic_parse_map.reshape(*sum_img_agnostic_parse_map.shape, 1)*img).astype(np.uint8)
    return img_rgb_agnostic


if __name__ == "__main__":
    import os
    import sys
    from tqdm import tqdm

    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parentdir)

    from utils import read_img, write_img

    imgs_folder = sorted(os.listdir("../../../zalando-hd-resized/train/image/"))
    imgs_agnostic_parse_map_folder = sorted(os.listdir("../../../zalando-hd-resized/train/image-parse-agnostic-v3.2/"))

    for img_name, img_agnostic_parse_map in tqdm(zip(imgs_folder, imgs_agnostic_parse_map_folder)):
        img = read_img(os.path.join("../../../zalando-hd-resized/train/image/", img_name))
        img_agnostic_parse_map = read_img(os.path.join("../../../zalando-hd-resized/train/image-parse-agnostic-v3.2/",
                                                       img_agnostic_parse_map))

        rgb_agnostic = generate_rgb_agnostic(img, img_agnostic_parse_map)

        write_img(rgb_agnostic, "../data/train/ia", img_name)


