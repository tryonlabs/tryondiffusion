import numpy as np


def get_upper_garment(img, img_parse_map):
    sum_img_parse_map = np.sum(img_parse_map, axis=2)
    sum_img_parse_map[sum_img_parse_map!=339] = 0
    sum_img_parse_map[sum_img_parse_map==339] = 1
    upper_garment_segment = (sum_img_parse_map.reshape(*sum_img_parse_map.shape,1)*img).astype(np.uint8)
    return upper_garment_segment


if __name__ == "__main__":
    import os
    import sys
    from tqdm import tqdm
    parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parentdir)

    from utils import read_img, write_img

    imgs_folder = sorted(os.listdir("../../../zalando-hd-resized/train/image/"))
    img_parse_map = sorted(os.listdir("../../../zalando-hd-resized/train/image-parse-agnostic-v3.2/"))

    for img_name, img_parse_map in tqdm(zip(imgs_folder, img_parse_map)):
        img = read_img(os.path.join("../../../zalando-hd-resized/train/image/", img_name))
        img_parse_map = read_img(os.path.join("../../../zalando-hd-resized/train/image-parse-v3/", img_parse_map))

        segmented_garment = get_upper_garment(img, img_parse_map)

        write_img(segmented_garment, "../data/train/ic", img_name)
