import os
from pathlib import Path

import cv2
from PIL import Image
from torchvision import transforms


class NormalizeImage(object):
    """Normalize given tensor into given mean and standard dev

    Args:
        mean (float): Desired mean to substract from tensors
        std (float): Desired std to divide from tensors
    """

    def __init__(self, mean, std):
        assert isinstance(mean, (float))
        if isinstance(mean, float):
            self.mean = mean

        if isinstance(std, float):
            self.std = std

        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self.normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)

        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)

        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)

        else:
            assert "Please set proper channels! Normalization implemented only for 1, 3 and 18"


def naive_cutout(img, mask):
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask.resize(img.size, Image.LANCZOS))
    return cutout


def resize_by_bigger_index(crop):
    # function resizes and keeps the aspect ratio same
    crop_shape = crop.shape  # hxwxc
    if crop_shape[0] / crop_shape[1] <= 1.33:
        resized_crop = image_resize(crop, width=768)
    else:
        resized_crop = image_resize(crop, height=1024)
    return resized_crop


def image_resize(image, width=None, height=None):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim)

    return resized


def convert_to_jpg(image_path, output_dir, size=None):
    """
    Convert image to jpg format
    :param image_path: image path
    :param output_dir: output directory
    :param size: desired size of the image (w, h)
    """
    img = cv2.imread(image_path)
    if size is not None:
        img = image_resize(img, width=size[0], height=size[1])

    filename = Path(image_path).name
    cv2.imwrite(os.path.join(output_dir, filename.split(".")[0] + ".jpg"), img)
