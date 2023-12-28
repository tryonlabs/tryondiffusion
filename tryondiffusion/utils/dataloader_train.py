import os

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torchvision import transforms as T

from .utils import load_pose_embed, read_img


class ToPaddedTensorImages:

    def __call__(self, image):
        """Padding image so that aspect ratio is maintained.
        And converting numpy arrays to tensors."""
        # cv2 image: H x W x C
        # torch image: C X H X W

        img = image.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        if img.shape[1] > img.shape[2]:
            pad_size = (img.shape[1] - img.shape[2]) // 2
            padding = (pad_size, pad_size, 0, 0)
        elif img.shape[2] > img.shape[1]:
            pad_size = (img.shape[2] - img.shape[1]) // 2
            padding = (0, 0, pad_size, pad_size)
        else:
            padding = (0, 0, 0, 0)

        img = F.pad(img, padding, "constant", 0)

        return img


class ToTensorEmbed:

    def __call__(self, pose_embed):

        return torch.from_numpy(pose_embed)


class UNetDataset(Dataset):
    """ This class is to be used while training, where all the conditional inputs and ground
     truth is pre-saved and are pre-processed."""

    def __init__(self, ip_dir, jp_dir, jg_dir, ia_dir, ic_dir, unet_size):
        """
        Get all the inputs from ../data directory in the main project directory
        :param ip_dir: Image of target person with source clothing on. Later
        to be used to generate zt and to be used as ground truth for training.
        :param jp_dir: person pose embeddings from ip
        :param jg_dir: garment pose embeddings from 'ig', ig is the source garment image
        :param ia_dir: clothing agnostic rgb from ip
        :param ic_dir: segmented garment from ig
        """
        self.ip_list = os.listdir(ip_dir)
        self.ip_paths = [os.path.join(ip_dir, i) for i in self.ip_list]

        self.jp_list = os.listdir(jp_dir)
        self.jp_paths = [os.path.join(jp_dir, i) for i in self.jp_list]

        self.jg_list = os.listdir(jg_dir)
        self.jg_paths = [os.path.join(jg_dir, i) for i in self.jg_list]

        self.ia_list = os.listdir(ia_dir)
        self.ia_paths = [os.path.join(ia_dir, i) for i in self.ia_list]

        self.ic_list = os.listdir(ic_dir)
        self.ic_paths = [os.path.join(ic_dir, i) for i in self.ic_list]

        self.transforms_imgs = T.Compose([
            ToPaddedTensorImages(),
            T.Resize(unet_size),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.ip_list)

    def __getitem__(self, item):
        ip = read_img(self.ip_paths[item])
        jp = load_pose_embed(self.jp_paths[item])
        jg = load_pose_embed(self.jg_paths[item])
        ia = read_img(self.ia_paths[item])
        ic = read_img(self.ic_paths[item])

        ip = self.transforms_imgs(ip)
        ia = self.transforms_imgs(ia)
        ic = self.transforms_imgs(ic)

        return ip, jp, jg, ia, ic
