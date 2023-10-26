import os
import cv2
import json

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

from utils import load_json, read_img, GaussianSmoothing


class UNet128DatasetTrain(Dataset):
    """ This class is to be used while training, where all the conditional inputs and ground
     truth is pre-saved and are pre-processed."""

    def __init__(self,
                 ip_dir="../train/data/ip",
                 jp_dir="../train/data/jp",
                 jg_dir="../train/data/jg",
                 ia_dir="../train/data/ia",
                 ic_dir="../train/data/ic"
                 ):
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

    def __len__(self):
        return len(self.ip_list)

    def __getitem__(self, item):
        ip = read_img(self.ip_paths[item])
        jp = load_json(self.jp_paths[item])
        jg = load_json(self.jg_paths[item])
        ia = read_img(self.ia_paths[item])
        ic = read_img(self.ic_paths[item])

        # ToDo: move this to training later and pass sigma to 'forward' of network

        # As suggested in:
        # https://jmlr.csail.mit.edu/papers/volume23/21-0635/21-0635.pdf Section 4.4
        # sigma = torch.FloatTensor(1).uniform_(0.4, 0.6)

        # smoothing2d = GaussianSmoothing(channels=3,
        #                                 kernel_size=3,
        #                                 sigma=sigma,
        #                                 conv_dim=2)
        #
        # # using same kernel for both images
        # ia = F.pad(ia, (1, 1, 1, 1), mode='reflect')
        # ia = smoothing2d(ia)
        #
        # ip = F.pad(ip, (1, 1, 1, 1), mode='reflect')
        # ip = smoothing2d(ip)

        return ip, jp, jg, ia, ic
