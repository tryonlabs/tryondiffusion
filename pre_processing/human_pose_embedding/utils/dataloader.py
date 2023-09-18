import os
import json

import torch
from torch.utils.data import DataLoader, Dataset


def normalize(val, lower, upper):
    return (val - lower) / (upper - lower)


def normalize_lst(lst):
    """
    Normalizing Keypoints between 0 to 1, Image size is 1024x786.
    Will be adding padding to both sides of images of 119 pixels to make image dims 1024x1024,
    to keep them in sync with paper, so networks can input 128x128 and output 1024x1024.
    Therefore Normalizing 'y' with lower bound 0 and upper bound 1024.
    Adding 119 to 'x' and normalizing it with lowe bound 0 and upper bound 1024.

    :param lst: list of coordinates; sample: [x1, y1, x2, y2...] (total keypoint are
    25, so length list is 50)
    :return: normalized list for x and y coordinate.
    """
    # lst = [normalize(lst[i], 0, 1024) if i % 2 else normalize(lst[i]+119, 0, 1024) for i in range(len(lst))]
    normalized_list = list()
    for i in range(len(lst)):
        if i % 2:  # y
            val = normalize(lst[i], 0, 1024)
        else:  # x
            # condition so that non-existent keypoint remain zero, if x and y both are 0 keypoint non-existent
            if lst[i] == 0 and lst[i + 1] == 0:
                val = lst[i]
            else:
                val = normalize(lst[i] + 119, 0, 1024)
        normalized_list.append(val)
    return normalized_list


class KeypointDataset(Dataset):

    def __init__(self, json_dir):
        self.json_lst = os.listdir(json_dir)
        self.json_paths = [os.path.join(json_dir, i) for i in self.json_lst]

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, item):
        json_path = self.json_paths[item]
        with open(json_path, "r") as f:
            lst = json.load(f)
        lst = normalize_lst(lst)
        return torch.tensor(lst)  # json_path


if __name__ == "__main__":
    json_di = "../data/test"
    test = KeypointDataset(json_di)
    dataloader = DataLoader(test, batch_size=4)
    for i in dataloader:
        print(i)
        break
