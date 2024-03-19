import os
from collections import OrderedDict

import torch

from .u2net import U2NET


def create_model(device, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("Invalid path")
        return

    model = U2NET(in_ch=3, out_ch=4)

    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device=device)

    print("Checkpoints loaded from path: {}".format(checkpoint_path))

    return model
