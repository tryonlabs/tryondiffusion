import os
from collections import OrderedDict

import torch

from tryon.preprocessing.u2net import u2net_cloth_segm, u2net_human_segm


def load_cloth_segm_model(device, checkpoint_path, in_ch=3, out_ch=1):
    if not os.path.exists(checkpoint_path):
        print("Invalid path")
        return

    model = u2net_cloth_segm.U2NET(in_ch=in_ch, out_ch=out_ch)

    model_state_dict = torch.load(checkpoint_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device=device)

    print("Checkpoints loaded from path: {}".format(checkpoint_path))

    return model


def load_human_segm_model(device, model_name):
    if model_name == 'u2net':
        print("loading U2NET(173.6 MB)...")
        net = u2net_human_segm.U2NET(3, 1)
    elif model_name == 'u2netp':
        print("loading U2NEP(4.7 MB)...")
        net = u2net_human_segm.U2NETP(3, 1)
    else:
        net = None

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(os.environ.get("U2NET_SEGM_CHECKPOINT_PATH")))
        net.cuda()
    else:
        net.load_state_dict(torch.load(os.environ.get("U2NET_SEGM_CHECKPOINT_PATH"), map_location=device))
    net.eval()

    return net
