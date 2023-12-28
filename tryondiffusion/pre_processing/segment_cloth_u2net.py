import argparse
import os

import torch

from u2net_cloth_seg import segment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='U Square Net based cloth segmentation')
    parser.add_argument('--inputs_dir', type=str,
                        help='Input images directory path')
    parser.add_argument('--outputs_dir', type=str,
                        help='Output images directory path')
    parser.add_argument('--checkpoint_path', type=str,
                        help='Specify the U2Net checkpoint file path')
    args = parser.parse_args()

    if args.inputs_dir is None or args.outputs_dir is None or args.checkpoint_path is None:
        print("Missing parameters: inputs_dir, outputs_dir, or checkpoint_path")
        exit()

    inputs_dir = args.inputs_dir
    outputs_dir = args.outputs_dir
    checkpoint_path = args.checkpoint_path
    os.makedirs(outputs_dir, exist_ok=True)

    os.makedirs(os.path.join(outputs_dir, "alpha_masks"), exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "final_seg"), exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "original"), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    segment(device, inputs_dir, outputs_dir, checkpoint_path)
