from dotenv import load_dotenv

load_dotenv()

import os
import argparse

from tryon.preprocessing import segment_garment, extract_garment

if __name__ == '__main__':
    argp = argparse.ArgumentParser(description="Tryon preprocessing")
    argp.add_argument('-d',
                      '--dataset',
                      type=str, default="data", help='Path of the dataset dir')
    argp.add_argument('-c',
                      '--cls',
                      type=str, default="upper", help='Ex. upper, lower, all')
    args = argp.parse_args()

    segment_garment(inputs_dir=os.path.join(args.dataset, "original_cloth"),
                    outputs_dir=os.path.join(args.dataset, "garment_segmented"), cls=args.cls)

    extract_garment(inputs_dir=os.path.join(args.dataset, "original_cloth"),
                    outputs_dir=os.path.join(args.dataset, "cloth"), cls=args.cls, resize_to_width=400)
