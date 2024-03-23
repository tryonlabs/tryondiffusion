from dotenv import load_dotenv

load_dotenv()

import time
import os
import argparse

from tryon.preprocessing import segment_human, segment_garment, extract_garment

if __name__ == '__main__':
    argp = argparse.ArgumentParser(description="Tryon preprocessing")
    argp.add_argument('-d',
                      '--dataset',
                      type=str, default="data", help='Path of the dataset dir')
    argp.add_argument('-a',
                      '--action',
                      type=str, default="", help='Ex. segment_garment, extract_garment, segment_human')
    argp.add_argument('-c',
                      '--cls',
                      type=str, default="upper", help='Ex. upper, lower, all')
    args = argp.parse_args()

    if args.action == "segment_garment":
        # 1. segment garment
        print('Start time:', int(time.time()))
        segment_garment(inputs_dir=os.path.join(args.dataset, "original_cloth"),
                        outputs_dir=os.path.join(args.dataset, "garment_segmented"), cls=args.cls)
        print("End time:", int(time.time()))

    elif args.action == "extract_garment":
        # 2. extract garment
        print('Start time:', int(time.time()))
        extract_garment(inputs_dir=os.path.join(args.dataset, "original_cloth"),
                        outputs_dir=os.path.join(args.dataset, "cloth"), cls=args.cls, resize_to_width=400)
        print("End time:", int(time.time()))

    elif args.action == "segment_human":
        # 2. segment human
        print('Start time:', int(time.time()))
        image_path = os.path.join(args.dataset, "original_human", "model.jpg")
        output_dir = os.path.join(args.dataset, "human_segmented")
        segment_human(image_path=image_path, output_dir=output_dir)
        print("End time:", int(time.time()))
