import os
from pathlib import Path
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

SAM2_DIR = os.path.join(str(Path.home()), 'sam2')

checkpoint = os.path.join(SAM2_DIR, "checkpoints/sam2.1_hiera_large.pt")
model_cfg = os.path.join(SAM2_DIR, "configs/sam2.1/sam2.1_hiera_l.yaml")
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image("image.png")
    masks, _, _ = predictor.predict("cat")
