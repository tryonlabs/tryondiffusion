import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from .u2net import load_cloth_segm_model
from .utils import NormalizeImage, naive_cutout, resize_by_bigger_index, image_resize


def segment_garment(inputs_dir, outputs_dir, cls="all"):
    os.makedirs(outputs_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform_fn = transforms.Compose(
        [transforms.ToTensor(),
         NormalizeImage(0.5, 0.5)]
    )

    net = load_cloth_segm_model(device, os.environ.get("U2NET_CLOTH_SEGM_CHECKPOINT_PATH"), in_ch=3, out_ch=4)

    images_list = sorted(os.listdir(inputs_dir))
    pbar = tqdm(total=len(images_list))

    for image_name in images_list:
        img = Image.open(os.path.join(inputs_dir, image_name)).convert('RGB')
        img_size = img.size
        img = img.resize((768, 768), Image.BICUBIC)
        image_tensor = transform_fn(img)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        with torch.no_grad():
            output_tensor = net(image_tensor.to(device))
            output_tensor = F.log_softmax(output_tensor[0], dim=1)
            output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
            output_tensor = torch.squeeze(output_tensor, dim=0)
            output_arr = output_tensor.cpu().numpy()

        if cls == "all":
            classes_to_save = []

            # Check which classes are present in the image
            for cls in range(1, 4):  # Exclude background class (0)
                if np.any(output_arr == cls):
                    classes_to_save.append(cls)
        elif cls == "upper":
            classes_to_save = [1]
        elif cls == "lower":
            classes_to_save = [2]
        elif cls == "dress":
            classes_to_save = [3]
        else:
            raise ValueError(f"Unknown cls: {cls}")

        for cls1 in classes_to_save:
            alpha_mask = (output_arr == cls1).astype(np.uint8) * 255
            alpha_mask = alpha_mask[0]  # Selecting the first channel to make it 2D
            alpha_mask_img = Image.fromarray(alpha_mask, mode='L')
            alpha_mask_img = alpha_mask_img.resize(img_size, Image.BICUBIC)
            alpha_mask_img.save(os.path.join(outputs_dir, f'{image_name.split(".")[0]}_{cls1}.jpg'))

        pbar.update(1)

    pbar.close()


def extract_garment(inputs_dir, outputs_dir, cls="all", resize_to_width=None):
    os.makedirs(outputs_dir, exist_ok=True)
    cloth_mask_dir = os.path.join(Path(outputs_dir).parent.absolute(), "cloth-mask")
    os.makedirs(cloth_mask_dir, exist_ok=True)

    segment_garment(inputs_dir, os.path.join(Path(outputs_dir).parent.absolute(), "cloth-mask"), cls=cls)

    images_path = sorted(glob.glob(os.path.join(inputs_dir, "*")))
    masks_path = sorted(glob.glob(os.path.join(cloth_mask_dir, "*")))
    img = Image.open(images_path[0])

    for mask_path in masks_path:
        mask = Image.open(mask_path)

        cutout = np.array(naive_cutout(img, mask))
        cutout = resize_by_bigger_index(cutout)

        canvas = np.ones((1024, 768, 3), np.uint8) * 255
        y1, y2 = (canvas.shape[0] - cutout.shape[0]) // 2, (canvas.shape[0] + cutout.shape[0]) // 2
        x1, x2 = (canvas.shape[1] - cutout.shape[1]) // 2, (canvas.shape[1] + cutout.shape[1]) // 2

        alpha_s = cutout[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            canvas[y1:y2, x1:x2, c] = (alpha_s * cutout[:, :, c] +
                                       alpha_l * canvas[y1:y2, x1:x2, c])

        # resize image before saving
        if resize_to_width:
            canvas = image_resize(canvas, width=resize_to_width)

        canvas = Image.fromarray(canvas)

        canvas.save(os.path.join(outputs_dir, f"{os.path.basename(mask_path).split('.')[0]}.jpg"), format='JPEG')
