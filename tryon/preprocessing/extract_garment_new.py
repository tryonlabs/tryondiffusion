import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .u2net import load_cloth_segm_model
from .utils import NormalizeImage, naive_cutout, resize_by_bigger_index, image_resize


def extract_garment(image, cls="all", resize_to_width=None, net=None, device=None):
    """
    extracts garments from the given image
    :param image: input image
    :param cls: garment classes to extract
    :param resize_to_width: if required
    :return: extracted garments
    """

    if net is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = load_cloth_segm_model(device, os.environ.get("U2NET_CLOTH_SEGM_CHECKPOINT_PATH"), in_ch=3, out_ch=4)

    transform_fn = transforms.Compose(
        [transforms.ToTensor(),
         NormalizeImage(0.5, 0.5)]
    )

    img_size = image.size
    img = image.resize((768, 768), Image.BICUBIC)
    image_tensor = transform_fn(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    with torch.no_grad():
        output_tensor = net(image_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

    classes = {1: "upper", 2: "lower", 3: "dress"}

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

    garments = dict()

    for cls1 in classes_to_save:
        alpha_mask = (output_arr == cls1).astype(np.uint8) * 255
        alpha_mask = alpha_mask[0]  # Selecting the first channel to make it 2D
        alpha_mask_img = Image.fromarray(alpha_mask, mode='L')
        alpha_mask_img = alpha_mask_img.resize(img_size, Image.BICUBIC)

        cutout = np.array(naive_cutout(image, alpha_mask_img))
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

        garments[classes[cls1]] = canvas

    return garments
