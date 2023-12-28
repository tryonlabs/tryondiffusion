import numpy as np
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from .utils import create_model, NormalizeImage, get_palette


def segment(device, inputs_dir, outputs_dir, checkpoint_path):
    os.makedirs(os.path.join(outputs_dir, "cloth-mask"), exist_ok=True)

    transform_fn = transforms.Compose(
        [transforms.ToTensor(),
         NormalizeImage(0.5, 0.5)]
    )

    net = create_model(device, checkpoint_path)

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

        classes_to_save = []

        # Check which classes are present in the image
        for cls in range(1, 4):  # Exclude background class (0)
            if np.any(output_arr == cls):
                classes_to_save.append(cls)

        # Save alpha masks
        for cls in classes_to_save:
            alpha_mask = (output_arr == cls).astype(np.uint8) * 255
            alpha_mask = alpha_mask[0]  # Selecting the first channel to make it 2D
            alpha_mask_img = Image.fromarray(alpha_mask, mode='L')
            alpha_mask_img = alpha_mask_img.resize(img_size, Image.BICUBIC)
            alpha_mask_img.save(os.path.join(outputs_dir, "cloth-mask", f'{image_name.split(".")[0]}.jpg'))

        # # Save final cloth segmentations
        # palette = get_palette(4)
        # cloth_seg = Image.fromarray(output_arr[0].astype(np.uint8), mode='P')
        # cloth_seg.putpalette(palette)
        # cloth_seg = cloth_seg.resize(img_size, Image.BICUBIC)
        # cloth_seg.save(os.path.join(outputs_dir, "final_seg", f'{image_name.split(".")[0]}.png'))

        # # save the original image
        # img = img.resize(img_size, Image.BICUBIC)
        # img.save(os.path.join(outputs_dir, "original", f'{image_name.split(".")[0]}.png'))

        pbar.update(1)

    pbar.close()
