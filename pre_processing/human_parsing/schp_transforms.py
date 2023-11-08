from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from schp import resnet101
from torchvision import transforms

LABELS = [
    "Background",
    "Hat",
    "Hair",
    "Glove",
    "Sunglasses",
    "Upper-clothes",
    "Dress",
    "Coat",
    "Socks",
    "Pants",
    "Jumpsuits",
    "Scarf",
    "Skirt",
    "Face",
    "Left-arm",
    "Right-arm",
    "Left-leg",
    "Right-leg",
    "Left-shoe",
    "Right-shoe",
]

LABELS_DICT = {label: index for index, label in enumerate(LABELS)}


class SCHPHumanParsing:
    def __init__(
        self,
        model: Union[str, torch.nn.Module],
        input_size: Tuple[int, int] = (473, 473),
        num_classes: int = 20,
        labels_to_segment: Union[str, List[str]] = "Upper-clothes",
        invert_mask: bool = False,
        mask_value: int = 127,
    ):
        self.input_size = input_size
        self.num_classes = num_classes
        self.aspect_ratio = input_size[1] * 1.0 / input_size[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prepare the labels to segment
        if isinstance(labels_to_segment, str):
            labels_to_segment = [labels_to_segment]  # Convert to list if only one label is provided
        self.labels_to_segment_indices = [LABELS_DICT[label] for label in labels_to_segment]
        self.invert_mask = invert_mask
        self.mask_value = mask_value

        # Load the pre-trained SCHP model
        if isinstance(model, torch.nn.Module):
            self.model = model
        elif isinstance(model, str):
            self.model = self._load_schp_model(model)
        else:
            raise ValueError("model must be either a torch.nn.Module or a path to a .pth file")

        # Define the transform function for input images
        self.img_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])]
        )

    def _load_schp_model(self, model_path: str):
        model = resnet101(num_classes=self.num_classes, pretrained=None)
        state_dict = torch.load(model_path)["state_dict"]
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k[7:]  # remove `module.` prefix
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model.to(self.device)
        model.eval()  # Set the model to inference mode
        return model

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    @torch.no_grad()
    def parse(self, img: Image.Image, return_pil=False) -> Union[np.array, Image.Image]:
        # Convert PIL Image to OpenCV format
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w, _ = img.shape

        # Get person center and scale
        input_size = np.asarray(self.input_size)
        c, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(c, s, r, input_size)
        img = cv2.warpAffine(
            img,
            trans,
            (int(input_size[1]), int(input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # Transform and normalize the image
        img = self.img_transforms(img)
        img = img.unsqueeze(0)

        # Inference
        img = img.to(self.device)
        output = self.model(img)

        # Upsample
        upsample = torch.nn.Upsample(size=self.input_size, mode="bilinear", align_corners=True)
        upsample_output = upsample(output[0][-1][0].unsqueeze(0))
        upsample_output = upsample_output.squeeze()
        upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

        logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=self.input_size)
        parsing_result = np.argmax(logits_result, axis=2)

        if return_pil:
            parsing_result = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
        return parsing_result

    def segment(self, img: Image.Image) -> Image.Image:
        parsing_result = self.parse(img)
        img_np = np.array(img)

        # Create a mask for selected labels
        selected_labels_mask = np.isin(parsing_result, self.labels_to_segment_indices)

        if self.invert_mask:
            # Keep only selected labels pixels
            img_np[~selected_labels_mask] = self.mask_value
        else:
            # Gray out the entire person bounding box to remove the selected garment region aggressively
            non_background_mask = parsing_result != LABELS_DICT["Background"]

            # Create the bounding box of the non-background (person) and Gray out the person area within bounding box
            person_bbox = cv2.boundingRect(non_background_mask.astype(np.uint8))
            x, y, w, h = person_bbox
            original_img_np = img_np.copy()
            img_np[y : y + h, x : x + w] = self.mask_value

            # Paste back all person parts that are not the garment or 'Background'
            person_parts_mask = non_background_mask & ~selected_labels_mask
            img_np[person_parts_mask] = original_img_np[person_parts_mask]

        # Convert back to PIL Image and return
        masked_img = Image.fromarray(img_np)

        return masked_img

    def __call__(self, img: Image.Image):
        return self.segment(img)


class SCHPHumanParsingKP(SCHPHumanParsing):
    def __call__(self, sample: Dict, image_key: str, keypoints_key: str) -> Dict:
        img = sample[image_key]
        sample[image_key] = self.segment(img)
        return sample


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale

    src_w = scale_tmp[0]
    dst_w = output_size[1]
    dst_h = output_size[0]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, (dst_w - 1) * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]
    dst[1, :] = np.array([(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def transform_logits(logits, center, scale, width, height, input_size):
    trans = get_affine_transform(center, scale, 0, input_size, inv=1)
    channel = logits.shape[2]
    target_logits = []
    for i in range(channel):
        target_logit = cv2.warpAffine(
            logits[:, :, i],
            trans,
            (int(width), int(height)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0),
        )
        target_logits.append(target_logit)
    target_logits = np.stack(target_logits, axis=2)

    return target_logits


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_palette(num_cls):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette
