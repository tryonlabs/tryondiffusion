from typing import List, Union

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForSemanticSegmentation,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)


class SegformerHumanParsing:
    LABELS = [
        "Background",  # 0
        "Hat",  # 1
        "Hair",  # 2
        "Sunglasses",  # 3
        "Upper-clothes",  # 4
        "Skirt",  # 5
        "Pants",  # 6
        "Dress",  # 7
        "Belt",  # 8
        "Left-shoe",  # 9
        "Right-shoe",  # 10
        "Face",  # 11
        "Left-leg",  # 12
        "Right-leg",  # 13
        "Left-arm",  # 14
        "Right-arm",  # 15
        "Bag",  # 16
        "Scarf",  # 17
    ]
    LABELS_DICT = {label: index for index, label in enumerate(LABELS)}

    def __init__(
        self,
        model: Union[str, SegformerForSemanticSegmentation] = "mattmdjaga/segformer_b2_clothes",
        processor: Union[str, SegformerImageProcessor] = "mattmdjaga/segformer_b2_clothes",
        labels_to_segment: Union[str, List[str]] = "Upper-clothes",
        invert_mask: bool = False,
        mask_value: int = 127,
    ):
        self.processor = SegformerImageProcessor.from_pretrained(model) if isinstance(processor, str) else processor
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model) if isinstance(model, str) else model

        self.labels_to_segment = [labels_to_segment] if isinstance(labels_to_segment, str) else labels_to_segment
        self.labels_to_segment_indices = [self.LABELS_DICT[label] for label in labels_to_segment]

        self.invert_mask = invert_mask
        self.mask_value = mask_value

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @property
    def num_classes(self):
        return len(self.LABELS)

    @torch.no_grad()
    def parse(self, img: Image.Image, return_pil=False) -> Union[np.array, Image.Image]:
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits.cpu()

        # Upsample logits to match the size of the original image
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=img.size[::-1], mode="bilinear", align_corners=False
        )

        # Get the most likely class for each pixel
        pred_seg = upsampled_logits.argmax(dim=1)[0]

        # Convert logits to numpy array for processing
        pred_seg_np = pred_seg.numpy()

        if return_pil:
            return Image.fromarray(pred_seg_np.astype(np.uint8))
        else:
            return pred_seg_np

    def segment(self, img: Image.Image) -> Image.Image:
        pred_seg_np = self.parse(img)

        # Create a mask for selected labels
        selected_labels_mask = np.isin(pred_seg_np, self.labels_to_segment_indices)

        img_np = np.array(img)

        if self.invert_mask:
            # Keep only selected labels pixels
            img_np[~selected_labels_mask] = self.mask_value
        else:
            # Gray out the entire person bounding box to remove the selected labels aggressively
            non_background_mask = pred_seg_np != self.LABELS_DICT["Background"]

            # Create the bounding box of the non-background (person) and gray out the person area within bounding box
            person_bbox = cv2.boundingRect(non_background_mask.astype(np.uint8))
            x, y, w, h = person_bbox
            original_img_np = img_np.copy()
            img_np[y : y + h, x : x + w] = self.mask_value

            # Paste back all person parts that are not the selected labels or 'Background'
            person_parts_mask = non_background_mask & ~selected_labels_mask
            img_np[person_parts_mask] = original_img_np[person_parts_mask]

        # Convert back to PIL Image and return
        masked_img = Image.fromarray(img_np)

        return masked_img

    def __call__(self, img: Image.Image):
        return self.segment(img)


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
