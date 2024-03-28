import os
from typing import Union

import PIL.Image
import PIL.ImageOps
import requests
from schp_transforms import SCHPHumanParsing, get_palette

TEST_IMG_PATH_OR_URL = "https://media.discordapp.net/attachments/1150166741763248249/1171765385217966130/13876_00.jpg?ex=655dde8e&is=654b698e&hm=8a4e74c51c4a4b3d3dbe483291fc600f903ffa1fbd26d0f1189700b7ca7b3698&=&width=920&height=1228"
LABELS_TO_SEGMENT = ["Upper-clothes", "Pants"]
CHECKPOINT_PATH: str  # DOWNLOAD FROM https://drive.google.com/file/d/1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH/view


def load_image(image: Union[str, PIL.Image.Image]) -> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


if __name__ == "__main__":
    schp_transform = SCHPHumanParsing(model=CHECKPOINT_PATH, labels_to_segment=LABELS_TO_SEGMENT)
    image = load_image(TEST_IMG_PATH_OR_URL)

    # Show parsing result
    parsing_result = schp_transform.parse(image, return_pil=True)

    palette = get_palette(schp_transform.num_classes)
    parsing_result.putpalette(palette)
    parsing_result.show()

    # Show segmentation result
    output_img = schp_transform(image)
    output_img.show()

    # Show inverse segmentation result
    schp_transform.invert_mask = True
    output_img = schp_transform(image)
    output_img.show()
