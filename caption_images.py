import glob
import json
import os

from PIL import Image

from tryon.preprocessing.captioning import caption_image, create_llava_next_pipeline

INPUT_IMAGES_DIR = os.path.join("fashion_datatset", "*")
OUTPUT_CAPTIONS_DIR = "fashion_datatset_captions"

os.makedirs(OUTPUT_CAPTIONS_DIR, exist_ok=True)


def change_extension(filename, new_extension):
    base_name, _ = os.path.splitext(filename)
    return f"{base_name}.{new_extension}"


if __name__ == '__main__':
    model, processor = create_llava_next_pipeline()

    images_path = sorted(glob.glob(INPUT_IMAGES_DIR))

    for index, image_path in enumerate(images_path):
        print(f"index: {index}, total images: {len(images_path)}, {image_path}")
        image = Image.open(image_path)

        prompt = """
        You're a fashion expert. The list of clothing properties includes [color, pattern, style, fit, type, hemline, 
        material, sleeve-length, fabric-elasticity, neckline, waistline]. Please provide the following information in 
        JSON format for the outfit shown in the image. Question: What are the color, pattern, style, fit, type, 
        hemline, material, sleeve length, fabric elasticity, neckline, and waistline of the outfit in the image?
        Answer: 
        """

        json_file_path = os.path.join(OUTPUT_CAPTIONS_DIR,
                                      change_extension(os.path.basename(image_path), "json"))
        caption_file_path = os.path.join(OUTPUT_CAPTIONS_DIR,
                                         change_extension(os.path.basename(image_path), "txt"))

        if os.path.exists(caption_file_path) and os.path.exists(json_file_path):
            print(f"caption already exists for {image_path}")
            continue

        json_data, generated_caption = caption_image(image, prompt, model, processor, json_only=False)

        with open(json_file_path, "w") as f:
            json.dump(json_data, f)

        with open(caption_file_path, "w") as f:
            f.write(generated_caption)
