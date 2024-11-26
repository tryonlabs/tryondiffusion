import glob
import json
import os

import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from tryon.preprocessing.captioning import generate_caption

INPUT_IMAGES_DIR = os.path.join("fashion_dataset", "*")
OUTPUT_CAPTIONS_DIR = "fashion_dataset_captions"
os.makedirs(OUTPUT_CAPTIONS_DIR, exist_ok=True)

if __name__ == '__main__':
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",
                                                              torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to("cuda:0")

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

        caption_file_path = os.path.join(OUTPUT_CAPTIONS_DIR,
                                         os.path.basename(image_path).replace(".png", ".json"))

        if os.path.exists(caption_file_path):
            print(f"caption already exists for {image_path}")
            continue

        generated_caption = generate_caption(image, prompt, model, processor)

        with open(caption_file_path, "w") as f:
            json.dump(json.loads(generated_caption.replace("```json", "").replace("```", "")), f)
