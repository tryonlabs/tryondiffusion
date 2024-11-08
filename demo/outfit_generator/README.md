# FLUX.1-dev LoRA Outfit Generator Gradio Demo
## by TryOn Labs (https://www.tryonlabs.ai)
Generate an outfit by describing the color, pattern, fit, style, material, type, etc.

## Model description 

FLUX.1-dev LoRA Outfit Generator can create an outfit by detailing the color, pattern, fit, style, material, and type.

## Inference

```
import random

from diffusers import FluxPipeline
import torch

seed=42
prompt = "denim dark blue 5-pocket ankle-length jeans in washed stretch denim slightly looser fit with a wide waist panel for best fit over the tummy and tapered legs with raw-edge frayed hems"
PRE_TRAINED_MODEL = "black-forest-labs/FLUX.1-dev"
FINE_TUNED_MODEL = "tryonlabs/FLUX.1-dev-LoRA-Outfit-Generator"

# Load Flux
pipe = FluxPipeline.from_pretrained(PRE_TRAINED_MODEL, torch_dtype=torch.float16).to("cuda")

# Load fine-tuned model
pipe.load_lora_weights(FINE_TUNED_MODEL, adapter_name="default", weight_name="outfit-generator.safetensors")

seed = random.randint(0, MAX_SEED)

generator = torch.Generator().manual_seed(seed)

image = pipe(prompt, height=1024, width=1024, num_images_per_prompt=1, generator=generator, 
guidance_scale=4.5, num_inference_steps=40).images[0]

image.save("gen_image.jpg")
```

## Dataset used

H&M Fashion Captions Dataset - 20.5k samples
https://huggingface.co/datasets/tomytjandra/h-and-m-fashion-caption

## Repository used

AI Toolkit by Ostris
https://github.com/ostris/ai-toolkit

## Download model

Weights for this model are available in Safetensors format.

[Download](https://huggingface.co/tryonlabs/FLUX.1-dev-LoRA-Outfit-Generator/tree/main) them in the Files & versions tab.

## Install dependencies

```
git clone https://github.com/tryonlabs/FLUX.1-dev-LoRA-Outfit-Generator.git
cd FLUX.1-dev-LoRA-Outfit-Generator
conda create -n demo python=3.12
pip install -r requirements.txt
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
```

## Run demo

```
gradio app.py
```

## Generated images

![alt](images/sample1.jpeg "sample1")
#### A dress with Color: Black, Department: Dresses, Detail: High Low,Fabric-Elasticity: No Sretch, Fit: Fitted, Hemline: Slit, Material: Gabardine, Neckline: Collared, Pattern: Solid, Sleeve-Length: Sleeveless, Style: Casual, Type: Tunic, Waistline: Regular
***
![alt](images/sample2.jpeg "sample2")
#### A dress with Color: Red, Department: Dresses, Detail: Belted, Fabric-Elasticity: High Stretch, Fit: Fitted, Hemline: Flared, Material: Gabardine, Neckline: Off The Shoulder, Pattern: Floral, Sleeve-Length: Sleeveless, Style: Elegant, Type: Fit and Flare, Waistline: High
***
![alt](images/sample3.jpeg "sample3")
#### A dress with Color: Multicolored, Department: Dresses, Detail: Split, Fabric-Elasticity: High Stretch, Fit: Fitted, Hemline: Slit, Material: Gabardine, Neckline: V Neck, Pattern: Leopard, Sleeve-Length: Sleeveless, Style: Casual, Type: T Shirt, Waistline: Regular
***
![alt](images/sample4.jpeg "sample4")
#### A dress with Color: Brown, Department: Dresses, Detail: Zipper, Fabric-Elasticity: No Sretch, Fit: Fitted, Hemline: Asymmetrical, Material: Satin, Neckline: Spaghetti Straps, Pattern: Floral, Sleeve-Length: Sleeveless, Style: Boho, Type: Cami Top, Waistline: High
***

## License
MIT [License](LICENSE)
