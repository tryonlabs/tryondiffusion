import json
import os.path
import random
import time

import gradio as gr
import numpy as np
import spaces
import torch
from diffusers import FluxPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
PRE_TRAINED_MODEL = "black-forest-labs/FLUX.1-dev"
FINE_TUNED_MODEL = "tryonlabs/FLUX.1-dev-LoRA-Outfit-Generator"
RESULTS_DIR = "~/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float32

# Load Flux
pipe = FluxPipeline.from_pretrained(PRE_TRAINED_MODEL, torch_dtype=torch.float16).to("cuda")

# Load your fine-tuned model
pipe.load_lora_weights(FINE_TUNED_MODEL, adapter_name="default", weight_name="outfit-generator.safetensors")

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024


@spaces.GPU(duration=65)
def infer(
        prompt,
        seed=42,
        randomize_seed=False,
        width=1024,
        height=1024,
        guidance_scale=4.5,
        num_inference_steps=40,
        progress=gr.Progress(track_tqdm=True),
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator().manual_seed(seed)

    image = pipe(prompt, height=width, width=height, num_images_per_prompt=1, generator=generator,
                 guidance_scale=guidance_scale,
                 num_inference_steps=num_inference_steps).images[0]

    try:
        # save image
        current_time = int(time.time() * 1000)
        image.save(os.path.join(RESULTS_DIR, f"gen_img_{current_time}.png"))
        with open(os.path.join(RESULTS_DIR, f"gen_img_{current_time}.json"), "w") as f:
            json.dump({"prompt": prompt, "height": height, "width": width, "guidance_scale": guidance_scale,
                       "num_inference_steps": num_inference_steps, "seed": seed}, f)
    except Exception as e:
        print(str(e))

    return image, seed


examples = [
    "stripe red striped jersey top in a soft cotton and modal blend with short sleeves a chest pocket and rounded hem",
    "A dress with Color: Orange, Department: Dresses, Detail: Split Thigh, Fabric-Elasticity: No Sretch, Fit: Fitted, Hemline: Slit, Material: Gabardine, Neckline: Gathered, Pattern: Tropical, Sleeve-Length: Sleeveless, Style: Boho, Type: A Line Skirt, Waistline: High",
    "treatment dark pink knee-length skirt in crocodile-patterned imitation leather high waist with belt loops and press-studs a zip fly diagonal side pockets and a slit at the front the polyester content of the skirt is partly recycled",
    "A dress with Color: Maroon, Department: Dresses, Detail: Ruched Bust, Fabric-Elasticity: Slight Stretch, Fit: Fitted, Hemline: Slit, Material: Gabardine, Neckline: Spaghetti Straps, Pattern: Floral, Sleeve-Length: Sleeveless, Style: Boho, Type: Cami Top, Waistline: Regular",
    "denim dark blue 5-pocket ankle-length jeans in washed stretch denim slightly looser fit with a wide waist panel for best fit over the tummy and tapered legs with raw-edge frayed hems"
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 768px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""
        # FLUX.1-dev LoRA Outfit Generator 
        ## by TryOn Labs (https://www.tryonlabs.ai)
        Generate an outfit by describing the color, pattern, fit, style, material, type, etc.
        """)
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )

            run_button = gr.Button("Run", scale=0, variant="primary")

        result = gr.Image(label="Result", show_label=False)

        with gr.Accordion("Advanced Settings", open=False):
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=512,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )

                height = gr.Slider(
                    label="Height",
                    minimum=512,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )

            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.0,
                    maximum=7.5,
                    step=0.1,
                    value=4.5,
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=40,
                )

        gr.Examples(examples=examples, inputs=[prompt], outputs=[result, seed], fn=infer, cache_examples=True,
                    cache_mode="lazy")
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[
            prompt,
            seed,
            randomize_seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
        ],
        outputs=[result, seed],
    )

if __name__ == "__main__":
    demo.launch(share=True)
