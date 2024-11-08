import glob
import os
from PIL import Image

import gradio as gr
from tryon import preprocessing


def extract_garment(input_img, cls):
    print(input_img, type(input_img), cls)

    input_dir = "input_image"
    output_dir = "output_image"

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    for f in glob.glob(input_dir + "/*.*"):
        os.remove(f)

    for f in glob.glob(output_dir + "/*.*"):
        os.remove(f)

    for f in glob.glob("cloth-mask/*.*"):
        os.remove(f)

    input_img.save(os.path.join(input_dir, "img.jpg"))

    preprocessing.extract_garment(inputs_dir=input_dir, outputs_dir=output_dir, cls=cls)

    return Image.open(glob.glob(output_dir + "/*.*")[0])


css = """
#col-container {
    margin: 0 auto;
    max-width: 720px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""
        # Clothes Extraction using U2Net
        Pull out clothes like tops, bottoms, and dresses from a photo. This implementation is based on the [U2Net](https://github.com/xuebinqin/U-2-Net) model.
        """)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type='pil', height="400px", show_label=True)
                dropdown = gr.Dropdown(["upper", "lower", "dress"], value="upper", label="Extract garment",
                                       info="Select the garment type you wish to extract!")

            output_image = gr.Image(label="Extracted garment", type='pil', height="400px", show_label=True,
                                    show_download_button=True)

        with gr.Row():
            submit_button = gr.Button("Submit", variant='primary', scale=1)
            reset_button = gr.ClearButton(value="Reset", scale=1)

    gr.on(
        triggers=[submit_button.click],
        fn=extract_garment,
        inputs=[input_image, dropdown],
        outputs=[output_image]
    )

    reset_button.click(
        fn=lambda: (None, "upper", None),
        inputs=[],
        outputs=[input_image, dropdown, output_image],
        concurrency_limit=1,
    )

if __name__ == '__main__':
    demo.launch()
