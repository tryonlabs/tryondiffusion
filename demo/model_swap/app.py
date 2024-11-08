import os.path

import gradio as gr
import json
import requests
import time
from gradio_modal import Modal
from io import BytesIO

TRYON_SERVER_HOST = "https://prod.server.tryonlabs.ai"
TRYON_SERVER_PORT = "80"
if TRYON_SERVER_PORT == "80":
    TRYON_SERVER_URL = f"{TRYON_SERVER_HOST}"
else:
    TRYON_SERVER_URL = f"{TRYON_SERVER_HOST}:{TRYON_SERVER_PORT}"

TRYON_SERVER_API_URL = f"{TRYON_SERVER_URL}/api/v1/"


def start_model_swap(input_image, prompt, cls, seed, guidance_scale, num_results, strength, inference_steps):
    # make a request to TryOn Server
    # 1. create an experiment image
    print("inputs:", input_image, prompt, cls, seed, guidance_scale, num_results, strength, inference_steps)

    if input_image is None:
        raise gr.Error("Select an image!")

    if prompt is None or prompt == "":
        raise gr.Error("Enter a prompt!")

    token = load_token()
    if token is None or token == "":
        raise gr.Error("You need to login first!")
    else:
        login(token)

    byte_io = BytesIO()
    input_image.save(byte_io, 'png')
    byte_io.seek(0)

    r = requests.post(f"{TRYON_SERVER_API_URL}experiment_image/",
                      files={"image": (
                          'ei_image.png',
                          byte_io,
                          'image/png'
                      )},
                      data={
                          "type": "model",
                          "preprocess": "false"},
                      headers={
                          "Authorization": f"Bearer {token}"
                      })
    # print(r.json())
    if r.status_code == 200 or r.status_code == 201:
        print("Experiment image created successfully", r.json())
        res = r.json()
        # 2 create an experiment
        r2 = requests.post(f"{TRYON_SERVER_API_URL}experiment/",
                           data={
                               "model_id": res['id'],
                               "action": "model_swap",
                               "params": json.dumps({"prompt": prompt,
                                                     "guidance_scale": guidance_scale,
                                                     "strength": strength,
                                                     "num_inference_steps": inference_steps,
                                                     "seed": seed,
                                                     "garment_class": f"{cls} garment",
                                                     "negative_prompt": "(hands:1.15), disfigured, ugly, bad, immature"
                                                                        ", cartoon, anime, 3d, painting, b&w, (ugly),"
                                                                        " (pixelated), watermark, glossy, smooth, "
                                                                        "earrings, necklace",
                                                     "num_results": num_results})
                           },
                           headers={
                               "Authorization": f"Bearer {token}"
                           })
        if r2.status_code == 200 or r2.status_code == 201:
            # 3. keep checking the status of the experiment
            res2 = r2.json()
            print("Experiment created successfully", res2)
            time.sleep(10)

            experiment = res2['experiment']
            status = fetch_experiment_status(experiment_id=experiment['id'], token=token)
            status_status = status['status']
            while status_status == "running":
                time.sleep(10)
                status = fetch_experiment_status(experiment_id=experiment['id'], token=token)
                status_status = status['status']
                print(f"Current status: {status_status}")

            if status['status'] == "success":
                print("Experiment successful")
                print(f"Results:{status['result_images']}")
                return status['result_images']
            elif status['status'] == "failed":
                print("Experiment failed")
                raise gr.Error("Experiment failed")
        else:
            print(f"Error: {r2.text}")
            raise gr.Error(f"Failure: {r2.text}")
    else:
        print(f"Error: {r.text}")
        raise gr.Error(f"Failure: {r.text}")


def fetch_experiment_status(experiment_id, token):
    print(f"experiment id:{experiment_id}")

    r3 = requests.get(f"{TRYON_SERVER_API_URL}experiment/{experiment_id}/",
                      headers={
                          "Authorization": f"Bearer {token}"
                      })
    if r3.status_code == 200:
        res = r3.json()
        if res['status'] == "running":
            return {"status": "running"}
        elif res['status'] == "success":
            experiment = r3.json()['experiment']
            result_images = [f"{TRYON_SERVER_URL}/{experiment['result']['image_url']}"]
            if len(experiment['results']) > 0:
                for result in experiment['results']:
                    result_images.append(f"{TRYON_SERVER_URL}/{result['image_url']}")
            return {"status": "success", "result_images": result_images}
        elif res['status'] == "failed":
            return {"status": "failed"}
    else:
        print(f"Error: {r3.text}")
        return {"status": "failed"}


def get_user_credits(token):
    if token == "":
        return None

    r = requests.get(f"{TRYON_SERVER_API_URL}user/get/", headers={
        "Authorization": f"Bearer {token}"
    })
    if r.status_code == 200:
        res = r.json()
        return res['credits']
    else:
        print(f"Error: {r.text}")
        return None


def load_token():
    if os.path.exists(".token"):
        with open(".token", "r") as f:
            return json.load(f)['token']
    else:
        return None


def save_token(access_token):
    if access_token != "":
        with open(".token", "w") as f:
            json.dump({"token": access_token}, f)
    else:
        raise gr.Error("No token provided!")


def is_logged_in():
    loaded_token = load_token()
    if loaded_token is None or loaded_token == "":
        return False
    else:
        return True


def login(token):
    print("logging in...")
    # validate token
    r = requests.post(f"{TRYON_SERVER_URL}/api/token/verify/", data={"token": token})
    if r.status_code == 200:
        save_token(token)
        return True
    else:
        raise gr.Error("Login failed")


def logout():
    print("logged out")
    with open(".token", "w") as f:
        json.dump({"token": ""}, f)
    return [False, ""]


css = """
#col-container {
    margin: 0 auto;
    max-width: 1024px;
}
#credits-col-container{
    display:flex;
    justify-content: right;
    align-items: center;
    font-size: 24px;
    margin-right: 1rem;
}
#login-modal{
    max-width: 728px;
    margin: 0 auto;
    margin-top: 1rem;
    margin-bottom: 1rem;
}
#login-logout-btn{
    display:inline;
    max-width: 124px;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Default()) as demo:
    print("is logged in:", is_logged_in())
    logged_in = gr.State(is_logged_in())
    if os.path.exists(".token"):
        with open(".token", "r") as f:
            user_token = gr.State(json.load(f)["token"])
    else:
        user_token = gr.State("")

    with Modal(visible=False) as modal:
        @gr.render(inputs=user_token)
        def rerender1(user_token1):
            with gr.Column(elem_id="login-modal"):
                access_token = gr.Textbox(
                    label="Token",
                    lines=1,
                    value=user_token1,
                    type="password",
                    placeholder="Enter your access token here!",
                    info="Visit https://playground.tryonlabs.ai to retrieve your access token."
                )

                login_submit_btn = gr.Button("Login", scale=1, variant='primary')
                login_submit_btn.click(
                    fn=lambda access_token: (login(access_token), Modal(visible=False), access_token),
                    inputs=[access_token], outputs=[logged_in, modal, user_token],
                    concurrency_limit=1)

    with gr.Row(elem_id="col-container"):
        with gr.Column():
            gr.Markdown(f"""
            # Model Swap AI
            ## by TryOn Labs (https://www.tryonlabs.ai)
            Swap a human model with a artificial model generated by Artificial Model while keeping the garment intact.
            """)


        @gr.render(inputs=logged_in)
        def rerender(is_logged_in):
            with gr.Column():
                if not is_logged_in:
                    with gr.Row(elem_id="credits-col-container"):
                        login_btn = gr.Button(value="Login", variant='primary', elem_id="login-logout-btn", size="sm")
                        login_btn.click(lambda: Modal(visible=True), None, modal)
                else:
                    user_credits = get_user_credits(load_token())
                    print("user_credits", user_credits)
                    gr.HTML(f"""<div><p id="credits-col-container">Your Credits: 
                    {user_credits if user_credits is not None else "0"}</p>
                    <p style="text-align: right;">Visit <a href="https://playground.tryonlabs.ai">
                    TryOn AI Playground</a> to acquire more credits</p></div>""")
                    with gr.Row(elem_id="credits-col-container"):
                        logout_btn = gr.Button(value="Logout", scale=1, variant='primary', size="sm",
                                               elem_id="login-logout-btn")
                        logout_btn.click(fn=logout, inputs=None, outputs=[logged_in, user_token], concurrency_limit=1)

    with gr.Column(elem_id="col-container"):
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Original image", type='pil', height="400px", show_label=True)
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    placeholder="Enter your prompt here!",
                )
                dropdown = gr.Dropdown(["upper", "lower", "dress"], value="upper", label="Retain garment",
                                       info="Select the garment type you want to retain in the generated image!")

            gallery = gr.Gallery(
                label="Generated images", show_label=True, elem_id="gallery"
                , columns=[3], rows=[1], object_fit="contain", height="auto")

            # output_image = gr.Image(label="Swapped model", type='pil', height="400px", show_label=True,
            #                         show_download_button=True)

        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                seed = gr.Number(label="Seed", value=-1, interactive=True, minimum=-1)
                guidance_scale = gr.Number(label="Guidance Scale", value=7.5, interactive=True, minimum=0.0,
                                           maximum=10.0,
                                           step=0.1)
                num_results = gr.Number(label="Number of results", value=2, minimum=1, maximum=5)

            with gr.Row():
                strength = gr.Slider(0.00, 1.00, value=0.99, label="Strength",
                                     info="Choose between 0.00 and 1.00", step=0.01, interactive=True)
                inference_steps = gr.Number(label="Inference Steps", value=20, interactive=True, minimum=1, step=1)

        with gr.Row():
            submit_button = gr.Button("Submit", variant='primary', scale=1)
            reset_button = gr.ClearButton(value="Reset", scale=1)

    gr.on(
        triggers=[submit_button.click],
        fn=start_model_swap,
        inputs=[input_image, prompt, dropdown, seed, guidance_scale, num_results, strength, inference_steps],
        outputs=[gallery]
    )

    reset_button.click(
        fn=lambda: (None, None, "upper", None, -1, 7.5, 2, 0.99, 20),
        inputs=[],
        outputs=[input_image, prompt, dropdown, gallery, seed, guidance_scale,
                 num_results, strength, inference_steps],
        concurrency_limit=1,
    )

if __name__ == '__main__':
    demo.launch()
