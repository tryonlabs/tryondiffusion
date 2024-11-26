import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


def caption_image(image, question, model=None, processor=None, json_only=False):
    """
    Extract outfit details using an image-to-text model
    :param image: input image
    :param question: question
    :param model: model pipeline
    :param processor: processor
    :param json_only: True or False - if json only
    :return: json data
    """
    if model is None and processor is None:
        model, processor = create_llava_next_pipeline()

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(image, prompt, return_tensors="pt").to("cuda:0")

    output = model.generate(**inputs, max_new_tokens=300)
    output = processor.decode(output[0], skip_special_tokens=True).split("[/INST]")[-1]
    json_data = json.loads(output.replace("```json", "").replace("```", "").strip())

    if not json_only:
        generated_caption = convert_outfit_json_to_caption(json_data)
    else:
        generated_caption = None

    return json_data, generated_caption


def create_phi35mini_pipeline():
    """
    Create Phi-3.5-mini-instruct pipeline
    :return: model pipeline
    """
    torch.random.manual_seed(0)

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct",
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    return pipe


def create_llava_next_pipeline():
    """
    Create LlaVA-NeXT pipeline
    :return: model pipeline
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",
                                                              torch_dtype=torch.float16,
                                                              low_cpu_mem_usage=True)
    model.to(device)

    return model, processor


def convert_outfit_json_to_caption(json_data, pipe=None):
    """
    Convert JSON data of an outfit into a natural language caption
    :param json_data: json data
    :param pipe: model pipeline
    :return: generated caption
    """
    if pipe is None:
        pipe = create_phi35mini_pipeline()

    generation_args = {
        "max_new_tokens": 300,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    messages = [{"role": "user",
                 "content": f'Convert the {json.dumps(json_data)} JSON data into a natural '
                            f'language paragraph beginning with "An outfit with"'}]

    output = pipe(messages, **generation_args)[0]['generated_text'].strip()
    print(f"Output: {output}")
    return output
