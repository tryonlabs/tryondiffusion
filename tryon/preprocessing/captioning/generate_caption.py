import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


def caption_image(image, question, model=None, processor=None):
    if model is None and processor is None:
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",
                                                                  torch_dtype=torch.float16, low_cpu_mem_usage=True)
        model.to("cuda:0")

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

    # auto-regressively complete prompt
    output = model.generate(**inputs, max_new_tokens=300)

    return processor.decode(output[0], skip_special_tokens=True).split("[/INST]")[-1]
