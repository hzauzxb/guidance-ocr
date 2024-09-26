from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from guidance_ocr.transformers import warp_tfmodel

model_path = '/workspace/model_weights/qwen2vl_2B'

# Load the model in half-precision on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

# Image
url = "test.png"
image = Image.open(url)

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {"type": "text", "text": "请给出图中的地址，不要输出多余内容"},
        ],
    }
]


# Preprocess the inputs
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
# Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

inputs = processor(
    text=[text_prompt], images=[image], padding=True, return_tensors="pt"
)
inputs = inputs.to("cuda")

import pdb; pdb.set_trace()

# # Inference: Generation of the output
# output_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids = [
#     output_ids[len(input_ids) :]
#     for input_ids, output_ids in zip(inputs.input_ids, output_ids)
# ]
tokenizer = AutoTokenizer.from_pretrained(model_path)



with warp_tfmodel(model, tokenizer, ['郭美金', '女', '住址', '深圳市', '南山区', '龙都花园', '阳台']) as ocr_guided_model:
    generated_ids = ocr_guided_model.generate(
        **inputs,
        max_new_tokens=128
    )
print('guided', tokenizer.decode(generated_ids[0]))

