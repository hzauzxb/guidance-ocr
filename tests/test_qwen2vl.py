from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from guidance_ocr import warp_model

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
            {"type": "text", "text": "请给出图中的姓名，不要输出多余内容"},
        ],
    }
]


text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(
    text=[text_prompt], images=[image], padding=True, return_tensors="pt"
)
inputs = inputs.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)
generated_ids = model.generate(
    **inputs,
    max_new_tokens=128
)
print('not guided')
print(tokenizer.decode(generated_ids[0][inputs['input_ids'].shape[1]:]))



with warp_model(
        model, tokenizer = tokenizer, 
        text_list = ['郭美全', '女', '住址', '深圳市', '南山区', '龙都花园', '阳台'], 
        model_type = 'qwen2vl', allow_texts = []
    ) as ocr_guided_model:
    generated_ids = ocr_guided_model.generate(
        **inputs,
        max_new_tokens=128
    )
print('guided')
print(tokenizer.decode(generated_ids[0]))

