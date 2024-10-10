from transformers import AutoModelForCausalLM, AutoTokenizer
from guidance_ocr import warp_model
import time
MODEL_PATH = '/workspace/model_weights/qwen2.5_1-1.5B'


model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
prompt = "请从下列公司中抽取保险公司的名称：支付宝，拼多多，腾讯，蚂蚁保险，吉利汽车，中国平安\n若有多个公司请用\\n隔开，不要输出多余内容"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)



generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=256
)
print('not guided')
print(tokenizer.decode(generated_ids[0]))

with warp_model(
        model, tokenizer = tokenizer, 
        text_list = ['蚂蚁保险', '中国平安', '中国太保', '阳光保险', '蚂蚁金服'], 
        model_type = 'llms', allow_texts = ['\n']
    ) as ocr_guided_model:
    generated_ids = ocr_guided_model.generate(
        **model_inputs,
        max_new_tokens=256
    )
print('guided')
print(tokenizer.decode(generated_ids[0]))
