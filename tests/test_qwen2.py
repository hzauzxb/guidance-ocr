from transformers import AutoModelForCausalLM, AutoTokenizer
from guidance_ocr.transformers import warp_tfmodel
import time
MODEL_PATH = '/workspace/model_weights/qwen2.5_1-1.5B'


model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
prompt = "请从以下文本中抽取日期：2023年9月25日，科技公司XYZ发布了其最新的智能手机产品——XYZ Pro 12。该手机配备了最新的处理器，支持5G网络，并配有128GB的存储空间。XYZ公司首席执行官在发布会上表示：“我们希望通过这款手机为用户带来更快的体验。"
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

text_list = [
    '2023-09-25',
    'N'
]

# st = time.time()
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=256
)
# en = time.time()
print('Not guided', tokenizer.decode(generated_ids[0]))

with warp_tfmodel(model, tokenizer, text_list) as ocr_guided_model:
    st = time.time()
    generated_ids = ocr_guided_model.generate(
        **model_inputs,
        max_new_tokens=256
    )
    en = time.time()
    print(len(generated_ids[0]) / (en - st))

print('guided', tokenizer.decode(generated_ids[0]))

# 中国平安是中国最大的金融控股公司之一，成立于1988年，总部位于深圳。公司业务涵盖保险、银行、证券、资产管理、信托、养老、医疗健康等多个领域，是中国金融行业的重要参与者和领导者。

# 中国平安在保险业务方面具有显著优势，旗下拥有平安人寿、平安财险、平安健康险等多家保险公司，提供包括寿险、健康险、意外险、财产险等多种保险产品。此外，公司还涉足银行、证券、资产管理、信托等多个金融领域，形成了较为完整的金融产业链。

# 中国平安在科技方面也取得了显著成就，拥有平安科技、平安好医生等子公司，致力于推动金融科技创新，提供包括人工智能、大数据、云计算等在内的金融科技解决方案。此外，公司还积极参与公益事业，致力于推动社会进步和可持续发展。

# 中国平安在国内外市场都具有较高的知名度和影响力，是中国金融行业的重要代表之一。<|im_end|>