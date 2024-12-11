import time
from PIL import Image
import requests
import torch

from typing import Dict
from torchvision import io
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoConfig
from guidance_ocr import get_json_processor, OCRAssistModel

model_path = '/workspace/model_weights/qwen2vl_2B'
text_list = ['票据爱', '上海市医疗住院收费募据（电子）', '口', '票据代码：310602', '票据号码：10000347', '交款人统住会信用代码：34032119990909***', '校验码：', '809d', '交款人：', '开票日期：2021-04-14', '项目名称', '金额（元）', '备注', '项目名称', '金额（元)', '备注', '项目名称', '金额（元）备注', '床位费', '8,100.00', '检查费', '8,809.00', '化验费', '31,240.00', '治疗费', '8,483.00', '护理费', '1, 180.00', '卫生材料费', '35, 070.50', '西药费', '181,120.96', '其他住院费', '2,017.20', '金额合计（大写）贰拾柒万陆仟零贰拾元零陆角陆分', '（小写）276,020.66', '业务流水号：', '病历号：', '住院号：', '住院科别：', '其住院时间：20210323-20210414', '预缴金额：140000.00', '补缴金额：0.00', '送费金额：81589.23', '医疗机构类型：综合性三级甲等医医保类型：住院社保卡', '医保编号：', '性别：男', '他', '医保统筹基金支付：217609.89', '其他支付：0.00', '个人账户支付：0.00', '个人现金支付：58410.77', '个人自付：31695.57', '信', '电费：26715.20', '附加基金支付：0.00', '住院天数：22.5', '分类自费：11027.47', ':20668.10', '医保当年账户余额：0.00', '医保历年账户余额：0.00', '备注：财政部全国财政', 'tp://pjey.mof.gov.cn/：医保流水号：', '收款单位（学）：上海', '复核人：系统核验', '收款人：006212']
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)


assist_model = OCRAssistModel(
    text_list = text_list,
    model_path = model_path,
    model = model,
    model_type = 'qwen2_vl'
)

# assist_model = Qwen2VLForConditionalGeneration.from_pretrained(
#     model_path, torch_dtype="auto", device_map="auto"
# )

processor = AutoProcessor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

img_path = "test.jpg"
image = Image.open(img_path).convert('RGB')

# resize img for stable gpu useage and can be removed
w, h = image.size
if max(w, h) >= 1024:
    scale = 1024 / max(w, h)
    new_w, new_h = int(w*scale), int(h*scale)
    image = image.resize((new_w, new_h))


conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {"type": "text", "text": f"请给出图中的个人自付、票据代码、票据号码、开票日期、卫生材料费和校验码，以json格式输出答案，不要输出多余内容"},
        ],
    }
]

text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)


inputs = processor(
    text=[text_prompt], images=[image], padding=True, return_tensors="pt"
)
inputs = inputs.to("cuda")


logit_processor = get_json_processor(
    text_list = text_list, 
    extract_keys = ['个人自付', '票据代码', '票据号码', '开票日期', '卫生材料费', '校验码'],
    tokenizer = tokenizer,
    top_k = 50, model_type = 'qwen2vl',
    eos_id = 151645
)


# import pdb; pdb.set_trace()
st = time.time()
generated_ids = model.generate(
    **inputs,
    logits_processor = [logit_processor],
    max_new_tokens=128,
    assistant_model = assist_model
)
en = time.time()
print('------ Guidance OCR ------')
res = tokenizer.decode(generated_ids[0][inputs['input_ids'].shape[1]:])
print(res)#, f'\t运行时间{en - st}')