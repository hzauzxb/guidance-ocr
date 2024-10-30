import time
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from guidance_ocr import get_json_processor


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
# path = 'OpenGVLab/InternVL2-8B'
path = '/workspace/model_weights/internvl2-8b'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
model.img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')

# set the max number of tiles in `max_num`
pixel_values = load_image('./test.jpg', max_num=12).to(torch.bfloat16).cuda()
generation_config = dict(max_new_tokens=1024, do_sample=True)


# single-image single-round conversation (单图单轮对话)
img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
num_patches_list = pixel_values.shape[0]



prompt = '<|im_start|>system\n你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。<|im_end|><|im_start|>user\n<img>'
prompt = prompt + f'<IMG_CONTEXT>' * num_patches_list * 256 + f'</img>请给出图中的个人自付、票据代码、票据号码、开票日期、卫生材料费和校验码，以json格式输出答案，不要输出多余内容"<|im_end|><|im_start|>assistant\n'
model_inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
input_ids = model_inputs['input_ids']
attention_mask = model_inputs['attention_mask']
    

st = time.time()
generation_output = model.generate(
    pixel_values=pixel_values,
    input_ids=input_ids,
    attention_mask = attention_mask,
    max_new_tokens = 128,
    eos_token_id = 92542,
    do_sample = False
)
en = time.time()
res = tokenizer.decode(generation_output[0])
print('------ Origin ------')
print(res)
print(f'time:{en-st}s')

text_list = ['票据爱', '上海市医疗住院收费募据（电子）', '口', '票据代码：310602', '票据号码：10000347', '交款人统住会信用代码：34032119990909***', '校验码：', '809d', '交款人：', '开票日期：2021-04-14', '项目名称', '金额（元）', '备注', '项目名称', '金额（元)', '备注', '项目名称', '金额（元）备注', '床位费', '8,100.00', '检查费', '8,809.00', '化验费', '31,240.00', '治疗费', '8,483.00', '护理费', '1, 180.00', '卫生材料费', '35, 070.50', '西药费', '181,120.96', '其他住院费', '2,017.20', '金额合计（大写）贰拾柒万陆仟零贰拾元零陆角陆分', '（小写）276,020.66', '业务流水号：', '病历号：', '住院号：', '住院科别：', '其住院时间：20210323-20210414', '预缴金额：140000.00', '补缴金额：0.00', '送费金额：81589.23', '医疗机构类型：综合性三级甲等医医保类型：住院社保卡', '医保编号：', '性别：男', '他', '医保统筹基金支付：217609.89', '其他支付：0.00', '个人账户支付：0.00', '个人现金支付：58410.77', '个人自付：31695.57', '信', '电费：26715.20', '附加基金支付：0.00', '住院天数：22.5', '分类自费：11027.47', ':20668.10', '医保当年账户余额：0.00', '医保历年账户余额：0.00', '备注：财政部全国财政', 'tp://pjey.mof.gov.cn/：医保流水号：', '收款单位（学）：上海', '复核人：系统核验', '收款人：006212']

logit_processor = get_json_processor(
    text_list = text_list, 
    extract_keys = ['个人自付', '票据代码', '票据号码', '开票日期', '卫生材料费', '校验码'],
    tokenizer = tokenizer,
    top_k = 100, model_type = 'internvl2',
    eos_id = 92542
)
st = time.time()
generated_ids = model.generate(
    pixel_values=pixel_values,
    input_ids=input_ids,
    attention_mask = attention_mask,
    max_new_tokens = 128,
    eos_token_id = 92542,
    logits_processor = [logit_processor],
)
en = time.time()
res = tokenizer.decode(generated_ids[0])
print('------ Guidance OCR ------')
print(res)
print(f'time:{en-st}s')


