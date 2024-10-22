# 使用OCR引导大模型输出

与专用OCR模型相比，当前多模态大模型的识字能力相对较弱。直接使用多模态大模型做视觉信息抽取往往会出现错字的问题。本项目使用OCR结果来引导大模型输出，以期得到更高的信息抽取准确率。



**使用Qwen2VL-2B的例子**
![](./imgs/bad_example.png)
||prompt|answer|
|---|---|---|
|origin|请给出图中的校验码，请只输出校验码不要输出多余内容|8094|
|ocr guided|请给出图中的校验码，请只输出校验码不要输出多余内容|809d|
|origin|请给出图中的个人自付，请只输出个人自付不要输出多余内容|31985.57|
|ocr guided|请给出图中的个人自付，请只输出个人自付不要输出多余内容|31695.57|

## 安装
```python
git clone https://github.com/hzauzxb/guidance-ocr.git
cd guidance_ocr
pip3 install .
```

## 使用方法
加载完transformers模型后，只需在调用generate函数时对模型进行warp。`warp_model`函数会将模型的generate方法替换为基于ocr辅助的generate方法，generate_config将不再起作用。
```python
from guidance_ocr import warp_model

model = AutoModel.from_pretrained(xxx)
tokenizer = AutoTokenizer.from_pretrained(XXX)

...

with warp_model(
        model,                          # VLM模型
        tokenizer = tokenizer,          # 对应的tokenizer
        text_list = ['姓名', '张三', ...],  # OCR识别出的文字
        model_type = 'qwen2vl',         # 模型类型，支持的模型见后续章节
        allow_texts = ['\n'],               # 除OCR文字外，允许大模型输出的其他token(主要用于一次性抽取多个字段的分隔符)
        top_k = 100                     # top_k采样的参数，默认100, 若模型效果不佳，则可以尝试设置较大的top-K
    ) as ocr_guided_model:
    generated_ids = ocr_guided_model.generate(
        **model_inputs,
        max_new_tokens=256
    )
print('guided')
print(tokenizer.decode(generated_ids[0]))

```
完整示例代码见`./examples`

## 支持的模型
```
qwen2vl:
Qwen/Qwen2-VL-2B-Instruct
Qwen/Qwen2-VL-7B-Instruct
Qwen/Qwen2-VL-72B-Instruct

internvl2:
OpenGVLab/InternVL2-1B # 不推荐，模型太小，信息抽取效果较差
OpenGVLab/InternVL2-2B # 不推荐，模型太小，信息抽取效果较差
OpenGVLab/InternVL2-8B
OpenGVLab/InternVL2-26B
```
后续会支持更多模型(MiniCPM, GLM4-V, Llava等)，若有其他需支持的模型可以提issue

## 特性
**支持多种常见的信息抽取样式**

|情况说明|需抽取的字段|图片|是否支持|
|---|---|---|---|
|需抽取的字段横跨多个文本框|住址|![](./imgs/idcard.jpg)|支持|
|需抽取的字段是文本框中的某一段|住宅用地使用起期|![](./imgs/date.png)|支持|

若有其他需支持的样式请提issue

**推理加速**

VLM模型通常有1k-2k的视觉token，若对同一张图问多个不同的问题，则可以复用之前视觉token对应的kv-cache。

若要使用加速，只需在`warp_model`中设置`accelerate=True`，并给出最多能缓存的图片数量`max_cache`
```python
with warp_model(
        model,                         
        tokenizer = tokenizer,         
        text_list = ['姓名', '张三', ...], 
        model_type = 'qwen2vl',        
        allow_texts = ['\n'],           
        top_k = 100,
        accelerate = True,
        max_cache = 20
    ) as ocr_guided_model:
    generated_ids = ocr_guided_model.generate(
        **model_inputs,
        max_new_tokens=256
    )
```

**使用Tips**
1. 一次模型调用只抽取一个字段，若一张图片有多个字段需要抽取则可以多次调用模型
2. 写prompt的时候需要提示模型仅输出答案，不要输出多余内容。建议使用如下的prompt格式(以使用Qwen2VL抽取姓名为例)
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|vision_content|><|vision_end|>请给出图中姓名，请仅输出姓名不要输出多余内容<|im_end|>
<|im_start|>assistant
姓名：
```