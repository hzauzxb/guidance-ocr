from transformers import AutoModel

model = AutoModel.from_pretrained(
    '/workspace/sources/guidance_ocr/internvl2-8b', 
    device_map = 'auto', trust_remote_code=True
)
