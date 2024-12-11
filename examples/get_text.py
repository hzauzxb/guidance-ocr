
from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np
# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
img_path = './test.jpg'
result = ocr.ocr(img_path, cls=True)[0]
text_list = [data[1][0] for data in result]
print(text_list)

img = cv2.imread('test.jpg')
for data in result:
    box = np.array(data[0])
    img = cv2.polylines(img, [box.astype(np.int32)], color = (0,0,255), thickness=2, isClosed=True)
cv2.imwrite('det.jpg', img)