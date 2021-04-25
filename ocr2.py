""" experiment for ocr with line extraction """

import cv2
import numpy as np
import pytesseract
from PIL import Image

import os

image_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/natsuki'

language = 'jpn_vert'
config = '--psm 3'
output_type = 'dict'

bgr = 212, 0, 255
color = np.array(bgr, dtype=np.uint8)

for filename in os.listdir(image_directory)[410:]:
    image_filepath = os.path.join(image_directory, filename)
    image = cv2.imread(image_filepath)

    img = (255 - image).astype(float)
    c = (255 - color[np.newaxis, np.newaxis, :]).astype(float)

    """ color mask """
    norm_img = np.linalg.norm(img, axis=2, keepdims=True)
    norm_c = np.linalg.norm(c, axis=2, keepdims=True)

    scale = np.divide(norm_c, norm_img, out=np.zeros_like(norm_img), where=norm_img != 0.0)
    scale = scale.clip(0.0, 1.0)
    img *= scale
    norm_img *= scale

    cos = np.sum(np.divide(img, norm_img, out=np.zeros_like(img), where=norm_img != 0.0) * (c / norm_c), axis=2,
                 keepdims=True)

    mask = (cos * norm_img / norm_c).squeeze()
    mask = (mask * 255).astype(np.uint8)

    """ text mask """
    text_mask = np.where(mask > 192, mask, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    text_mask = cv2.dilate(text_mask, kernel, iterations=2)

    text_mask = np.where(mask < text_mask, mask, text_mask)

    """ ocr """
    data = pytesseract.image_to_data(255 - text_mask, lang=language, config=config, output_type=output_type)

    """ bounding box """
    bbox_keys = 'left', 'top', 'width', 'height', 'conf'

    for x, y, w, h, c in zip(*map(data.get, bbox_keys)):
        if c != '-1':
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Image.fromarray(image).show()

    print(''.join(data['text']))
    print(sorted(set(data['block_num'])))
    break
