""" experiment for ocr with absolute difference """

import cv2
import numpy as np
import pytesseract
from PIL import Image

import os

image1_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/natsuki'
image2_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/sources/sources2'

language = 'jpn_vert'
config = '--psm 3'
output_type = 'dict'
bbox_keys = 'left', 'top', 'width', 'height', 'conf'

filenames = list(zip(os.listdir(image1_directory), os.listdir(image2_directory)))

for filename1, filename2 in filenames[7:]:
    image1_filepath = os.path.join(image1_directory, filename1)
    image2_filepath = os.path.join(image2_directory, filename2)

    image1 = cv2.imread(image1_filepath)
    image2 = cv2.imread(image2_filepath)

    mask = cv2.absdiff(image1, image2).max(axis=2)
    mask = 255 - mask

    black, white = 0, 255
    mask = ((mask.clip(black, white) - black) * (255 / (white - black))).astype(np.uint8)

    """ ocr """
    data = pytesseract.image_to_data(mask, lang=language, config=config, output_type=output_type)

    """ bounding box """
    image = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    for x, y, w, h, c in zip(*map(data.get, bbox_keys)):
        if c != '-1':
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0))

    Image.fromarray(image).show()
    print(''.join(data['text']))

    break
