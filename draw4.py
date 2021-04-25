""" show difference """

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

import json
import os

image1_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/natsuki'
image2_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/sources/sources2'
text_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/text1'

block_keys = 'color', 'coordinate', 'font', 'size', 'spacing', 'text'
encoding = 'utf-8'

anchor = 'rt'
border_color = 255, 255, 255
direction = 'ttb'
language = 'ja'
stroke_width = 5

filenames = list(zip(os.listdir(image1_directory), os.listdir(image2_directory), os.listdir(text_directory)))

for image1_filename, image2_filename, text_filename in filenames[80:]:
    image1_filepath = os.path.join(image1_directory, image1_filename)
    image2_filepath = os.path.join(image2_directory, image2_filename)
    text_filepath = os.path.join(text_directory, text_filename)

    image1 = Image.open(image1_filepath)
    array1 = np.asarray(image1, dtype=int)

    with open(text_filepath, encoding=encoding) as f:
        text_data = json.load(f)

    image2 = Image.open(image2_filepath)
    draw = ImageDraw.Draw(image2)

    for block in text_data['block']:
        color, coord, font, size, spacing, texts = map(block.get, block_keys)

        color = tuple(color)
        x, y = coord
        font = ImageFont.truetype(font, size)

        for text in texts:
            draw.text((x, y), text, fill=color, font=font, anchor=anchor, direction=direction,
                      language=language, stroke_width=stroke_width, stroke_fill=border_color)

            x -= size + spacing

    ImageChops.difference(image1, image2).show()
    break
