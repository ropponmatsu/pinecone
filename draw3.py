""" grid search """

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

import json
import os
import sys
from itertools import product

image1_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/natsuki'
image2_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/sources/sources2'
text_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/text'
output_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/text1'

block_keys = 'color', 'coordinate', 'font', 'size', 'spacing', 'text'
encoding = 'utf-8'
indent = 2
sort_keys = True

anchor = 'rt'
border_color = 255, 255, 255
direction = 'ttb'
language = 'ja'
stroke_width = 5

coord_params = list(product(range(-5, 6), range(-5, 6)))
size_params = [0]
spacing_params = [0]

if not os.path.exists(output_directory):
    os.makedirs(output_directory, exist_ok=True)

filenames = list(zip(os.listdir(image1_directory), os.listdir(image2_directory), os.listdir(text_directory)))

for image1_filename, image2_filename, text_filename in filenames[80:]:
    image1_filepath = os.path.join(image1_directory, image1_filename)
    image2_filepath = os.path.join(image2_directory, image2_filename)
    text_filepath = os.path.join(text_directory, text_filename)

    image1 = Image.open(image1_filepath)
    array1 = np.asarray(image1, dtype=int)

    with open(text_filepath, encoding=encoding) as f:
        text_data = json.load(f)

    """ iterate over blocks """
    block_params = []

    for block in text_data['block']:
        color, coord, font, size, spacing, texts = map(block.get, block_keys)
        color = tuple(color)
        x, y = coord

        params = None
        error = sys.float_info.max

        """ grid search """
        for (dx, dy), dw, ds in product(coord_params, size_params, spacing_params):
            image2 = Image.open(image2_filepath)
            draw = ImageDraw.Draw(image2)

            px = x + dx
            py = y + dy
            pw = size + dw
            ps = spacing + ds
            f = ImageFont.truetype(font, pw)

            for text in texts:
                draw.text((px, py), text, fill=color, font=f, anchor=anchor, direction=direction,
                          language=language, stroke_width=stroke_width, stroke_fill=border_color)

                px -= pw + ps

            sad = np.abs(array1 - np.asarray(image2, dtype=int)).sum()

            if sad < error:
                params = {'coordinate': (x + dx, py), 'size': pw, 'spacing': ps}
                error = sad

        block_params.append(params)

    """ update blocks """
    for block, params in zip(text_data['block'], block_params):
        block['coordinate'] = params['coordinate']
        block['size'] = params['size']
        block['spacing'] = params['spacing']

    """ save text """
    text_filepath = os.path.join(output_directory, text_filename)

    with open(text_filepath, encoding=encoding, mode='w') as f:
        json.dump(text_data, f, ensure_ascii=False, indent=indent, sort_keys=sort_keys)

    """ save difference """
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

    image_filepath = os.path.join(output_directory, image1_filename.replace('.jpg', '.png'))
    ImageChops.difference(image1, image2).save(image_filepath)
