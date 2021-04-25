""" grid search: natsuki2 """

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

import json
import os
import sys
from itertools import product

image1_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki2/sources1'
image2_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki2/sources2'
text_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki2/text'
output_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki2/text1'

block_keys = 'color', 'coordinate', 'font', 'size', 'spacing', 'text'
encoding = 'utf-8'
indent = 2
sort_keys = True

anchor = 'st'
border_color = 255, 255, 255
direction = 'ttb'
language = 'ja'
stroke_width = 5

matching_method = cv2.TM_CCOEFF_NORMED
correlation_threshold = 0.75

coord_param = 300
size_params = sorted(range(-10, 11), key=abs)
spacing_params = [0]

if not os.path.exists(output_directory):
    os.makedirs(output_directory, exist_ok=True)

filenames = list(zip(os.listdir(image1_directory), os.listdir(image2_directory), os.listdir(text_directory)))

for image1_filename, image2_filename, text_filename in filenames:
    image1_filepath = os.path.join(image1_directory, image1_filename)
    image2_filepath = os.path.join(image2_directory, image2_filename)
    text_filepath = os.path.join(text_directory, text_filename)

    image1 = Image.open(image1_filepath)
    array1 = np.asarray(image1)
    array1 = cv2.cvtColor(array1, cv2.COLOR_RGB2BGR)

    with open(text_filepath, encoding=encoding) as f:
        text_data = json.load(f)

    """ iterate over blocks """
    block_params = []

    for block in text_data['block']:
        color, coord, font, size, spacing, texts = map(block.get, block_keys)
        color = tuple(color)
        x, y = coord

        params = None
        correlation = -sys.float_info.max

        for dw, ds in product(size_params, spacing_params):
            """ draw template """
            pw = size + dw
            ps = spacing + ds

            w = len(texts) * pw + (len(texts) - 1) * ps
            h = max(len(t) for t in texts) * pw

            template = Image.new('RGB', (w, h), (255, 255, 255))
            draw = ImageDraw.Draw(template)

            px, py = w, 0
            f = ImageFont.truetype(font, pw)

            for text in texts:
                draw.text((px - pw // 2, py), text.replace('裕樹', '太一'), fill=color, font=f, anchor=anchor,
                          direction=direction, language=language, stroke_width=stroke_width, stroke_fill=border_color)

                px -= pw + ps

            """ calculate correlation """
            px1, px2 = np.clip([x - w - coord_param, x + coord_param], 0, array1.shape[1])
            py1, py2 = np.clip([y - coord_param, y + h + coord_param], 0, array1.shape[0])
            arr1 = array1[py1:py2, px1:px2]

            array2 = np.asarray(template)
            array2 = cv2.cvtColor(array2, cv2.COLOR_RGB2BGR)

            correlations = cv2.matchTemplate(arr1, array2, matching_method)
            _, corr, _, (px, py) = cv2.minMaxLoc(correlations)
            px += px1
            py += py1

            if corr > correlation:
                params = {'coordinate': (int(px) + w, int(py)), 'size': pw, 'spacing': ps}

                if corr >= correlation_threshold:
                    break
                else:
                    correlation = corr

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
            draw.text((x - size // 2, y), text, fill=color, font=font, anchor=anchor, direction=direction,
                      language=language, stroke_width=stroke_width, stroke_fill=border_color)

            x -= size + spacing

    image_filepath = os.path.join(output_directory, image1_filename.replace('.jpg', '.png'))
    ImageChops.difference(image1, image2).save(image_filepath)
