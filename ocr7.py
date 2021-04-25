""" extract text data """

import numpy as np
import pytesseract
from PIL import Image, ImageFilter

import json
import os

import color
import ocr

image1_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/natsuki'
image2_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/sources/sources2'
text_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/text1'

text_color = 255, 0, 212
border_color = 255, 255, 255
radius = 20

text_threshold = 254
diff_threshold = 8
dilation_size = 5

language = 'jpn_vert'
config = '--psm 3'
output_type = 'data.frame'

font = 'meiryo.ttc'
spacing = 10
fix_scale = 0.65

encoding = 'utf-8'
indent = 2
sort_keys = True

if not os.path.exists(text_directory):
    os.makedirs(text_directory, exist_ok=True)

for filename1, filename2 in zip(os.listdir(image1_directory), os.listdir(image2_directory)):
    image1_filepath = os.path.join(image1_directory, filename1)
    image2_filepath = os.path.join(image2_directory, filename2)
    image1 = Image.open(image1_filepath)
    image2 = Image.open(image2_filepath)
    array1 = np.asarray(image1)
    array2 = np.asarray(image2)

    """ mask """
    mask = color.reset_diagonal(array1, border_color, text_color)
    mask = color.discard_cylindrical(mask, radius)
    mask = color.project(mask)
    mask = Image.fromarray(255 - mask).convert('RGB')

    """ ocr """
    ocr_data = pytesseract.image_to_data(mask, lang=language, config=config, output_type='data.frame')
    text_data1 = ocr.get_text_data(ocr_data, image1.size)

    """ update """
    block_data = {'font': font, 'color': text_color}

    for block in text_data1:
        block.update(block_data)

        for i in range(len(block['text'])):
            text = block['text'][i]

            for a, b in [('!', '！'), ('1', '！'), ('i', '！'), ('j', '！'), ('?', '？'), ('2', '？'), ('7', '？'),
                         ('天', '♥'), ('馬', '♥'), ('e', '♥'), ('v', '♥'), ('w', '♥'),
                         (':', '…'), ('-', 'ー'), ('~', '～'), ('・', ''), ('(', '（'), (')', '）'), ('[', '「'), (']', '」'),
                         ('！！', '！'), ('？？', '？'), ('♥♥', '♥'), ('……', '…'),
                         ('ば', 'ぱ'), ('び', 'ぴ'), ('ぶ', 'ぷ'), ('べ', 'ぺ'), ('ぼ', 'ぽ'),
                         ('バ', 'パ'), ('ビ', 'ピ'), ('ブ', 'プ'), ('ベ', 'ペ'), ('ボ', 'ポ'),
                         ('おっはい', 'おっぱい'), ('オッハイ', 'オッパイ'),
                         ('ちんほ', 'ちんぽ'), ('チンホ', 'チンポ'), ('へニス', 'ペニス'), ('ヘニス', 'ペニス'),
                         ('いっはい', 'いっぱい'), ('はあ', 'はぁ')]:
                text = text.replace(a, b)

            block['text'][i] = text

        if block['spacing'] is None:
            block['spacing'] = spacing

        if (spacing - 5) <= block['spacing'] <= (spacing + 5):
            block['spacing'] = spacing

        if (39 - 5) <= block['size'] <= (39 + 5):
            block['size'] = 39

        if block['text'][0][0] in ('「', '（'):
            x, y = block['coordinate']
            block['coordinate'] = x, y - int(np.round(block['size'] * fix_scale))

    """ mask """
    mask = 255 - color.project(array1)
    mask = np.where(mask > text_threshold, mask, 0)

    diff = np.abs(array1.astype(int) - array2.astype(int))
    diff = color.project(diff)
    mask = np.where(diff > diff_threshold, mask, 0)

    mask = Image.fromarray(mask)
    mask = mask.filter(ImageFilter.MaxFilter(dilation_size))

    mask = (255 - array1) * (np.asarray(mask)[:, :, np.newaxis] / 255)
    mask = Image.fromarray(mask.astype(np.uint8)).convert('RGB')

    """ ocr """
    ocr_data = pytesseract.image_to_data(mask, lang=language, config=config, output_type=output_type)
    text_data2 = ocr.get_text_data(ocr_data, image1.size)

    """ update """
    block_data = {'font': font, 'color': (0, 0, 0)}

    for block in text_data2:
        block.update(block_data)

        for i in range(len(block['text'])):
            text = block['text'][i]

            for a, b in [('!', '！'), ('1', '！'), ('i', '！'), ('j', '！'), ('?', '？'), ('2', '？'), ('7', '？'),
                         ('天', '♥'), ('馬', '♥'), ('e', '♥'), ('v', '♥'), ('w', '♥'),
                         (':', '…'), ('-', 'ー'), ('~', '～'), ('・', ''), ('(', '（'), (')', '）'), ('[', '「'), (']', '」'),
                         ('！！', '！'), ('？？', '？'), ('♥♥', '♥'), ('……', '…'),
                         ('ば', 'ぱ'), ('び', 'ぴ'), ('ぶ', 'ぷ'), ('べ', 'ぺ'), ('ぼ', 'ぽ'),
                         ('バ', 'パ'), ('ビ', 'ピ'), ('ブ', 'プ'), ('ベ', 'ペ'), ('ボ', 'ポ'),
                         ('おっはい', 'おっぱい'), ('オッハイ', 'オッパイ'),
                         ('ちんほ', 'ちんぽ'), ('チンホ', 'チンポ'), ('へニス', 'ペニス'), ('ヘニス', 'ペニス'),
                         ('いっはい', 'いっぱい'), ('はあ', 'はぁ')]:
                text = text.replace(a, b)

            block['text'][i] = text

        if block['spacing'] is None:
            block['spacing'] = spacing

        if (spacing - 5) <= block['spacing'] <= (spacing + 5):
            block['spacing'] = spacing

        if (39 - 5) <= block['size'] <= (39 + 5):
            block['size'] = 39

        if block['text'][0][0] in ('「', '（'):
            x, y = block['coordinate']
            block['coordinate'] = x, y - int(np.round(block['size'] * fix_scale))

    """ merge """
    text_data = text_data1 + text_data2
    text_data = sorted(text_data, key=lambda b: b['coordinate'][0], reverse=True)
    text_data = {'block': text_data, 'size': image1.size}

    """ save """
    text_filename = filename1.replace('.jpg', '.json').replace('.png', '.json')
    text_filepath = os.path.join(text_directory, text_filename)

    with open(text_filepath, encoding=encoding, mode='w') as f:
        json.dump(text_data, f, ensure_ascii=False, indent=indent, sort_keys=sort_keys)
