""" decompose and compose 1 """

import numpy as np
import pytesseract
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

import os

import color

image1_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/natsuki'
image2_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/sources/sources2'

text_color = 255, 0, 212
border_color = 255, 255, 255
radius = 20

text_threshold = 254
diff_threshold = 8
dilation_size = 5

language = 'jpn_vert'
config = '--psm 3'
output_type = 'data.frame'
bbox_keys = 'left', 'top', 'width', 'height', 'text', 'conf'

font_filepath = 'meiryo.ttc'
font_size = 40
spacing = 10
direction = 'ttb'

filenames = list(zip(os.listdir(image1_directory), os.listdir(image2_directory)))

for filename1, filename2 in filenames[159:]:
    image1_filepath = os.path.join(image1_directory, filename1)
    image2_filepath = os.path.join(image2_directory, filename2)

    image1 = Image.open(image1_filepath)
    image2 = Image.open(image2_filepath)

    array1 = np.asarray(image1)
    array2 = np.asarray(image2)

    """ text mask """
    mask1 = color.reset_diagonal(array1, border_color, text_color)
    mask1 = color.discard_cylindrical(mask1, radius)
    mask1 = color.project(mask1)

    mask1 = Image.fromarray(255 - mask1).convert('RGB')

    """ ocr """
    data1 = pytesseract.image_to_data(mask1, lang=language, config=config, output_type=output_type)

    """ text mask """
    mask2 = 255 - color.project(array1)
    mask2 = np.where(mask2 > text_threshold, mask2, 0)

    diff = np.abs(array1.astype(int) - array2.astype(int))
    diff = color.project(diff)
    mask2 = np.where(diff > diff_threshold, mask2, 0)

    mask2 = Image.fromarray(mask2)
    mask2 = mask2.filter(ImageFilter.MaxFilter(dilation_size))

    mask2 = (255 - array1) * (np.asarray(mask2)[:, :, np.newaxis] / 255)
    mask2 = Image.fromarray(mask2.astype(np.uint8)).convert('RGB')

    """ ocr """
    data2 = pytesseract.image_to_data(mask2, lang=language, config=config, output_type=output_type)

    """ draw text """
    draw = ImageDraw.Draw(image2)
    font = ImageFont.truetype(font_filepath, font_size)

    blocks = data1.loc[:, 'block_num']

    for i in blocks.unique()[1:]:
        rx = (data1.loc[blocks == i, 'left'] + data1.loc[blocks == i, 'width']).max()
        ry = data1.loc[blocks == i, 'top'].min()

        texts = data1.loc[blocks == i, 'text']
        lines = data1.loc[blocks == i, 'line_num']

        for j in lines.unique()[1:]:
            lx = rx - font_size * j - spacing * (j - 1)
            ly = ry

            text = ''.join(texts.loc[lines == j].dropna())
            draw.text((lx, ly), text, fill=text_color, font=font, direction=direction, stroke_width=5,
                      stroke_fill=border_color)

    """ draw text """
    draw = ImageDraw.Draw(image2)
    font = ImageFont.truetype(font_filepath, font_size)

    blocks = data2.loc[:, 'block_num']

    for i in blocks.unique()[1:]:
        rx = (data2.loc[blocks == i, 'left'] + data2.loc[blocks == i, 'width']).max()
        ry = data2.loc[blocks == i, 'top'].min()

        texts = data2.loc[blocks == i, 'text']
        lines = data2.loc[blocks == i, 'line_num']

        for j in lines.unique()[1:]:
            lx = rx - font_size * j - spacing * (j - 1)
            ly = ry

            text = ''.join(texts.loc[lines == j].dropna())
            draw.text((lx, ly), text, fill=(0, 0, 0), font=font, direction=direction, stroke_width=5,
                      stroke_fill=border_color)

    """ show """
    ImageChops.difference(image1, image2).show()
    break
