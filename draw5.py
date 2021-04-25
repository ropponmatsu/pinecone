""" composite text data: natsuki2 """

from PIL import Image, ImageDraw, ImageFont

import json
import os

image_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki2/sources2'
text_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki2/text'

encoding = 'utf-8'
block_keys = 'color', 'coordinate', 'font', 'size', 'spacing', 'text'

anchor = 'st'
border_color = 255, 255, 255
direction = 'ttb'
language = 'ja'
stroke_width = 5

filenames = list(zip(os.listdir(image_directory), os.listdir(text_directory)))

for image_filename, text_filename in filenames[77:]:
    image_filepath = os.path.join(image_directory, image_filename)
    text_filepath = os.path.join(text_directory, text_filename)

    image = Image.open(image_filepath)
    draw = ImageDraw.Draw(image)

    with open(text_filepath, encoding=encoding) as f:
        text_data = json.load(f)

    for block in text_data['block']:
        color, coord, font, size, spacing, texts = map(block.get, block_keys)

        color = tuple(color)
        x, y = coord
        font = ImageFont.truetype(font, size)

        for text in texts:
            draw.text((x - size // 2, y), text, fill=color, font=font, anchor=anchor, direction=direction,
                      language=language, stroke_width=stroke_width, stroke_fill=border_color)

            x -= size + spacing

    image.show()
    break
