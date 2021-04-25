""" experiment for ocr """

from PIL import Image
import pytesseract

import os

image_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/natsuki'

language = 'jpn_vert'

for filename in os.listdir(image_directory):
    image_filepath = os.path.join(image_directory, filename)
    image = Image.open(image_filepath)

    text = pytesseract.image_to_string(image, lang=language)
    break
