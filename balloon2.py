""" extract balloon with ocr on connected components """

import cv2
import numpy as np
from PIL import Image

import pytesseract

import os

input_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki2/sources'
output_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki2/balloon1'

binary_threshold = 127
connectivity = 8

balloon_area_threshold = 5000
text_area_threshold = 2000
text_conf_threshold = 0

language = 'jpn_vert'
config = '--psm 3'
output_type = 'data.frame'

if not os.path.exists(output_directory):
    os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(input_directory):
    image_filepath = os.path.join(input_directory, filename)
    image = Image.open(image_filepath)

    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY)
    mask = np.zeros_like(gray)

    """ iterate over balloon candidates """
    _, labels1, stats1, _ = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)

    for i, (x1, y1, w1, h1, a1) in enumerate(stats1[1:], 1):
        if a1 < balloon_area_threshold:
            continue

        bln_mask = np.zeros((h1 + 2, w1 + 2), dtype=np.uint8)
        bln_mask[1: -1, 1: -1] = np.where(labels1 == i, binary, 0)[y1: y1 + h1, x1: x1 + w1]
        text_mask = np.zeros_like(bln_mask)

        """ iterate over text candidates """
        _, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(255 - bln_mask, connectivity=connectivity)

        for j, (x2, y2, w2, h2, a2) in enumerate(stats2[1:], 1):
            if a2 > text_area_threshold:
                continue

            if (h2, w2) == bln_mask.shape:
                continue

            text_mask[labels2 == j] = 255

        """ extract text """
        bln_mask = np.where(bln_mask > text_mask, bln_mask, text_mask)[1: -1, 1: -1]
        text = np.where(bln_mask == 255, gray[y1: y1 + h1, x1: x1 + w1], 255)

        """ ocr """
        ocr_data = pytesseract.image_to_data(text, lang=language, config=config, output_type=output_type)
        text_data = ocr_data.loc[:, ['text', 'conf']].dropna(axis=0)
        confidences = [c for t, c in text_data.to_numpy() if str(t).strip() and c != -1]

        if not confidences or sum(confidences) / len(confidences) < text_conf_threshold:
            continue

        """ update """
        m = mask[y1: y1 + h1, x1: x1 + w1]
        m[:] = np.where(m > bln_mask, m, bln_mask)

    output_filepath = os.path.join(output_directory, filename.replace('.jpg', '.png'))
    Image.fromarray(mask).show()
