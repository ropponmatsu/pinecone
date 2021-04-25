""" experiment for line extraction """

import cv2
import numpy as np

import os

input_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/test-placeholder=(1,2,3)'
output_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/test_line'

bgr = 212, 0, 255
color = np.array(bgr, dtype=np.uint8)

if not os.path.exists(output_directory):
    os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(input_directory):
    input_filepath = os.path.join(input_directory, filename)
    image = cv2.imread(input_filepath)

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

    """ border mask """
    border_mask = text_mask.copy()

    for _ in range(10):
        border_mask = (border_mask.astype(int) + cv2.GaussianBlur(border_mask, (3, 3), 1)).clip(0, 255).astype(np.uint8)

    """ write """
    output_filepath = os.path.join(output_directory, filename.replace('.jpg', '.png'))
    cv2.imwrite(output_filepath, border_mask)
    break
