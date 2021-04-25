""" experiment for line offset """

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

import os

image1_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/sources/sources1'
image2_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/sources/sources2'
output_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/test1'

name_filepath = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/psd/name/name2.png'
name_mask_filepath = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/psd/name/name2_mask.png'
proxy_filepath = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/psd/placeholder/name1.png'
proxy_mask_filepath = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/psd/placeholder/name1_mask.png'

canny_threshold1 = 100
canny_threshold2 = 100

matching_method = cv2.TM_CCOEFF_NORMED
matching_threshold = 0.5

inpainting_method = cv2.INPAINT_TELEA
inpainting_radius = 3

if not os.path.exists(output_directory):
    os.makedirs(output_directory, exist_ok=True)

""" read """
name = cv2.imread(name_filepath)
name_mask = cv2.imread(name_mask_filepath, cv2.IMREAD_GRAYSCALE)
proxy = cv2.imread(proxy_filepath)
proxy_mask = cv2.imread(proxy_mask_filepath, cv2.IMREAD_GRAYSCALE)

name_weights = cv2.cvtColor(name_mask, cv2.COLOR_GRAY2BGR) / 255

proxy_grey = cv2.cvtColor(proxy, cv2.COLOR_BGR2GRAY)
proxy_edge = cv2.Canny(proxy_grey, canny_threshold1, canny_threshold2)

for filename1, filename2 in zip(os.listdir(image1_directory), os.listdir(image2_directory)):
    image1_filepath = os.path.join(image1_directory, filename1)
    image2_filepath = os.path.join(image2_directory, filename2)

    image1 = cv2.imread(image1_filepath)
    image2 = cv2.imread(image2_filepath)

    black, white = 16, 32
    diff = cv2.absdiff(image1, image2).max(axis=2)
    diff = ((diff.clip(black, white) - black) * (255 / (white - black))).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)

    image = image1
    text_mask = diff
    text_weights = cv2.cvtColor(text_mask, cv2.COLOR_GRAY2BGR) / 255

    """ match """
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_edge = cv2.Canny(image_grey, canny_threshold1, canny_threshold2)

    corr = cv2.matchTemplate(image_edge, proxy_edge, matching_method)
    matches = np.where(corr >= matching_threshold)
    points = list(zip(*matches[::-1]))

    if not points:
        continue

    """ merge """
    clustering = DBSCAN(eps=min(proxy.shape[:2]) * 0.5, min_samples=1)
    clustering.fit(points)

    labels = [label for label in sorted(set(clustering.labels_)) if not label < 0]
    clusters = [np.compress(clustering.labels_ == label, points, axis=0) for label in labels]
    centroids = [c.mean(axis=0).round().astype(int).tolist() for c in clusters]

    """ inpaint """
    for x, y in centroids:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        h, w = proxy.shape[:2]
        height = np.where(text_mask[y:, x: x + w].any(axis=1))[0].max() + 1

        line = image[y: y + height, x: x + w].copy()
        line_mask = text_mask[y: y + height, x: x + w].copy()

        line_mask[: h // 2] = cv2.min(line_mask[:h // 2], proxy_mask[: h // 2])
        mask[y: y + height, x: x + w] = line_mask
        image = cv2.inpaint(image, mask, inpainting_radius, inpainting_method)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilation = cv2.dilate(proxy_mask, kernel, iterations=1)
        line_mask[:h] = cv2.min(line_mask[:h], 255 - dilation)

        y_diff = name.shape[0] - proxy.shape[0]
        line_weights = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2BGR) / 255
        i = image[y + y_diff: y + y_diff + height, x: x + w]
        i[:] = line * line_weights + i * (1.0 - line_weights)

        h, w = name.shape[:2]
        i = image[y: y + h, x: x + w]
        i[:] = name * name_weights + i * (1.0 - name_weights)

    """ write """
    output_filepath = os.path.join(output_directory, filename1.replace('.jpg', '.png'))
    cv2.imwrite(output_filepath, image)
