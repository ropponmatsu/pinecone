""" latest """

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

import os
import shutil

input_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/sources/sources1'
output_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/test'

name_filepath = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/psd/name/name1.png'
name_mask_filepath = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/psd/name/name1_mask.png'
proxy_filepath = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/psd/placeholder/placeholder1-2.png'
proxy_mask_filepath = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/psd/placeholder/placeholder1_mask.png'

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
name_weights = cv2.cvtColor(name_mask, cv2.COLOR_GRAY2BGR) / 255

proxy = cv2.imread(proxy_filepath)
proxy_mask = cv2.imread(proxy_mask_filepath, cv2.IMREAD_GRAYSCALE)

proxy_grey = cv2.cvtColor(proxy, cv2.COLOR_BGR2GRAY)
proxy_edge = cv2.Canny(proxy_grey, canny_threshold1, canny_threshold2)

for filename in os.listdir(input_directory):
    input_filepath = os.path.join(input_directory, filename)
    image = cv2.imread(input_filepath)

    """ match """
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_edge = cv2.Canny(image_grey, canny_threshold1, canny_threshold2)

    corr = cv2.matchTemplate(image_edge, proxy_edge, matching_method)
    matches = np.where(corr >= matching_threshold)
    points = list(zip(*matches[::-1]))

    if not points:
        output_filepath = os.path.join(output_directory, filename)
        shutil.copy(input_filepath, output_filepath)
        continue

    """ merge """
    eps = min(proxy.shape[:2]) * 0.5
    clustering = DBSCAN(eps=eps, min_samples=1)
    clustering.fit(points)

    labels = [label for label in sorted(set(clustering.labels_)) if not label < 0]
    clusters = [np.compress(clustering.labels_ == label, points, axis=0) for label in labels]
    centroids = [c.mean(axis=0).round().astype(int).tolist() for c in clusters]

    """ inpaint """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    h, w = proxy.shape[:2]

    for x, y in centroids:
        mask[y: y + h, x: x + w] = proxy_mask

    image = cv2.inpaint(image, mask, inpainting_radius, inpainting_method)

    """ composite """
    h, w = name.shape[:2]

    for x, y in centroids:
        img = image[y: y + h, x: x + w]
        comp = name * name_weights + img * (1.0 - name_weights)
        img[:] = comp.round().astype(np.uint8)

    """ write """
    output_filepath = os.path.join(output_directory, filename.replace('.jpg', '.png'))
    cv2.imwrite(output_filepath, image)
