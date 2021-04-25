""" Initial script using contours """

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

import os
import shutil

input_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/sources'
output_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/test1'
template_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/psd'

template1_filename = 'name1.png'
template1_mask_filename = 'name1_mask.png'
template2_filename = 'name2.png'
template2_mask_filename = 'name2_mask.png'

canny_threshold1 = 100
canny_threshold2 = 100

matching_method = cv2.TM_CCOEFF_NORMED
matching_threshold = 0.5

inpainting_method = cv2.INPAINT_TELEA
inpainting_radius = 1

""" read """
template1_filepath = os.path.join(template_directory, template1_filename)
template1_mask_filepath = os.path.join(template_directory, template1_mask_filename)
template2_filepath = os.path.join(template_directory, template2_filename)
template2_mask_filepath = os.path.join(template_directory, template2_mask_filename)

template1 = cv2.imread(template1_filepath)
template1_mask = cv2.imread(template1_mask_filepath, cv2.IMREAD_GRAYSCALE)
template2 = cv2.imread(template2_filepath)
template2_mask = cv2.imread(template2_mask_filepath, cv2.IMREAD_GRAYSCALE)

template1_grey = cv2.cvtColor(template1, cv2.COLOR_BGR2GRAY)
template1_edge = cv2.Canny(template1_grey, canny_threshold1, canny_threshold2)

if not os.path.exists(output_directory):
    os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(input_directory):
    input_filepath = os.path.join(input_directory, filename)
    image = cv2.imread(input_filepath)

    """ match """
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_edge = cv2.Canny(image_grey, canny_threshold1, canny_threshold2)

    corr = cv2.matchTemplate(image_edge, template1_edge, matching_method)
    matches = np.where(corr >= matching_threshold)

    if (corr >= matching_threshold).sum() == 0:
        output_filepath = os.path.join(output_directory, filename)
        shutil.copy(input_filepath, output_filepath)
        continue

    """ inpaint """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    h, w = template1.shape[:2]

    for y, x in zip(*matches):
        m = mask[y: y + h, x: x + w]
        np.putmask(m, template1_mask > m, template1_mask)

    filled = cv2.inpaint(image, mask, inpainting_radius, inpainting_method)

    """ merge by contour """
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for y, x in zip(*matches):
        mask[y: y + h, x: x + w] = 255

    _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    centers = [c.squeeze().mean(axis=0) for c in contours]
    """

    """ merge by DBSCAN """
    points = list(zip(*matches[::-1]))
    clustering = DBSCAN(eps=min(w, h) * 0.5, min_samples=1)
    clustering.fit(points)

    labels = clustering.labels_
    n_clusters = labels.max() + 1
    clusters = [np.compress(labels == i, points, axis=0) for i in range(n_clusters)]

    centers = [c.mean(axis=0).round().astype(int).tolist() for c in clusters]

    """ composite """
    weights = cv2.cvtColor(template2_mask, cv2.COLOR_GRAY2BGR) / 255

    for x, y in centers:
        # x = np.round(x - w * 0.5).astype(int)
        # y = np.round(y - h * 0.5).astype(int)
        i = filled[y: y + h, x: x + w]
        i[:] = template2 * weights + i * (1.0 - weights)

    """ write """
    output_filepath = os.path.join(output_directory, filename.replace('.jpg', '.png'))
    cv2.imwrite(output_filepath, filled)
