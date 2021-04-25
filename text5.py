""" latest for manga """

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

import os
import shutil

input_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki2/sources'
output_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki2/test-threshold=0.7'

name_filepath = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki2/psd/name/name.png'
proxy_filepath = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki2/psd/placeholder/placeholder.png'

canny_threshold1 = 100
canny_threshold2 = 100

matching_method = cv2.TM_CCOEFF_NORMED
matching_threshold = 0.7

if not os.path.exists(output_directory):
    os.makedirs(output_directory, exist_ok=True)

""" read """
name = cv2.imread(name_filepath)
proxy = cv2.imread(proxy_filepath)

for filename in os.listdir(input_directory):
    input_filepath = os.path.join(input_directory, filename)
    image = cv2.imread(input_filepath)

    """ match """
    corr = cv2.matchTemplate(image, proxy, matching_method)
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

    """ composite """
    h, w = name.shape[:2]

    for x, y in centroids:
        image[y: y + h, x: x + w] = name

    """ write """
    output_filepath = os.path.join(output_directory, filename.replace('.jpg', '.png'))
    cv2.imwrite(output_filepath, image)
