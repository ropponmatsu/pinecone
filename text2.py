""" experiment for multiple proxies """

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

import os

input_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/sources'
output_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/test1'

image1_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/sources/sources1'
image2_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/sources/sources2'

name_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/psd/name'
proxy_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/psd/placeholder'
mask_suffix = '_mask'

canny_threshold1 = 100
canny_threshold2 = 100

matching_method = cv2.TM_CCOEFF_NORMED
matching_threshold = 0.5

inpainting_method = cv2.INPAINT_TELEA
inpainting_radius = 1

if not os.path.exists(output_directory):
    os.makedirs(output_directory, exist_ok=True)

""" read """
name_filenames = [n for n in os.listdir(name_directory) if mask_suffix not in n]
mask_filenames = [n + mask_suffix + ext for n, ext in map(os.path.splitext, name_filenames)]

name_paths = [os.path.join(name_directory, n) for n in name_filenames]
name_mask_paths = [os.path.join(name_directory, n) for n in mask_filenames]
proxy_paths = [os.path.join(proxy_directory, n) for n in name_filenames]
proxy_mask_paths = [os.path.join(proxy_directory, n) for n in mask_filenames]

names = [cv2.imread(path) for path in name_paths]
names_mask = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in name_mask_paths]
proxies = [cv2.imread(path) for path in proxy_paths]
proxies_mask = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in proxy_mask_paths]

proxies_grey = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in proxies]
proxies_edge = [cv2.Canny(img, canny_threshold1, canny_threshold2) for img in proxies_grey]

for filename1, filename2 in list(zip(os.listdir(image1_directory), os.listdir(image2_directory)))[7:8]:
    image1_filepath = os.path.join(input_directory, filename1)
    image2_filepath = os.path.join(input_directory, filename2)

    image1 = cv2.imread(image1_filepath)
    image2 = cv2.imread(image2_filepath)

    black, white = 16, 64
    diff = cv2.absdiff(image1, image2).max(axis=2)
    diff = ((diff.clip(black, white) - black) * (255 / (white - black))).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)

    image = image1
    text_mask = diff
    text_weights = cv2.cvtColor(text_mask, cv2.COLOR_GRAY2BGR) / 255

    for name, name_mask, proxy, proxy_mask, proxy_edge in zip(names, names_mask, proxies, proxies_mask, proxies_edge):
        """ match """
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_edge = cv2.Canny(image_grey, canny_threshold1, canny_threshold2)

        corr = cv2.matchTemplate(image_edge, proxy_edge, matching_method)
        matches = np.where(corr >= matching_threshold)
        points = list(zip(*matches[::-1]))

        if not points:
            continue

        """ inpaint """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        h, w = proxy.shape[:2]

        for x, y in points:
            m = mask[y: y + h, x: x + w]
            np.putmask(m, proxy_mask > m, proxy_mask)

        image = cv2.inpaint(image, mask, inpainting_radius, inpainting_method)

        """ merge """
        clustering = DBSCAN(eps=min(w, h) * 0.5, min_samples=1)
        clustering.fit(points)

        labels = [label for label in sorted(set(clustering.labels_)) if not label < 0]
        clusters = [np.compress(clustering.labels_ == label, points, axis=0) for label in labels]
        centroids = [c.mean(axis=0).round().astype(int).tolist() for c in clusters]

    """ write """
    output_filepath = os.path.join(output_directory, filename1.replace('.jpg', '.png'))
    cv2.imwrite(output_filepath, image)
