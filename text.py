import cv2
import numpy as np
from sklearn.cluster import DBSCAN


def match_template(image, template, threshold=0.8, method=cv2.TM_CCOEFF_NORMED):
    corr = cv2.matchTemplate(image, template, method)
    matches = np.where(corr >= threshold)[::-1]

    return list(zip(*matches))


def merge(points, distance):
    clustering = DBSCAN(eps=distance, min_samples=1)
    clustering.fit(points)

    labels = clustering.labels_
    n_clusters = labels.max() + 1
    clusters = [np.compress(labels == i, points, axis=0) for i in range(n_clusters)]

    return [c.mean(axis=0).round().astype(int).tolist() for c in clusters]


def generate_mask(image, template, threshold=0.8, method=cv2.TM_CCOEFF_NORMED):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = match_template(image, template, threshold=threshold, method=method)
    h, w = template.shape[:2]

    for x, y in points:
        m = mask[y: y + h, x: x + w]
        np.putmask(m, template > m, template)

    return mask


def generate_bounding_box(image, template, threshold=0.8, method=cv2.TM_CCOEFF_NORMED):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = match_template(image, template, threshold=threshold, method=method)
    h, w = template.shape[:2]

    for x, y in points:
        mask[y: y + h, x: x + w] = 255

    return mask


def inpaint(image, mask, radius=1, method=cv2.INPAINT_TELEA):
    return cv2.inpaint(image, mask, radius, method)
