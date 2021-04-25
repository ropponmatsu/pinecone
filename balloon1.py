""" check if connected components are balloons """

import cv2
import numpy as np
from PIL import Image

path = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki2/1573156-16.jpg'
image = np.asarray(Image.open(path))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 252, 255, cv2.THRESH_BINARY)
n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

mask = np.zeros_like(binary)
start = 1
margin = 1

for i, (x, y, w, h, a) in enumerate(stats[start:], start):
    if a < 5000:
        continue

    if w > h:
        continue

    seg = np.zeros((h + margin * 2, w + margin * 2), dtype=np.uint8)
    seg[margin: -margin, margin: -margin] = np.where(labels == i, binary, 0)[y: y + h, x: x + w]
    n, l, s, c = cv2.connectedComponentsWithStats(255 - seg, connectivity=8)
    text = np.zeros_like(seg)

    for ii, (xx, yy, ww, hh, aa) in enumerate(s[1:], 1):
        if aa > 2000:
            continue

        text[l == ii] = 255

    if not text.any():
        continue

    indices = np.where(np.any(text, axis=0))[0]
    ww = indices.max() - indices.min()
    indices = np.where(np.any(text, axis=1))[0]
    hh = indices.max() - indices.min()

    if ww > hh:
        continue

    seg = np.where(seg > text, seg, text)[margin: -margin, margin: -margin]
    m = mask[y: y + h, x: x + w]
    m[:] = np.where(m > seg, m, seg)

Image.fromarray(mask).show()
