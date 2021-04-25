""" extract text data: natsuki2 """

import cv2
import numpy as np
import pytesseract
from PIL import Image
from sklearn.cluster import DBSCAN

import json
import os

import ocr

image_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki2/sources1'
mask_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki2/balloon'
text_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki2/text1'

erosion = 1
connectivity = 8

language = 'jpn_vert'
config = '--psm 3'
output_type = 'data.frame'

font = 'meiryo.ttc'
size = 39
spacing = 10
fix_scale = 0.65

encoding = 'utf-8'
indent = 2
sort_keys = True

if not os.path.exists(text_directory):
    os.makedirs(text_directory, exist_ok=True)

filenames = list(zip(os.listdir(image_directory), os.listdir(mask_directory)))

for image_filename, mask_filename in filenames:
    image_filepath = os.path.join(image_directory, image_filename)
    image = Image.open(image_filepath)
    image = np.asarray(image)

    mask_filepath = os.path.join(mask_directory, mask_filename)
    mask = Image.open(mask_filepath)
    mask = np.asarray(mask)

    """ mask """
    if not mask.any():
        text_data = {'block': [], 'size': mask.T.shape}

        """ save """
        text_filename = image_filename.replace('.jpg', '.json')
        text_filepath = os.path.join(text_directory, text_filename)

        with open(text_filepath, encoding=encoding, mode='w') as f:
            json.dump(text_data, f, ensure_ascii=False, indent=indent, sort_keys=sort_keys)

        continue

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kernel, iterations=erosion)

    """ iterate over balloons """
    text_data = []
    points = []

    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
    labels = labels[:, :, np.newaxis]

    for i, (x, y, w, h, a) in enumerate(stats[1:], 1):
        if a < 1000:
            continue

        points.append((int(x + w // 2), int(y + h // 2)))
        balloon = np.where(labels == i, image, 255)[y: y + h, x: x + w]

        """ ocr """
        ocr_data = pytesseract.image_to_data(balloon, lang=language, config=config, output_type=output_type)
        block_data = ocr.get_text_data(ocr_data, (w, h))

        if block_data:
            block = block_data[0]
            block['coordinate'] = (int(x) + block['coordinate'][0], int(y) + block['coordinate'][1])
            text_data.append(block)
        else:
            text_data.append({
                'coordinate': (int(x + w // 2), int(y + h // 2)),
                'size': size,
                'spacing': spacing,
                'text': [],
            })

    """ update """
    block_data = {'font': font, 'color': (0, 0, 0)}

    for block in text_data:
        block.update(block_data)

        for i in range(len(block['text'])):
            text = block['text'][i]

            for a, b in [('!', '！'), ('1', '！'), ('i', '！'), ('j', '！'), ('?', '？'), ('2', '？'), ('7', '？'),
                         ('天', '♥'), ('馬', '♥'), ('e', '♥'), ('v', '♥'), ('w', '♥'),
                         (':', '…'), ('-', 'ー'), ('~', '～'), ('・', ''), ('(', '（'), (')', '）'), ('[', '「'), (']', '」'),
                         ('！！', '！'), ('？？', '？'), ('♥♥', '♥'), ('……', '…'),
                         ('ば', 'ぱ'), ('び', 'ぴ'), ('ぶ', 'ぷ'), ('べ', 'ぺ'), ('ぼ', 'ぽ'),
                         ('バ', 'パ'), ('ビ', 'ピ'), ('ブ', 'プ'), ('ベ', 'ペ'), ('ボ', 'ポ'),
                         ('おっはい', 'おっぱい'), ('オッハイ', 'オッパイ'),
                         ('ちんほ', 'ちんぽ'), ('チンホ', 'チンポ'), ('へニス', 'ペニス'), ('ヘニス', 'ペニス'),
                         ('いっはい', 'いっぱい'), ('はあ', 'はぁ')]:
                text = text.replace(a, b)

            block['text'][i] = text

        if block['spacing'] is None:
            block['spacing'] = spacing

        if (spacing - 5) <= block['spacing'] <= (spacing + 5):
            block['spacing'] = spacing

        if (39 - 5) <= block['size'] <= (39 + 5):
            block['size'] = 39

        if block['text'] and block['text'][0][0] in ('「', '（'):
            x, y = block['coordinate']
            block['coordinate'] = x, y - int(np.round(block['size'] * fix_scale))

    """ sort """
    y = np.array(points)[:, 1]
    eps = mask.shape[0] // 5
    clustering = DBSCAN(eps=eps, min_samples=1)
    clustering.fit(y.reshape(-1, 1))

    labels = [label for label in sorted(set(clustering.labels_)) if not label < 0]
    clusters = [np.compress(clustering.labels_ == label, y, axis=0) for label in labels]
    centroids = [c.mean() for c in clusters]

    clusters = [np.compress(clustering.labels_ == label, range(len(points)), axis=0) for label in labels]
    clusters = sorted(zip(clusters, centroids), key=lambda c: c[1])
    clusters = [[(text_data[i], points[i][0]) for i in c] for c in list(zip(*clusters))[0]]
    clusters = [sorted(c, key=lambda d: d[1], reverse=True) for c in clusters]

    text_data = list(zip(*sum(clusters, [])))[0]

    """ save """
    text_data = {'block': text_data, 'size': mask.T.shape}

    text_filename = image_filename.replace('.jpg', '.json')
    text_filepath = os.path.join(text_directory, text_filename)

    with open(text_filepath, encoding=encoding, mode='w') as f:
        json.dump(text_data, f, ensure_ascii=False, indent=indent, sort_keys=sort_keys)
