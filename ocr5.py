""" decompose and compose 2 """

import numpy as np
import pytesseract
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from sklearn.cluster import DBSCAN

import os

import color

image1_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/natsuki'
image2_directory = 'D:/logs/E/unsorted/E-HentaiGalleries/unsorted/natsuki/sources/sources2'

text_color = 255, 0, 212
border_color = 255, 255, 255
radius = 20

text_threshold = 254
diff_threshold = 8
dilation_size = 5

language = 'jpn_vert'
config = '--psm 3'
output_type = 'data.frame'
bbox_keys = 'x', 'y', 'width', 'height', 'text', 'conf'

font_filepath = 'meiryo.ttc'
font_size = 40
spacing = 10
direction = 'ttb'

text1 = [
    [
        '「じゅるるる…♥ずっぽずっぽ…♥」'
    ],
    [
        'じゅるるうぅううううう…♥'
    ],
    [
        '「ふぅふぅ…♥」',
        '（裕樹のチンポ吸ってると脳がビリビリするぐらい感じる…♥',
        '早く裕樹の子種飲みたい…！）'
    ]
]

text2 = [
    [
        '「あ、あ、あ、…♥」'
    ],
    [
        '「おぉおお…♥」',
        '（睾丸ごと精子持ってかれちゃう…！）',
        '夏姫の掃除機の様なバキュームに俺の睾丸が縮み上がる'
    ]
]

filenames = list(zip(os.listdir(image1_directory), os.listdir(image2_directory)))

for filename1, filename2 in filenames[159:]:
    image1_filepath = os.path.join(image1_directory, filename1)
    image2_filepath = os.path.join(image2_directory, filename2)

    image1 = Image.open(image1_filepath)
    image2 = Image.open(image2_filepath)

    array1 = np.asarray(image1)
    array2 = np.asarray(image2)

    """ text mask """
    mask1 = color.reset_diagonal(array1, border_color, text_color)
    mask1 = color.discard_cylindrical(mask1, radius)
    mask1 = color.project(mask1)

    mask1 = Image.fromarray(255 - mask1).convert('RGB')

    """ ocr """
    data1 = pytesseract.image_to_data(mask1, lang=language, config=config, output_type=output_type)

    """ text mask """
    mask2 = 255 - color.project(array1)
    mask2 = np.where(mask2 > text_threshold, mask2, 0)

    diff = np.abs(array1.astype(int) - array2.astype(int))
    diff = color.project(diff)
    mask2 = np.where(diff > diff_threshold, mask2, 0)

    mask2 = Image.fromarray(mask2)
    mask2 = mask2.filter(ImageFilter.MaxFilter(dilation_size))

    mask2 = (255 - array1) * (np.asarray(mask2)[:, :, np.newaxis] / 255)
    mask2 = Image.fromarray(mask2.astype(np.uint8)).convert('RGB')

    """ ocr """
    data2 = pytesseract.image_to_data(mask2, lang=language, config=config, output_type=output_type)

    """ draw text """
    draw = ImageDraw.Draw(image2)
    font = ImageFont.truetype(font_filepath, font_size)

    blocks = data1.loc[:, 'block_num']

    for i, texts in zip(blocks.unique()[1:], text1):
        lines = data1.loc[blocks == i, 'line_num']

        lefts = data1.loc[blocks == i, 'left']
        tops = data1.loc[blocks == i, 'top']
        widths = data1.loc[blocks == i, 'width']
        heights = data1.loc[blocks == i, 'height']
        confidences = data1.loc[blocks == i, 'conf']

        for j, text in zip(lines.unique()[1:], texts):
            m = Image.new('L', image2.size, 0)
            d = ImageDraw.Draw(m)
            boxes = zip(
                lefts.loc[lines == j],
                tops.loc[lines == j],
                widths.loc[lines == j],
                heights.loc[lines == j],
                confidences.loc[lines == j],
            )

            for x, y, w, h, c in boxes:
                if c != '-1':
                    d.rectangle((x, y, x + w, y + h), fill=255)

            """ x """
            x = np.asarray(m).astype(bool)
            x = [np.where(row)[0] for row in x]
            x = np.array([row.max() for row in x if row.size]).reshape(-1, 1)

            clustering = DBSCAN(eps=1, min_samples=1)
            clustering.fit(x)

            labels = [label for label in sorted(set(clustering.labels_)) if not label < 0]
            n_clusters = [(clustering.labels_ == label).sum() for label in labels]
            label = max(zip(labels, n_clusters), key=lambda label: label[1])[0]

            x = x.compress(clustering.labels_ == label, axis=0).mean()
            x = x.round().astype(int)

            """ size """
            w = np.asarray(m).astype(bool).sum(axis=1)
            w = w.compress(w != 0).reshape(-1, 1)

            clustering = DBSCAN(eps=1, min_samples=1)
            clustering.fit(w)

            labels = [label for label in sorted(set(clustering.labels_)) if not label < 0]
            n_clusters = [(clustering.labels_ == label).sum() for label in labels]
            label = max(zip(labels, n_clusters), key=lambda label: label[1])[0]

            size = w.compress(clustering.labels_ == label, axis=0).mean()
            size = size.round().astype(int)

            """ y """
            y = np.asarray(m).astype(bool).T
            y[:x - size] = False
            y[x + 1:] = False
            y = [np.where(col)[0] for col in y]
            y = np.array([col.min() for col in y if col.size]).reshape(-1, 1)

            clustering = DBSCAN(eps=1, min_samples=1)
            clustering.fit(y)

            labels = [label for label in sorted(set(clustering.labels_)) if not label < 0]
            n_clusters = [(clustering.labels_ == label).sum() for label in labels]
            label = max(zip(labels, n_clusters), key=lambda label: label[1])[0]

            y = y.compress(clustering.labels_ == label, axis=0).mean()
            y = y.round().astype(int)

            """ fix """
            if text[0] == '「':
                y -= np.round(size * 0.65).astype(int)

            if text[0] == '（':
                y -= np.round(size * 0.65).astype(int)

            """ draw """
            font = ImageFont.truetype(font_filepath, size)
            draw.text((x, y), text, fill=text_color, font=font, anchor='rt', direction=direction, stroke_width=5,
                      stroke_fill=border_color)

    """ draw text """
    draw = ImageDraw.Draw(image2)
    font = ImageFont.truetype(font_filepath, font_size)

    blocks = data2.loc[:, 'block_num']

    for i, texts in zip(blocks.unique()[1:], text2):
        lines = data2.loc[blocks == i, 'line_num']

        lefts = data2.loc[blocks == i, 'left']
        tops = data2.loc[blocks == i, 'top']
        widths = data2.loc[blocks == i, 'width']
        heights = data2.loc[blocks == i, 'height']
        confidences = data2.loc[blocks == i, 'conf']

        for j, text in zip(lines.unique()[1:], texts):
            m = Image.new('L', image2.size, 0)
            d = ImageDraw.Draw(m)
            boxes = zip(
                lefts.loc[lines == j],
                tops.loc[lines == j],
                widths.loc[lines == j],
                heights.loc[lines == j],
                confidences.loc[lines == j],
            )

            for x, y, w, h, c in boxes:
                if c != '-1':
                    d.rectangle((x, y, x + w, y + h), fill=255)

            """ x """
            x = np.asarray(m).astype(bool)
            x = [np.where(row)[0] for row in x]
            x = np.array([row.max() for row in x if row.size]).reshape(-1, 1)

            clustering = DBSCAN(eps=1, min_samples=1)
            clustering.fit(x)

            labels = [label for label in sorted(set(clustering.labels_)) if not label < 0]
            n_clusters = [(clustering.labels_ == label).sum() for label in labels]
            label = max(zip(labels, n_clusters), key=lambda label: label[1])[0]

            x = x.compress(clustering.labels_ == label, axis=0).mean()
            x = x.round().astype(int)

            """ size """
            w = np.asarray(m).astype(bool).sum(axis=1)
            w = w.compress(w != 0).reshape(-1, 1)

            clustering = DBSCAN(eps=1, min_samples=1)
            clustering.fit(w)

            labels = [label for label in sorted(set(clustering.labels_)) if not label < 0]
            n_clusters = [(clustering.labels_ == label).sum() for label in labels]
            label = max(zip(labels, n_clusters), key=lambda label: label[1])[0]

            size = w.compress(clustering.labels_ == label, axis=0).mean()
            size = size.round().astype(int)

            """ y """
            y = np.asarray(m).astype(bool).T
            y[:x - size] = False
            y[x + 1:] = False
            y = [np.where(col)[0] for col in y]
            y = np.array([col.min() for col in y if col.size]).reshape(-1, 1)

            clustering = DBSCAN(eps=1, min_samples=1)
            clustering.fit(y)

            labels = [label for label in sorted(set(clustering.labels_)) if not label < 0]
            n_clusters = [(clustering.labels_ == label).sum() for label in labels]
            label = max(zip(labels, n_clusters), key=lambda label: label[1])[0]

            y = y.compress(clustering.labels_ == label, axis=0).mean()
            y = y.round().astype(int)

            """ fix """
            if text[0] == '「':
                y -= np.round(size * 0.65).astype(int)

            if text[0] == '（':
                y -= np.round(size * 0.65).astype(int)

            """ draw """
            font = ImageFont.truetype(font_filepath, size)
            draw.text((x, y), text, fill=(0, 0, 0), font=font, anchor='rt', direction=direction, stroke_width=5,
                      stroke_fill=border_color)

    """ show """
    ImageChops.difference(image1, image2).show()
    break
