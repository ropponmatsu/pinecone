import numpy as np
from PIL import Image, ImageDraw

from collections import Counter


def get_text_data(data, size):
    text_data = []
    hierarchy = get_text_hierarchy(data, threshold=-1)

    for block in hierarchy:
        coords = []
        sizes = []
        texts = []

        for paragraph in block:
            for line in paragraph:
                text = ''.join(word['text'] for word in line if isinstance(word['text'], str))
                text = text.strip()

                if not text:
                    continue

                x, y, w, _ = get_text_bounding_box(line, size)

                texts.append(text)
                coords.append((x, y))
                sizes.append(w)

        if texts:
            w = int(np.array(sizes).mean().round())
            x = [x for x, _ in coords]
            s = [x1 - x2 for x1, x2 in zip(x[:-1], x[1:])]
            s = int(np.array(s).mean().round()) - w if s else None

            text_data.append({
                'coordinate': coords[0],
                'size': w,
                'spacing': s,
                'text': texts,
            })

    return text_data


def get_text_hierarchy(data, threshold=0):
    text_data = []
    keys = ['level', 'left', 'top', 'width', 'height', 'conf', 'text']

    for bn in data.loc[:, 'block_num'].unique():
        block = data.loc[data.loc[:, 'block_num'] == bn]
        block_data = []

        for pn in block.loc[:, 'par_num'].unique():
            paragraph = block.loc[block.loc[:, 'par_num'] == pn]
            paragraph_data = []

            for ln in paragraph.loc[:, 'line_num'].unique():
                line = paragraph.loc[paragraph.loc[:, 'line_num'] == ln]
                line_data = []

                for wn in line.loc[:, 'word_num'].unique():
                    word = line.loc[line.loc[:, 'word_num'] == wn]

                    if word['conf'].item() >= threshold:
                        values = word[keys].squeeze().to_list()
                        word_data = dict(zip(keys, values))
                        line_data.append(word_data)

                if line_data:
                    paragraph_data.append(line_data)

            if paragraph_data:
                block_data.append(paragraph_data)

        if block_data:
            text_data.append(block_data)

    return text_data


def get_text_mask(data, size):
    def draw_mask(_data):
        for d in _data:
            if isinstance(d, dict):
                x, y, w, h = map(d.get, ('left', 'top', 'width', 'height'))
                draw.rectangle((x, y, x + w, y + h), fill=255)
            else:
                draw_mask(d)

    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    draw_mask(data)

    return np.array(mask)


def get_text_bounding_box(data, size):
    mask = get_text_mask(data, size)

    indices = [np.where(row == 255)[0] for row in mask]
    indices = [i for i in indices if i.size]
    max_indices = [i[-1] for i in indices]
    min_indices = [i[0] for i in indices]
    x_max = max(Counter(max_indices).items(), key=lambda c: c[1])[0]
    x_min = max(Counter(min_indices).items(), key=lambda c: c[1])[0]
    x_max = int(x_max)
    x_min = int(x_min)

    # mask[:, :x_min] = 0
    # mask[:, x_max + 1:] = 0

    indices = [np.where(col == 255)[0] for col in mask.T]
    indices = [i for i in indices if i.size]
    max_indices = [i[-1] for i in indices]
    min_indices = [i[0] for i in indices]
    y_max = max(Counter(max_indices).items(), key=lambda c: c[1])[0]
    y_min = max(Counter(min_indices).items(), key=lambda c: c[1])[0]
    y_max = int(y_max)
    y_min = int(y_min)

    return x_max, y_min, x_max - x_min + 1, y_max - y_min + 1
