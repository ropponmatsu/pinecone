""" experiment for line drawing """

from PIL import Image, ImageDraw, ImageFont

font_filepath = 'meiryo.ttc'
text = 'テスト'
direction = 'ttb'

image = Image.new('RGB', (512, 512), (255, 255, 255))
draw = ImageDraw.Draw(image)
font = ImageFont.truetype(font_filepath, 40)

draw.text((256, 256), text, fill=(0, 0, 0), font=font, direction=direction)
image.show()
