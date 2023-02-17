import visualkeras
from model import unet
from tensorflow.keras.layers import BatchNormalization, Conv2D, UpSampling2D, Dropout, MaxPooling2D, concatenate
from collections import defaultdict

model = unet()

from PIL import ImageFont

color_map = defaultdict(dict)
color_map[Conv2D]['fill'] = 'orange'
color_map[BatchNormalization]['fill'] = 'gray'
color_map[Dropout]['fill'] = 'pink'
color_map[MaxPooling2D]['fill'] = 'red'
color_map[UpSampling2D]['fill'] = 'green'
color_map[concatenate]['fill'] = 'teal'

font = ImageFont.truetype("arial.ttf", 32)  # using comic sans is strictly prohibited!
visualkeras.layered_view(model, color_map = color_map,legend=True, font=font, to_file='model_visualize.png', draw_volume=False, spacing=50, padding=100, ).show()  # font is optional!