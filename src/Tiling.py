OPENSLIDE_PATH = r'C:\Users\phili\OpenSlide\openslide-win64-20230414\bin'

import os
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
       import openslide
else:
    import openslide
from openslide import deepzoom
from PIL import Image
import datetime
import numpy as np


slide = openslide.open_slide("C:/Users/phili/DataspellProjects/xAIMasterThesis/data/WSIs/86HE.mrxs")
#slide_thumb = slide.get_thumbnail(size=(1200, 1200))

tiles = deepzoom.DeepZoomGenerator(slide,tile_size=256, overlap=0, limit_bounds=False)
print(tiles.tile_count)
#%%

#%%
