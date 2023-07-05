#check if data is in RGB
from PIL import Image
import numpy as np

img = Image.open( "D:/QuPath Projekt/tiles/1HE/annotation_1/1HE x-11762_y-38874_w-512_h-512.png")
img.load()
print(img.mode)

#%%

#%%
import torch

torch.cuda.is_available()
#%%
# The path can also be read from a config file, etc.
OPENSLIDE_PATH = r'C:\Users\phili\OpenSlide\openslide-win64-20230414\bin'

import os
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
#%%
import sys
for p in sys:
    print(p)
#%%
