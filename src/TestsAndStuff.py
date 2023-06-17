#check if data is in RGB
from PIL import Image
import numpy as np

img = Image.open( "D:/QuPath Projekt/tiles/1HE/annotation_1/1HE x-11762_y-38874_w-512_h-512.png")
img.load()
print(img.mode)

#%%

#%%
