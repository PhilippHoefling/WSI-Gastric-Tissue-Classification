from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

#workaround for Openslide import
OPENSLIDE_PATH = r'C:\Users\phili\OpenSlide\openslide-win64-20230414\bin'
import os
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
from openslide import deepzoom



def importTestSlide(slidepath="C:/Users/phili/DataspellProjects/xAIMasterThesis/data/WSIs/86HE.mrxs"):
    slide = openslide.open_slide(slidepath)


    tiles = deepzoom.DeepZoomGenerator(slide, tile_size=256, overlap=0, limit_bounds=True)
    print(tiles.level_dimensions)
    print(tiles.level_tiles)

    #slide_thumb = slide.get_thumbnail(size=(1200, 1200))
    single_tile1 = tiles.get_tile(8, (0,0))
    single_tile2 = tiles.get_tile(13, (5,1))
    single_tile3 = tiles.get_tile(15, (20,5))
    #slide_thumb.show()
    single_tile1.show()
    single_tile2.show()
    single_tile3.show()
    #col, rows= tiles.level_tiles[15]
    #for c in range(col):
    #    for r in range(rows):
    #        tile1=tiles.get_tile(15,(c, r))
    #        plt.figure(figsize=(5,5))
    #        plt.imshow(tile1)
    #        break
    #    break
    #return tiles.get_tile(10, (0,0))
def thresholding(img, method='otsu'):
    # convert to grayscale complement image
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_c = 255 - grayscale_img
    thres, thres_img = 0, img_c.copy()
    if method == 'otsu':
        thres, thres_img = cv2.threshold(img_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'triangle':
        thres, thres_img = cv2.threshold(img_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return thres, thres_img, img_c


def histogram(img, thres_img, img_c, thres):
    """
    style: ['color', 'grayscale']
    """
    plt.figure(figsize=(15, 15))

    plt.subplot(3, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')

    plt.subplot(3, 2, 2)
    sns.histplot(img.ravel(), bins=np.arange(0, 256), color='orange', alpha=0.5)
    sns.histplot(img[:, :, 0].ravel(), bins=np.arange(0, 256), color='red', alpha=0.5)
    sns.histplot(img[:, :, 1].ravel(), bins=np.arange(0, 256), color='Green', alpha=0.5)
    sns.histplot(img[:, :, 2].ravel(), bins=np.arange(0, 256), color='Blue', alpha=0.5)
    plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
    plt.ylim(0, 0.3e6)
    plt.xlabel('Intensity value')
    plt.title('Color Histogram')

    plt.subplot(3, 2, 3)
    plt.imshow(img_c, cmap='gist_gray')
    plt.title('Complement Grayscale Image')

    plt.subplot(3, 2, 4)
    sns.histplot(img_c.ravel(), bins=np.arange(0, 256))
    plt.axvline(thres, c='red', linestyle="--")
    plt.ylim(0, 0.3e6)
    plt.xlabel('Intensity value')
    plt.title('Grayscale Complement Histogram')

    plt.subplot(3, 2, 5)
    plt.imshow(thres_img, cmap='gist_gray')
    plt.title('Thresholded Image')

    plt.subplot(3, 2, 6)
    sns.histplot(thres_img.ravel(), bins=np.arange(0, 256))
    plt.axvline(thres, c='red', linestyle="--")
    plt.ylim(0, 0.3e6)
    plt.xlabel('Intensity value')
    plt.title('Thresholded Histogram')

    plt.tight_layout()
    plt.show()


importTestSlide()

#thres_otsu, thres_img, img_c = thresholding(test_slide, method='otsu')
#histogram(test_slide, thres_img, img_c, thres_otsu)

#%%
