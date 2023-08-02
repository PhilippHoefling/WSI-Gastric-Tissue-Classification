#check if data is in RGB
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


#%%
import torch

torch.cuda.is_available()
#%%

from PIL import Image
import matplotlib.pyplot as plt
def is_white_or_grey_png(image_path, threshold=0.95):
    try:
        image = Image.open(image_path)
    except IOError:
        raise ValueError("Unable to open the image file")

    # Convert the image to grayscale for easy white/grey detection
    grayscale_image = image.convert("L")

    # Get the pixel data from the image
    pixels = grayscale_image.load()

    # Get the image size
    width, height = image.size

    # Count the number of white/grey pixels in the image
    white_or_grey_pixel_count = sum(pixels[x, y] >= 200 for x in range(width) for y in range(height))

    # Calculate the percentage of white/grey pixels in the image
    white_or_grey_percentage = white_or_grey_pixel_count / float(width * height)

    # Check if the white/grey percentage is above the threshold
    return white_or_grey_percentage > threshold

# Usage example


# Usage example
image_path = "C:/Users/phili/DataspellProjects/xAIMasterThesis/data/Processed/train/antrum/23HE d-10_x-8810_y-5210_w-2560_h-2560_antrum.png"
image = Image.open(image_path)
plt.imshow(image)
plt.axis('off')
plt.show()
if is_white_or_grey_png(image_path):
    print("The image contains just a white/grey background.")

else:
    print("The image contains tissue.")
#%%



