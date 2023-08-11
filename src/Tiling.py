from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from evaluation import get_model

import torch
import torchvision
from torchvision import transforms

from config import config_hyperparameter as cfg_hp




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



def TestonSlide(model_folder: str, slidepath: str):
    #load classes
    class_names = cfg_hp["class_names"]

    #get model and parameters
    trained_model, model_results, dict_hyperparameters, summary = get_model(model_folder)


    manual_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Turn on model evaluation mode and inference mode
    trained_model.eval()

    #open slide with openslide
    slide = openslide.open_slide(slidepath)

    tiles = deepzoom.DeepZoomGenerator(slide, tile_size=512, overlap=0, limit_bounds=True)


    col, rows= tiles.level_tiles[15]
    predictions = np.empty([col,rows])
    for c in range(col):
        for r in range(rows):
            single_tile=tiles.get_tile(15,(c, r))

            #single_tile.save(str(c) + " "+ str(r) +".png", "PNG")

            #Transform
            target_image = manual_transforms(single_tile)

            # Divide the image pixel values by 255 to get them between [0, 1]
            target_image = target_image / 255


            with torch.inference_mode():
                # Add an extra dimension to the image
                target_image = target_image.unsqueeze(dim=0)
                # Make a prediction on image with an extra dimension
                target_image_pred = trained_model(target_image.cuda())

            target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
            predictions[c][r]= target_image_pred_probs.tolist()[0][0]

    print(predictions)

    # Create a heatmap using matplotlib
    plt.imshow(predictions, cmap='viridis', interpolation='nearest')

    # Add colorbar to indicate values
    plt.colorbar()

    # Show the plot
    plt.show()
#%%








