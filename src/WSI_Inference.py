import PIL.ImageShow
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torchvision
from skimage.measure import shannon_entropy
from torchvision import transforms
from process import is_white_or_grey_png

from config import config_hyperparameter as cfg_hp
import torch.nn as nn
import torch.optim as optim
from auxiliaries import get_model, image_entropy

# workaround for Openslide import
OPENSLIDE_PATH = r'C:\Users\phili\OpenSlide\openslide-win64-20230414\bin'
import os

if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
from openslide import deepzoom


def TestOnSingleSlide(model_folder: str, slidepath: str):
    # load classes
    class_names = cfg_hp["class_names"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # get model and parameters
    trained_model, model_results, dict_hyperparameters, summary = get_model(model_folder)

    manual_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    trained_model.to(device)
    # Turn on model evaluation mode and inference mode
    trained_model.eval()

    # open slide with openslide
    slide = openslide.open_slide(slidepath)

    tiles = deepzoom.DeepZoomGenerator(slide, tile_size=224, overlap=112, limit_bounds=False)

    col, rows = tiles.level_tiles[15]
    predictions = np.empty([rows, col])

    for c in range(col):
        for r in range(rows):
            single_tile = tiles.get_tile(15, (c, r))

            if r == 21 and c == 22:
                single_tile.save(str(c) + " " + str(r) + ".png", "PNG")

            # Sample usage
            np_tile = np.array(single_tile)

            # Pink lower bound in HSV
            lower_bound = np.array([135, 40, 40])
            # Purple upper bound in HSV
            upper_bound = np.array([172, 255, 255])

            if is_tile_of_interest(np_tile, lower_bound, upper_bound):
                # if is_white_or_grey_png(single_tile):
                with torch.inference_mode():
                    # Add an extra dimension to the image
                    single_tile = manual_transforms(single_tile).unsqueeze(dim=0)

                    # Divide the image pixel values by 255 to get them between [0, 1]
                    # target_image = single_tile / 255

                    # Make a prediction on image with an extra dimension
                    target_image_pred = trained_model(single_tile.cuda())

                target_image_pred_probs = torch.sigmoid(target_image_pred)
                predictions[r][c] = target_image_pred_probs.tolist()[0][0]
            else:
                predictions[r][c] = -1

    # Visualize Results in a Heatmap
    SlideHeatmap(heatmap_data=predictions)

def TestOnSlides(model_folder: str, test_folder: str):
    # load classes
    class_names = cfg_hp["class_names"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # get model and parameters
    trained_model, model_results, dict_hyperparameters, summary = get_model(model_folder)

    manual_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    trained_model.to(device)
    # Turn on model evaluation mode and inference mode
    trained_model.eval()
    wsipathlist = list(Path(test_folder).glob("*/*.*"))

    #array for true class
    y_test = []

    all_predictions= []

    for wsipath in wsipathlist:
        # open slide with openslide
        slide = openslide.open_slide(wsipath)

        tiles = deepzoom.DeepZoomGenerator(slide, tile_size=224, overlap=112, limit_bounds=False)

        col, rows = tiles.level_tiles[15]
        predictions = np.empty([rows, col])

        for c in range(col):
            for r in range(rows):
                single_tile = tiles.get_tile(15, (c, r))

                # Sample usage
                np_tile = np.array(single_tile)

                # Pink lower bound in HSV
                lower_bound = np.array([135, 40, 40])
                # Purple upper bound in HSV
                upper_bound = np.array([172, 255, 255])

                if is_tile_of_interest(np_tile, lower_bound, upper_bound):
                    # if is_white_or_grey_png(single_tile):
                    with torch.inference_mode():
                        # Add an extra dimension to the image
                        single_tile = manual_transforms(single_tile).unsqueeze(dim=0)

                        # Divide the image pixel values by 255 to get them between [0, 1]
                        # target_image = single_tile / 255

                        # Make a prediction on image with an extra dimension
                        target_image_pred = trained_model(single_tile.cuda())

                    target_image_pred_probs = torch.sigmoid(target_image_pred)
                    predictions[r][c] = target_image_pred_probs.tolist()[0][0]
                else:
                    predictions[r][c] = -1

        #Aggregate to WSI Result via majority voting
        #1. get rid of -1 and rount probabilities
        predictions_labels = predictions[predictions != -1].round()
        pred_class = class_names[int(predictions_labels)]
        true_class = wsipath.parts[3]

        all_predictions.append(predictions_labels.item())
        y_test.append(class_names.index(true_class))



def SlideHeatmap(heatmap_data: np.array):
    # Create a heatmap using matplotlib
    plt.imshow(heatmap_data, cmap='viridis', interpolation='nearest')
    # Add colorbar to indicate values
    plt.colorbar()

    # Show the plot
    plt.show()


def is_tile_of_interest(tile, lower_thresh, upper_thresh, percentage_thresh=0.05):
    """
    Determines if a tile is of interest based on color.

    :param tile: Input tile image
    :param lower_thresh: Lower bound for HSV values
    :param upper_thresh: Upper bound for HSV values
    :param percentage_thresh: Threshold for percentage of pixels of interest
    :return: True if the tile is of interest, False otherwise
    """

    # Convert the tile to HSV color space
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)

    # Create a mask using inRange function
    mask = cv2.inRange(hsv, lower_thresh, upper_thresh)

    # Compute the percentage of pixels of interest
    percentage = np.sum(mask > 0) / float(mask.size)

    return percentage > percentage_thresh


def is_white_or_grey_png(image: Image.Image, threshold=0.95):
    """
    Checks if the given PIL image has a high percentage of white or grey pixels.

    Parameters:
    - image: A PIL.Image.Image instance.
    - threshold: The percentage threshold to determine if an image is mostly white or grey.

    Returns:
    - True if the image's white or grey percentage is less than the given threshold, otherwise False.
    """

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
    return white_or_grey_percentage < threshold


# Define a simple Aggregation Network
class AggregationNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AggregationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.softmax = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


def aggregateResults(predictions: np.array):
    patch_level_predictions = torch.tensor(predictions)
    batch_size, num_patches, num_classes = predictions.size()
    input_size = num_patches * num_classes
    flattened_predictions = patch_level_predictions.view(batch_size, -1)

    # Create the Aggregation Network
    hidden_size = 64  # Adjust as needed
    aggregation_net = AggregationNetwork(input_size, hidden_size)

    # Perform forward pass to obtain slide-level prediction
    slide_level_prediction = aggregation_net(flattened_predictions)

    print(slide_level_prediction)


def train_aggregation_network(aggregation_net, train_loader, num_epochs, learning_rate):
    # Define Binary Cross-Entropy Loss and Adam optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(aggregation_net.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0

        for inputs, labels in train_loader:
            # Flatten patch-level predictions for each batch
            inputs = inputs.view(inputs.size(0), -1)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = aggregation_net(inputs)

            # Compute the loss
            loss = criterion(outputs, labels.float())  # Convert labels to float
            total_loss += loss.item()

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

        # Print average loss for this epoch
        average_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

    print('Training complete.')
# %%
