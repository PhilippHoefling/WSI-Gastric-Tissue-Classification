import PIL.ImageShow
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from evaluation import get_model

import torch
import torchvision
from torchvision import transforms

from config import config_hyperparameter as cfg_hp
import torch.nn as nn
import torch.optim as optim



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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #get model and parameters
    trained_model, model_results, dict_hyperparameters, summary = get_model(model_folder)


    manual_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    trained_model.to(device)
    # Turn on model evaluation mode and inference mode
    trained_model.eval()

    #open slide with openslide
    slide = openslide.open_slide(slidepath)

    tiles = deepzoom.DeepZoomGenerator(slide, tile_size=256, overlap=0, limit_bounds=False)

    col, rows= tiles.level_tiles[15]
    predictions = np.empty([col,rows])
    for c in range(col):
        for r in range(rows):
            single_tile=tiles.get_tile(15,(c, r))

            #single_tile.save(str(c) + " "+ str(r) +".png", "PNG")

            with torch.inference_mode():
                # Add an extra dimension to the image
                single_tile = manual_transforms(single_tile).unsqueeze(dim=0)

                # Divide the image pixel values by 255 to get them between [0, 1]
                #target_image = single_tile / 255

                # Make a prediction on image with an extra dimension
                target_image_pred = trained_model(single_tile.cuda())

            target_image_pred_probs = torch.sigmoid(target_image_pred)
            predictions[c][r]= target_image_pred_probs.tolist()[0][0]
            print(predictions)

    #Visualize Results in a Heatmap
    SlideHeatmap(predictions=predictions)
    #
def SlideHeatmap(predictions: np.array):
    # Create a heatmap using matplotlib
    plt.imshow(np.rot90(predictions), cmap='viridis', interpolation='nearest')

    # Add colorbar to indicate values
    plt.colorbar()

    # Show the plot
    plt.show()


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
#%%








