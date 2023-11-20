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
image_path = ""
image = Image.open(image_path)
plt.imshow(image)
plt.axis('off')
plt.show()
if is_white_or_grey_png(image_path):
    print("The image contains just a white/grey background.")

else:
    print("The image contains tissue.")
#%%



import torch
from torchvision.models.resnet import Bottleneck, ResNet


class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def resnet50(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model


if __name__ == "__main__":
    # initialize resnet50 trunk using BT pre-trained weight
    model = resnet50(pretrained=True, progress=False, key="BT")
    print(model)
#%%
import torch

torch.cuda.empty_cache()
#%%
import os

folder_path = "C:/Users/phili/OneDrive/Masterarbeit/jsons"  # Replace with the actual path to your folder

def delete_small_json_files(folder_path):
    delteted_files = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file is a JSON file and its size is 3 bytes or less
        if filename.endswith('.json') and os.path.getsize(file_path) <= 3:
            print(f"Deleting {file_path}...")
            os.remove(file_path)
            delteted_files =+1
            print(f"{filename} deleted.")
    print("total files deleated" + str(delteted_files))

if os.path.exists(folder_path) and os.path.isdir(folder_path):
    delete_small_json_files(folder_path)
    print("Deletion process completed.")
else:
    print(f"The folder {folder_path} does not exist.")
#%%
import os

def delete_empty_folders(folder_path):
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):  # Check if the folder is empty
                print(f"Deleting empty folder: {dir_path}")
                os.rmdir(dir_path)  # Remove the empty folder

folder_to_clean = "D:/DigPatTissue2/tiles"  # Replace with the actual path to your folder

if os.path.exists(folder_to_clean) and os.path.isdir(folder_to_clean):
    delete_empty_folders(folder_to_clean)
    print("Empty folder cleanup completed.")
else:
    print(f"The folder {folder_to_clean} does not exist.")
#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(csv_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path, sep=';', decimal='.')


    # Pivot the DataFrame to create a matrix with dropout rates as rows,
    # learning rates as columns, and validation loss as values
    heatmap_data = df.pivot('batch_size', 'learning_rate', 'val_acc')

    # Create the heatmap using seaborn
    heatmap = sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap='viridis')

    # Add a label to the color bar
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label('Validation Accuracy')

    # Add labels and a title for clarity
    plt.xlabel('Learning Rate')
    plt.ylabel('Batch Size')



    plt.savefig("heatmap_ResNet50_Tissue_vallacc", bbox_inches='tight', dpi=300)
    # Display the heatmap
    plt.show()

# Replace 'path_to_csv.csv' with your actual CSV file path
csv_file_path = 'D:/ResNet50TissueAugmentation.csv'
plot_heatmap(csv_file_path)
#%%
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class RandomRot90:
    np.random.seed(42)
    def __init__(self, p=0.33):
        self.p = p

    def __call__(self, x):
        # Rotate to the left with p=0.33
        if np.random.random() < self.p:
            return torch.rot90(x, 1, [1, 2])
        # Rotate to the right with p=0.33
        elif np.random.random() < (2 * self.p):
            return torch.rot90(x, -1, [1, 2])
        return x



# Load an image
image_path = 'C:/Users/phili/OneDrive - Otto-Friedrich-Universität Bamberg/DataSpell/xAIMasterThesis/data/InflamedTiles/test/inflamed/7CHE d-5_x-58735_y-139860_w-2560_h-2560_inflamed.png'  # Replace with your image path
original_image = Image.open(image_path)

# Define a transform pipeline with all three transforms
transform_pipeline = transforms.Compose([
    transforms.RandomRotation(degrees=180),
])

# Apply the transform pipeline to create five transformed versions of the image
transformed_images = [transform_pipeline(original_image) for _ in range(5)]

# Define gap size
gap_size = 10  # Gap of 10 pixels

# Calculate total width with gaps
total_width = sum(img.width for img in transformed_images) + (len(transformed_images) - 1) * gap_size

# Create a new image that is wide enough to hold all transformed images with gaps
combined_image = Image.new('RGB', (total_width, transformed_images[0].height))

# Paste each transformed image onto the canvas with gaps
x_offset = 0
for img in transformed_images:
    combined_image.paste(img, (x_offset, 0))
    x_offset += img.width + gap_size

# Save the combined image
combined_image.save('RandomRotation.jpg')

print("The combined image with all transformations applied has been saved.")


#%%
import numpy as np
TestSlides =   {
    '25HE': ['non-inflamed',['corpus']],
    '15BHE': ['inflamed',['antrum','corpus']],
    '36CHE': ['inflamed',['antrum','corpus']],
    '6HE': ['non-inflamed',['corpus']],
    '2CHE': ['inflamed',['antrum','corpus']],
    '77HE': ['non-inflamed',['corpus']],
    '3CHE': ['inflamed',['antrum','corpus']],
    '40BHE': ['inflamed',['antrum','corpus']],
    '15CHE': ['inflamed',['antrum']],
    '29HE': ['non-inflamed',['corpus']],
    '51HE': ['non-inflamed',['corpus']],
    '35HE': ['non-inflamed',['corpus','intermediate']],
    '18HE': ['non-inflamed',['corpus','intermediate']],
    '47BHE': ['inflamed',['antrum','corpus']],
    '19CHE': ['inflamed',['antrum','corpus']],

    #nicht für test relevant
    '20BHE': ['inflamed',['antrum']],
    '66HE': ['non-inflamed',['antrum','corpus']],

    '22BHE': ['inflamed',['corpus']],
    '23CHE': ['inflamed',['antrum','corpus']]
}
# Setting the seed for reproducibility
np.random.seed(42)

# Randomly select 15 keys
random_keys = np.random.choice(list(TestSlides.keys()), size=15, replace=False)

# Create a new dictionary with the selected entries
selected_entries = {key: TestSlides[key] for key in random_keys}

print(selected_entries)
#%%
import os

def find_png_in_subfolders(main_folder, search_folder):
    # Set to store all PNG paths in main_folder
    all_pngs_in_main_folder = set()

    # Walk through all directories and subdirectories in main_folder
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.endswith('.png'):
                all_pngs_in_main_folder.add(file)

    # Set to store all PNG paths in search_folder
    all_pngs_in_search_folder = set()

    # Walk through all directories and subdirectories in search_folder
    for root, dirs, files in os.walk(search_folder):
        for file in files:
            if file.endswith('.png'):
                all_pngs_in_search_folder.add(file)

    # Check each file in the main folder if it's in the search folder
    for file in all_pngs_in_main_folder:
        if file not in all_pngs_in_search_folder:
            print(f"{file} not found in any subfolders of {search_folder}")


# Example usage
main_folder = 'D:/TomsGastricTest'
search_folder = 'C:/Users/phili/OneDrive - Otto-Friedrich-Universität Bamberg/DataSpell/xAIMasterThesis/data/InflamedTomTiles/test'
find_png_in_subfolders(main_folder, search_folder)

#%%
import os
import csv

def write_filenames_to_csv(folder_path, csv_file_path):
    """
    Writes every filename in the given folder to a CSV file.

    :param folder_path: Path to the folder whose filenames are to be written.
    :param csv_file_path: Path to the CSV file where the filenames will be stored.
    """
    # Check if the given path is indeed a folder
    if not os.path.isdir(folder_path):
        print("The provided path is not a folder.")
        return

    with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Writing header
        writer.writerow(['Filename'])

        # Writing file names
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # Check if it's a file and not a sub-folder
            if os.path.isfile(file_path):
                writer.writerow([filename])

    print(f"File names from '{folder_path}' have been written to '{csv_file_path}'")

# Example usage

import numpy as np
import torch
import matplotlib.pyplot as plt

# Assuming an arbitrary initial learning rate for illustration
initial_lr = 0.1
T_max = 75
eta_min = 0
epochs = 50


# Dummy model and optimizer
model = torch.nn.Linear(10, 2)  # Example model
optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)

# Scheduler setup
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=75, eta_min=0)

# Recording learning rates
lrs = []
for epoch in range(epochs):
    if epoch >= 10:
        scheduler.step()
    lrs.append(optimizer.param_groups[0]['lr'])

# Plotting
plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 14})
plt.plot(range(epochs), lrs, label='Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule with CosineAnnealingLR')
plt.legend()
plt.grid(True)
plt.savefig("CosineAnnealing")
plt.show()


#%%
