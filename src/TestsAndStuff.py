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
    heatmap_data = df.pivot('batch_size', 'learning_rate', 'val_loss')

    # Create the heatmap using seaborn
    heatmap = sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap='viridis')

    # Add a label to the color bar
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label('Validation Loss')

    # Add labels and a title for clarity
    plt.xlabel('Learning Rate')
    plt.ylabel('Batch Size')



    plt.savefig("heatmap_ResNet50_Tissue_valloss", bbox_inches='tight', dpi=300)
    # Display the heatmap
    plt.show()

# Replace 'path_to_csv.csv' with your actual CSV file path
csv_file_path = 'D:/ResNet50Tissue.csv'
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

TestSlides = {
    '40BHE': ['inflamed',['Corpus','Antrum']],
    '20BHE': ['inflamed',['Antrum']]

}

for slide in TestSlides:
    print(slide)
    print(TestSlides[slide][0])
#%%
