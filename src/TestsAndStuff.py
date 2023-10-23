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
