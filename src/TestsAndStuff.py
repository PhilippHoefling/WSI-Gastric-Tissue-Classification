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

import numpy as np
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, f1_score, recall_score, roc_curve, roc_auc_score
import torch

x =torch.rand(1, 100)
print(x)
#%%
