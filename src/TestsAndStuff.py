#check if data is in RGB
from PIL import Image
import numpy as np



img = Image.open( "D:/QuPath Projekt/tiles/1HE/annotation_1/1HE x-11762_y-38874_w-512_h-512.png")
img.load()
print(img.mode)

#%%

#%%
import torch

torch.cuda.is_available()
#%%
# The path can also be read from a config file, etc.
OPENSLIDE_PATH = r'C:\Users\phili\OpenSlide\openslide-win64-20230414\bin'

import os
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
#%%
import sys
for p in sys:
    print(p)
#%%

import torch
from torchvision import datasets, transforms
import torchvision
from torch import nn
from torchvision.models.resnet import Bottleneck, ResNet
from pytorch_model_summary import summary
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
def resnet50(pretrained, progress, key, **kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)

    # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Load weights from
    weights = torchvision.models.ResNet50_Weights.DEFAULT

    pretrained_url= "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights/bt_rn50_ep200.torch"
    model.load_state_dict(torch.hub.load_state_dict_from_url(pretrained_url, progress=False))

    # Speed up training
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model

if __name__ == "__main__":
    # initialize resnet50 trunk using BT pre-trained weight
    model = resnet50(pretrained=True, progress=False, key="BT")
    summary(model)
#%%
