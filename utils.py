import glob
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def image_loader(image_name):
    imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(get_device(), torch.float)


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
