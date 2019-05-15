"""
Pytorch dataset encapsulation.

Author: Alexander Stonard (ads1g15@soton.ac.uk)
Created: 13/05/19
"""

import numpy as np
import os
import torch
from os import listdir
from torch.utils import data

from image_handler import load_image_as_tensor


class Dataset(data.Dataset):
    """
    A Dataset for loading the data used to train the network
    """

    def __init__(self, image_dir,  style_dir, loss_network):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.image_dir = image_dir
        self.images = listdir(image_dir)

        self.style_images = []
        self.style_tensors = []
        for index, style in enumerate(listdir(style_dir)):
            style_image = load_image_as_tensor(style_dir + style).unsqueeze(0).to(self.device)
            style_tensor = loss_network.calculate_style_outputs(style_image)

            _style_tensor = []
            for st in style_tensor:
                _style_tensor.append(st.detach())
            style_tensor = _style_tensor

            self.style_images.append(style_image.cpu())
            self.style_tensors.append(style_tensor)

            print('STYLE: \'' + str(style) + '\' is index ' + str(index))

        self.loss_network = loss_network

        print('Dataset Loaded, Found: ' + str(len(self.style_images)) + ' style images and ' + str(len(self.images)) + ' content tensors')

    def get_style_count(self):
        return len(self.style_images)

    def get_style_image(self, index):
        return self.style_images[index]

    def get_style_tensor(self, index):
        return self.style_tensors[index]

    def get_image_count(self):
        return len(self.images)

    def get_image_tensor(self, index):
        image = load_image_as_tensor(self.image_dir + self.images[index])
        return image

    def get_content_tensor(self, index):
        name = self.images[index]

        content_image = load_image_as_tensor(self.image_dir + name).to(self.device).detach()
        content_tensor = self.loss_network.calculate_content_outputs(content_image.unsqueeze(0))

        return content_tensor

    def __len__(self):
        return self.get_image_count()

    def __getitem__(self, index):
        image_tensor = self.get_image_tensor(index)
        content_tensor = self.get_content_tensor(index)
        return image_tensor, content_tensor
