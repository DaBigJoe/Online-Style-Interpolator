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

    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.images = listdir(image_dir)

    def get_image_tensor(self, index):
        return load_image_as_tensor(self.image_dir + self.images[index])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.get_image_tensor(index)
