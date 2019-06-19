"""
Pytorch dataset encapsulation.

Author: Alexander Stonard (ads1g15@soton.ac.uk)
Created: 13/05/19
"""

from os import listdir

import torch
from torch.utils import data

from src.interpolator.image_handler import load_image_as_tensor


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


class StyleManager(data.Dataset):

    def __init__(self, style_dir, device):
        self.style_dir = style_dir
        self.device = device
        self.styles = sorted(listdir(style_dir))
        print('Found', len(self.styles), 'styles:')
        for i, s in enumerate(self.styles):
            print(' Style', i, '=', s)

        style_tensors = []
        for style in self.styles:
            style_tensors.append(load_image_as_tensor(style_dir + style))
        self.style_tensors = torch.stack(style_tensors).to(device)

    def get_style_tensors(self):
        return self.get_style_tensors()

    def get_style_tensor_name(self, idx):
        return self.styles[idx]

    def get_style_tensor_subset(self, idxs):
        subset = []
        for idx in idxs:
            subset.append(self.style_tensors[idx])
        return torch.stack(subset).to(self.device)

    def __len__(self):
        return len(self.style_tensors)

    def __getitem__(self, index):
        return self.style_tensors[index]
