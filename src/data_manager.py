





from torch.utils import data
from torch.utils.data import DataLoader

from os import listdir

from loss_network import LossNetwork
from image_handler import load_image_as_tensor

import torch
import numpy as np

class Dataset(data.Dataset):
    'A Dataset for loading the data used to train the network'

    def __init__(self, image_dir, content_temp_dir,  style_dir, loss_network):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.image_dir = image_dir
        self.images = listdir(image_dir)

        self.content_temp_dir = content_temp_dir
        self.content_temps = set(np.array(listdir(content_temp_dir)))

        self.style_images = []
        self.style_tensors = []
        for style in listdir(style_dir):
            style_image = load_image_as_tensor(style_dir + style).to(self.device)
            style_tensor = loss_network.calculate_style(style_image)
            style_image.cpu()

            self.style_images.append(style_image)
            self.style_tensors.append(style_tensor)

        self.loss_network = loss_network

    def get_style_count(self):
        return len(self.style_images)

    def get_style_image(self, index):
        return self.style_images[index]

    def get_style_tensor(self, index):
        return self.style_tensor(index)


    def get_image_count(self):
        return len(self.content_items)

    def get_image_tensor(self, index):
        image = load_image_as_tensor(self.image_dir + self.images[index])
        return image

    def get_content_tensor(self, index):
        name = self.images[index]

        content_tensor = None
        if name in self.content_temps:
            content_tensor = torch.load(self.content_temp_dir + name)
        else:
            content_image = self.get_image_tensor(index)
            content_image = content_image.to(self.device)

            content_tensor = self.loss_network.calculate_content(content_image)

            torch.save(content_tensor.cpu().detach(), self.content_temp_dir + name)
            self.content_temps.add(name)

        return content_tensor


    def __len__(self):
        'Denotes the total number of samples'
        return self.get_image_count()


    def __getitem__(self, index):
        image_tensor = self.get_image_tensor(index)
        content_tensor = self.get_content_tensor(index)

        return image_tensor, content_tensor

ds = Dataset('../data/coco/', '../data/temp/', '../data/images/style/', LossNetwork())


for index, item in enumerate(ds):
    if not (index % 100):
        print(str(index))

print('Done')

