"""
Image helper functions.

Author: Joseph Early (je5g15@soton.ac.uk)
Created: 17/05/19
"""

import matplotlib.pyplot as plt
from PIL import Image
from torch.autograd import Variable


def load_image_as_tensor(image_path, transform=None):
    """
    Load a single image as a tensor from the given path. Applies a transformation if one is given.
    """
    image = Image.open(image_path)
    if transform is not None:
        image = transform(image).float()
    else:
        image.float()
    image = Variable(image, requires_grad=False)
    image = image.unsqueeze(0)
    return image


def plot_image_tensor(image_tensor):
    """
    Plot a single image using matplotlib.
    """
    plt.figure()
    plt.imshow(image_tensor[0].permute(1, 2, 0))
    plt.show()
