"""
Image helper functions.

Author: Joseph Early (je5g15@soton.ac.uk)
Created: 17/05/19
"""

import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

import torch

# Consistent transform to scale image to 256 x 256
transform_256 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])


def load_image_as_tensor(image_path, transform=transform_256):
    """
    Load a single image as a tensor from the given path. Applies a transformation if one is given.
    """
    image = Image.open(image_path)
    image = transform(image).float()
    image = Variable(image, requires_grad=False)

    # Deal with greyscale
    if not (len(image) == 3):
        _image = torch.zeros((3, image.shape[1], image.shape[2]))
        _image[0] = image[0]
        _image[1] = image[0]
        _image[2] = image[0]
        image = _image

    return image


def save_tensor_as_image(tensor, image_path):
    """
    Save a single 3D tensor to the given image path
    """
    #torchvision.utils.save_image(tensor, image_path)
    img = tensor.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(image_path)

def save_tensors_as_grid(tensors, image_path, nrow, cwidth=256, cheight=256):
    """
    Save a list of image tensors to the given image path as a grid
    """
    reshaped_tensors = []
    for tensor in tensors:
        reshaped_tensors.append(tensor.view((3, cwidth, cheight)))
    torchvision.utils.save_image(reshaped_tensors, image_path, nrow=nrow, padding=10, normalize=True,
                                 scale_each=True, pad_value=255)


def plot_image_tensor(image_tensor):
    """
    Plot a single image using matplotlib.
    """
    plt.figure()
    plt.imshow(image_tensor.permute(1, 2, 0))
    plt.show()


def normalise_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1) # new_tensor for same dimension of tensor
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0) # back to tensor within 0, 1
    return (batch - mean) / std
