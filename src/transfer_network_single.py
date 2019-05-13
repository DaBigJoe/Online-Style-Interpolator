"""
Transfer network implementation.

Trains on a single style image to produce a network that will learn that one particular style.

Author: Joseph Early (je5g15@soton.ac.uk)
Created: 17/05/19
"""

import os
import torch
from torch import optim
from torch.nn.functional import interpolate
from torchvision import transforms
from tqdm import tqdm

from image_handler import load_image_as_tensor, save_tensor_as_image, plot_image_tensor, transform_256, \
    save_tensors_as_grid
from loss_network import LossNetwork
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils import data
from os import listdir


class ResidualBlock(torch.nn.Module):
    """
    An encapsulation of the layers and forward pass for a residual block.
    """

    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.norm1 = torch.nn.InstanceNorm2d(128, affine=True)
        self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.norm2 = torch.nn.InstanceNorm2d(128, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        # Pass input through conv and norm layers as usual
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm1(self.conv2(out))
        # Add original input to output ('residual' part)
        out = out + residual
        return out


class TransferNetworkSingle(torch.nn.Module):
    """
    A network implementation that takes an input images and applies a style to.

    Can only learn a single style.
    """

    def __init__(self):
        super(TransferNetworkSingle, self).__init__()

        # Input = 3 x 255 x 255
        # Downsampling layers
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4)
        self.norm1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.norm2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.norm3 = torch.nn.InstanceNorm2d(128, affine=True)

        # Residual blocks
        self.res1 = ResidualBlock()
        self.res2 = ResidualBlock()
        self.res3 = ResidualBlock()
        self.res4 = ResidualBlock()
        self.res5 = ResidualBlock()

        # Upsampling layers
        self.conv4 = torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.norm4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv5 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.norm5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv6 = torch.nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)

        self.relu = torch.nn.ReLU()

    def forward(self, x):

        # Apply downsampling
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))
        # Apply residual blocks
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        # Apply upsampling
        y = torch.nn.functional.interpolate(y, mode='nearest', scale_factor=2)  # Upsample by 2
        y = self.relu(self.norm4(self.conv4(y)))
        y = torch.nn.functional.interpolate(y, mode='nearest', scale_factor=2)  # Upsample by 2
        y = self.relu(self.norm5(self.conv5(y)))
        y = self.conv6(y)
        return y


class Dataset(data.Dataset):

    def __init__(self, image_dir):
        self.data = []

        items = listdir(image_dir)

        self.image_dir = image_dir
        self.data = items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = load_image_as_tensor(self.image_dir + self.data[index], transform=transform_256).squeeze(0)
        if not (len(X) == 3):
            _X = torch.zeros((3, X.shape[1], X.shape[2]))
            _X[0] = X[0]
            _X[1] = X[0]
            _X[2] = X[0]
            X = _X
        return X


class TransferNetworkTrainerSingle:

    def __init__(self, style_path, content_dir, save_directory, test_image_path):
        print('Creating single transfer network')
        self.style_path = style_path

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Load content images
        print('Loading content images')
        train_dataset = Dataset(content_dir)
        self.train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        print('Found', len(self.train_loader), 'images')
        self.test_image_tensor = load_image_as_tensor(test_image_path, transform=transform_256).to(self.device)

        # Load style
        self.style_tensor = load_image_as_tensor(self.style_path, transform=transform_256).to(self.device)

        # Load loss network
        self.loss_network = LossNetwork()

        # Setup saving
        num_previous_runs = 0
        if os.path.exists(save_directory):
            num_previous_runs = len([i for i in os.listdir(save_directory)
                                     if os.path.isdir(os.path.join(save_directory, i))])
        self.save_directory = os.path.join(save_directory, "{:04d}".format(num_previous_runs+1))
        print(' Save dir:', self.save_directory)
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

    def train(self, num_parameter_updates=40000, num_checkpoints=9):
        print('Training single transfer network')
        model = TransferNetworkSingle().to(self.device)
        optimiser = optim.Adam(model.parameters(), lr=1e-3)
        checkpoint_freq = num_parameter_updates // num_checkpoints

        # Weight of style vs content
        style_weight = 1e12
        content_weight = 1e5

        # Train
        checkpoint = 0
        update_count = 0
        with tqdm(total=num_parameter_updates, ncols=120) as progress_bar:
            image_tensors = []
            while update_count < num_parameter_updates:

                for batch_num, content_tensors in enumerate(self.train_loader):
                    if update_count >= num_parameter_updates:
                        break

                    optimiser.zero_grad()
                    # Pass through transfer network
                    content_tensors = content_tensors.to(self.device)
                    output = model(content_tensors)

                    # Calculate loss
                    style_loss, content_loss = self.loss_network.calculate_image_loss(output, self.style_tensor,
                                                                                      content_tensors)
                    style_loss = style_loss.mul(style_weight)
                    content_loss = content_loss.mul(content_weight)
                    loss = content_loss.add(style_loss)

                    # Backprop (train) transfer network
                    loss.backward()
                    optimiser.step()

                    # Update tqdm bar
                    style_loss_formatted = "%.0f" % style_loss
                    content_loss_formatted = "%.0f" % style_loss
                    progress_bar.set_postfix(checkpoint=checkpoint, style_loss=style_loss_formatted,
                                             content_loss=content_loss_formatted)

                    # Checkpoint
                    if update_count % checkpoint_freq == 0:
                        checkpoint_file_path = os.path.join(self.save_directory, str(checkpoint+1) + '.jpeg')
                        test_output = model(self.test_image_tensor)
                        image_tensors.append(test_output)
                        save_tensor_as_image(test_output, checkpoint_file_path)
                        checkpoint += 1

                    update_count += 1
                    progress_bar.update(1)

        # Save image
        final_output = model(self.test_image_tensor)
        image_tensors.append(self.test_image_tensor)
        image_tensors.append(self.style_tensor)
        image_tensors.append(final_output)

        final_file_path = os.path.join(self.save_directory, 'final.jpeg')
        save_tensor_as_image(final_output, final_file_path)

        grid_file_path = os.path.join(self.save_directory, 'grid.jpeg')
        save_tensors_as_grid(image_tensors, grid_file_path, 5)

        # Show images (requires reload)
        plot_image_tensor(load_image_as_tensor(final_file_path))
        plot_image_tensor(load_image_as_tensor(grid_file_path, transform=transforms.ToTensor()))


if __name__ == '__main__':
    style_path = '../data/images/style/Van_Gogh_Starry_Night.jpg'
    content_dir = '/home/data/train2014/'
    save_path = '../data/images/produced/venice'
    test_image_path = '../data/images/content/venice.jpeg'
    transfer_network = TransferNetworkTrainerSingle(style_path, content_dir, save_path, test_image_path)
    transfer_network.train()
