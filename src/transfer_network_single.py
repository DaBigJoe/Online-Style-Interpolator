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
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data_manager import Dataset
from image_handler import load_image_as_tensor, save_tensor_as_image, plot_image_tensor, transform_256, \
    save_tensors_as_grid
from loss_network import LossNetwork


class ConditionalInstanceNorm2d(torch.nn.Module):

    def __init__(self, num_channels, num_styles, affine=True):
        super(ConditionalInstanceNorm2d, self).__init__()
        # Create one norm 2d for each style
        self.norm2ds = torch.nn.ModuleList([torch.nn.InstanceNorm2d(num_channels, affine=affine)
                                            for _ in range(num_styles)])

    def forward(self, x, style_idx):
        return self.norm2ds[style_idx](x)


class ResidualBlock(torch.nn.Module):
    """
    An encapsulation of the layers and forward pass for a residual block.
    """

    def __init__(self, num_styles):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.norm1 = ConditionalInstanceNorm2d(128, num_styles, affine=True)
        self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.norm2 = ConditionalInstanceNorm2d(128, num_styles, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x, style_idx):
        residual = x
        # Pass input through conv and norm layers as usual
        out = self.relu(self.norm1(self.conv1(x), style_idx))
        out = self.norm1(self.conv2(out), style_idx)
        # Add original input to output ('residual' part)
        out = out + residual
        return out


class TransferNetworkSingle(torch.nn.Module):
    """
    A network implementation that takes an input images and applies a style to.

    Can only learn a single style.
    """

    def __init__(self, num_styles):
        super(TransferNetworkSingle, self).__init__()

        # Input = 3 x 255 x 255
        # Downsampling layers
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4)
        self.norm1 = ConditionalInstanceNorm2d(32, num_styles, affine=True)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.norm2 = ConditionalInstanceNorm2d(64, num_styles, affine=True)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.norm3 = ConditionalInstanceNorm2d(128, num_styles, affine=True)

        # Residual blocks
        self.res1 = ResidualBlock(num_styles)
        self.res2 = ResidualBlock(num_styles)
        self.res3 = ResidualBlock(num_styles)
        self.res4 = ResidualBlock(num_styles)
        self.res5 = ResidualBlock(num_styles)

        # Upsampling layers
        self.conv4 = torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.norm4 = ConditionalInstanceNorm2d(64, num_styles, affine=True)
        self.conv5 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.norm5 = ConditionalInstanceNorm2d(32, num_styles, affine=True)
        self.conv6 = torch.nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)

        self.relu = torch.nn.ReLU()

    def forward(self, x, style_idx):
        # Apply downsampling
        y = self.relu(self.norm1(self.conv1(x), style_idx))
        y = self.relu(self.norm2(self.conv2(y), style_idx))
        y = self.relu(self.norm3(self.conv3(y), style_idx))
        # Apply residual blocks
        y = self.res1(y, style_idx)
        y = self.res2(y, style_idx)
        y = self.res3(y, style_idx)
        y = self.res4(y, style_idx)
        y = self.res5(y, style_idx)
        # Apply upsampling
        y = torch.nn.functional.interpolate(y, mode='nearest', scale_factor=2)  # Upsample by 2
        y = self.relu(self.norm4(self.conv4(y), style_idx))
        y = torch.nn.functional.interpolate(y, mode='nearest', scale_factor=2)  # Upsample by 2
        y = self.relu(self.norm5(self.conv5(y), style_idx))
        y = self.conv6(y)
        return y


class TransferNetworkTrainerSingle:

    def __init__(self, content_dir, style_dir, save_directory, test_image_path, stats_file_path):
        print('Creating single transfer network')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Load loss network
        self.loss_network = LossNetwork()

        # Load content images
        print('Loading content images')
        self.train_dataset = Dataset(content_dir, style_dir, self.loss_network)
        self.train_loader = DataLoader(self.train_dataset, batch_size=4, shuffle=False)
        print('Found', len(self.train_loader), 'images')
        self.test_image_tensor = load_image_as_tensor(test_image_path, transform=transform_256)
        self.test_image_tensor = self.test_image_tensor.unsqueeze(0).to(self.device)

        # Load style
        self.style_tensor = self.train_dataset.get_style_tensor(0)

        # Setup saving
        num_previous_runs = 0
        if os.path.exists(save_directory):
            num_previous_runs = len([i for i in os.listdir(save_directory)
                                     if os.path.isdir(os.path.join(save_directory, i))])
        self.save_directory = os.path.join(save_directory, "{:04d}".format(num_previous_runs+1))
        print(' Save dir:', self.save_directory)
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        self.stats_file = open(stats_file_path, 'w+')

    def train(self, num_parameter_updates=40000, num_checkpoints=1000):
        num_styles = self.train_dataset.get_style_count()

        print('Training single transfer network')
        model = TransferNetworkSingle(num_styles).to(self.device)
        optimiser = optim.Adam(model.parameters(), lr=1e-3)
        checkpoint_freq = num_checkpoints

        # Weight of style vs content
        style_weight = 1e12
        content_weight = 1e5

        # Train
        checkpoint = 0
        update_count = 0
        style_count = 0
        with tqdm(total=num_parameter_updates, ncols=120) as progress_bar:
            checkpoint_tensors = []
            while update_count < num_parameter_updates:

                for batch_num, (image_tensors, content_target) in enumerate(self.train_loader):
                    if update_count >= num_parameter_updates:
                        break

                    optimiser.zero_grad()

                    # Style tensor
                    style_tensor = self.train_dataset.get_style_tensor(style_count % num_styles)

                    # Pass through transfer network
                    image_tensors = image_tensors.to(self.device)
                    output = model(image_tensors, style_count % num_styles)

                    # Calculate loss
                    style_loss, content_loss = self.loss_network.calculate_loss_with_precomputed(output, style_tensor, content_target)
                    style_loss = style_loss.mul(style_weight)
                    content_loss = content_loss.mul(content_weight)
                    loss = content_loss.add(style_loss)

                    # Backprop (train) transfer network
                    loss.backward()
                    optimiser.step()

                    # Update tqdm bar
                    style_loss_formatted = "%.0f" % style_loss
                    content_loss_formatted = "%.0f" % content_loss
                    progress_bar.set_postfix(checkpoint=checkpoint, style_loss=style_loss_formatted,
                                             content_loss=content_loss_formatted)

                    # Checkpoint
                    if update_count % checkpoint_freq == 0:
                        checkpoint_file_path = os.path.join(self.save_directory, str(checkpoint+1) + '.jpeg')
                        test_output = model(self.test_image_tensor, style_count % num_styles)
                        checkpoint_tensors.append(test_output)
                        save_tensor_as_image(test_output, checkpoint_file_path)
                        checkpoint += 1

                    self.stats_file.write(str(update_count) + ', ' + str(style_loss.item()) + ', ' + str(content_loss.item()) + '\n')

                    update_count += 1
                    progress_bar.update(1)

                    style_count += 1

        self.stats_file.close()
        torch.save(model.state_dict(), '/home/stonarda/uni/deep_learning/updated_model.pt')

        # Save image
        final_output = model(self.test_image_tensor, 0)
        checkpoint_tensors.append(self.test_image_tensor)
        checkpoint_tensors.append(self.style_tensor)
        checkpoint_tensors.append(final_output)

        final_file_path = os.path.join(self.save_directory, 'final.jpeg')
        #save_tensor_as_image(final_output, final_file_path)

        grid_file_path = os.path.join(self.save_directory, 'grid.jpeg')
        #save_tensors_as_grid(checkpoint_tensors, grid_file_path, 5)

        # Show images (requires reload)
        #plot_image_tensor(load_image_as_tensor(final_file_path))
        #plot_image_tensor(load_image_as_tensor(grid_file_path, transform=transforms.ToTensor()))

        print('DONE')


if __name__ == '__main__':
    style_dir = '../data/images/style/'
    content_dir = '../data/coco/'
    save_path = '../data/checkpoints/'
    test_image_path = '../data/images/content/venice.jpeg'
    stats_file_path = '../stats_file.csv'
    transfer_network = TransferNetworkTrainerSingle(content_dir, style_dir, save_path, test_image_path, stats_file_path)
    transfer_network.train()
