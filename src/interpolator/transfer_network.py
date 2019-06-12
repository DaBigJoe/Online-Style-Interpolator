"""
Transfer network implementation.

Author: Joseph Early (je5g15@soton.ac.uk)
Created: 17/05/19
"""

import torch


class TransferNetwork(torch.nn.Module):
    """
    A network implementation that takes an input images and applies a style to.

    Can only learn a single style.
    """

    def __init__(self, num_styles):
        super(TransferNetwork, self).__init__()

        # Input = 3 x 255 x 255
        # Downsampling layers
        self.refl1 = torch.nn.ReflectionPad2d(4)
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=0)
        self.norm1 = ConditionalInstanceNorm2d(32, num_styles, affine=True)
        self.refl2 = torch.nn.ReflectionPad2d(1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0)
        self.norm2 = ConditionalInstanceNorm2d(64, num_styles, affine=True)
        self.refl3 = torch.nn.ReflectionPad2d(1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.norm3 = ConditionalInstanceNorm2d(128, num_styles, affine=True)

        # Residual blocks
        self.res1 = ResidualBlock(num_styles)
        self.res2 = ResidualBlock(num_styles)
        self.res3 = ResidualBlock(num_styles)
        self.res4 = ResidualBlock(num_styles)
        self.res5 = ResidualBlock(num_styles)

        # Upsampling Layers
        self.ups4 = torch.nn.Upsample(mode='nearest', scale_factor=2)
        self.refl4 = torch.nn.ReflectionPad2d(1)
        self.conv4 = torch.nn.Conv2d(128,64,kernel_size=3,stride=1)

        self.norm4 = ConditionalInstanceNorm2d(64, num_styles, affine=True)

        self.ups5 = torch.nn.Upsample(mode='nearest', scale_factor=2)
        self.refl5 = torch.nn.ReflectionPad2d(1)
        self.conv5 = torch.nn.Conv2d(64,32,kernel_size=3,stride=1)

        self.norm5 = ConditionalInstanceNorm2d(32, num_styles, affine=True)

        self.refl6 = torch.nn.ReflectionPad2d(4)
        self.conv6 = torch.nn.Conv2d(32,3,kernel_size=9,stride=1)

        self.relu = torch.nn.ReLU()

    def forward(self, x, style_idx):
        # Apply downsampling
        y = self.relu(self.norm1(self.conv1(self.refl1(x)), style_idx))
        y = self.relu(self.norm2(self.conv2(self.refl2(y)), style_idx))
        y = self.relu(self.norm3(self.conv3(self.refl3(y)), style_idx))
        # Apply residual blocks
        y = self.res1(y, style_idx)
        y = self.res2(y, style_idx)
        y = self.res3(y, style_idx)
        y = self.res4(y, style_idx)
        y = self.res5(y, style_idx)

        # Apply upsampling
        y = self.ups4(y)
        y = self.refl4(y)
        y = self.conv4(y)

        y = self.norm4(y, style_idx)
        y = self.relu(y)

        y = self.ups5(y)
        y = self.refl5(y)
        y = self.conv5(y)

        y = self.norm5(y, style_idx)
        y = self.relu(y)

        y = self.refl6(y)
        y = self.conv6(y)

        return y

    def get_all_conditional_norms(self):
        conditional_norms = [self.norm1, self.norm2, self.norm3,
                             self.res1.norm1, self.res1.norm2,
                             self.res2.norm1, self.res2.norm2,
                             self.res3.norm1, self.res3.norm2,
                             self.res4.norm1, self.res4.norm2,
                             self.res5.norm1, self.res5.norm2,
                             self.norm4, self.norm5]
        return conditional_norms

    def get_style_parameters(self, style_idx):
        conditional_norms = self.get_all_conditional_norms()
        weight_tensors = []
        bias_tensors = []
        for conditional_norm in conditional_norms:
            weight_tensors.append(conditional_norm.norm2ds[style_idx].weight)
            bias_tensors.append(conditional_norm.norm2ds[style_idx].bias)
        return weight_tensors, bias_tensors

    def set_style_parameters(self, style_parameters, style_idx):
        """
        Load a set of style parameters into a particular style slice.
        """
        conditional_norms = self.get_all_conditional_norms()
        weight_tensors, bias_tensors = style_parameters
        for i in range(len(conditional_norms)):
            conditional_norms[i].norm2ds[style_idx].weight.data = weight_tensors[i]
            conditional_norms[i].norm2ds[style_idx].bias.data = bias_tensors[i]


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
        self.refl1 = torch.nn.ReflectionPad2d(1)
        self.conv1 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.norm1 = ConditionalInstanceNorm2d(128, num_styles, affine=True)
        self.refl2 = torch.nn.ReflectionPad2d(1)
        self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.norm2 = ConditionalInstanceNorm2d(128, num_styles, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x, style_idx):
        residual = x
        # Pass input through conv and norm layers as usual
        out = self.relu(self.norm1(self.conv1(self.refl1(x)), style_idx))
        out = self.norm1(self.conv2(self.refl2(out)), style_idx)
        # Add original input to output ('residual' part)
        out = out + residual
        return out
