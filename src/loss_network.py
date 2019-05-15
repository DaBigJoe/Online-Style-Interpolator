"""
Loss network implementation.

Uses a pre-trained network (VGG-16 trained on ImageNet) to evaluate the loss of a transformed image with reference
style and content images.

Author: Joseph Early (je5g15@soton.ac.uk)
Created: 17/04/19
"""

import torch
from torch import nn
from torchvision.models import vgg16
from collections import namedtuple


class TruncatedVgg16(torch.nn.Module):
    """
    A truncated version of the Vgg16 network that only goes as deep as we need it to.
    The final output of the network doesn't actually matter - only the intermediary layers are used to approximate
     the loss functions for the image comparisons.
    """

    def __init__(self):
        super(TruncatedVgg16, self).__init__()
        vgg_pretrained_features = vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class LossNetwork:
    """
    A wrapper around the truncated Vgg16 network provided computation of the style and content loss functions.
    """

    def __init__(self, style_tensors, device):
        self.model = TruncatedVgg16().to(device)
        self.model.eval()
        self.mse_loss = torch.nn.MSELoss()

        # Pre-compute style features
        style_features = self.model(style_tensors)
        self.style_grams = [gram_matrix(y) for y in style_features]

    def calculate_loss(self, batched_x, batched_y, style_idx):
        """
        Calculate the style loss and content loss of an input image compared to a target style image and a
         target content image.
        """
        x_features = self.model(batched_x)
        y_features = self.model(batched_y)
        content_loss = self.content_loss(x_features, y_features)
        style_loss = self.style_loss(y_features, style_idx)
        return content_loss, style_loss

    def content_loss(self, x_features, y_features):
        """
        Calculate the content loss between a set of predicted outputs and a set of target content outputs.
        """
        return self.mse_loss(x_features.relu2_2, y_features.relu2_2)

    def style_loss(self, y_features, style_idx):
        loss = 0.
        for layer_idx in range(len(y_features)):
            loss += self.style_loss_single(y_features, style_idx, layer_idx)
        return loss

    def style_loss_single(self, y_features, style_idx, layer_idx):
        y_feature = y_features[layer_idx]
        style_gram = self.style_grams[layer_idx]
        y_gram = gram_matrix(y_feature)
        return self.mse_loss(y_gram, style_gram[style_idx, :, :])


def gram_matrix(batched_matrices):
    batch, channel, height, width = batched_matrices.size()
    # Reshape target outs from b x c x h x w to b x c x hw
    m1 = batched_matrices.view(batch, channel, width * height)
    m1_t = m1.transpose(1, 2)  # Ignore batch dim when transposing
    # Actually compute gram matrix
    p = channel * height * width
    mul_mats = []
    for i in range(batch):
        mul_mats.append(m1[i].mm(m1_t[i]))
    gram_m = torch.stack(mul_mats) / p
    return gram_m
