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


class TruncatedVgg16(torch.nn.Module):
    """
    A truncated version of the Vgg16 network that only goes as deep as we need it to.
    The final output of the network doesn't actually matter - only the intermediary layers are used to approximate
     the loss functions for the image comparisons.
    """

    def __init__(self):
        super(TruncatedVgg16, self).__init__()
        # relu2_2 (8) for content
        # relu1_2 (3), relu2_2 (8), relu3_3 (15) and relu4_3 (22) for style
        self.target_layers = [3, 8, 15, 22]
        features = list(vgg16(pretrained=True).features)[:self.target_layers[-1] + 1]
        self.features = nn.ModuleList(features).eval()
        # print(self.features)

    def forward(self, x):
        outputs = []
        # Pass input through the network and grab the outputs of the layers we need.
        for layer_index, feature in enumerate(self.features):
            x = feature(x)
            if layer_index in self.target_layers:
                outputs.append(x)
        return outputs


class LossNetwork:
    """
    A wrapper around the truncated Vgg16 network provided computation of the style and content loss functions.
    """

    def __init__(self):
        self.model = TruncatedVgg16().to("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        for p in self.model.parameters():
            p.require_grads = False
        self.mse_loss = torch.nn.MSELoss()

    def calculate_style_outputs(self, tensor):
        """
        Pass a tensor through the network and get the style outputs.
        """
        return self.model(tensor)

    def calculate_content_outputs(self, tensor):
        """
        Pass a tensor through the network and get the content outputs.
        """
        return self.model(tensor)[1]

    def calculate_image_loss(self, image_tensor, style_tensor, content_tensor):
        """
        Calculate the style loss and content loss of an input image compared to a target style image and a
         target content image.
        """
        # Forward pass each image tensor and extract outputs
        style_target_outputs = self.calculate_style_outputs(style_tensor)
        content_target_outputs = self.calculate_content_outputs(content_tensor.cuda())

        return self.calculate_loss_with_precomputed(image_tensor, style_target_outputs, content_target_outputs)

    def calculate_loss_with_precomputed(self, image_tensor, style_target_outputs, content_target_outputs):
        """
        Calculate the style loss and content loss of an input image compared to the target style outputs and the
         target content outputs.
        """

        # Forward pass each image tensor and extract outputs
        predicted_outputs = self.model(image_tensor)

        # Compute and return style loss and content loss
        style_loss = self._style_loss(predicted_outputs, style_target_outputs)
        content_loss = self._content_loss(predicted_outputs[1], content_target_outputs)
        return style_loss, content_loss

    def _style_loss(self, predicted_outputs, target_outputs):
        """
        Calculate the style loss between a set of predicted outputs and a set of target style outputs.
        """
        # Sum over all of the target outputs
        loss = 0
        for i in range(len(target_outputs)):
            loss += self._style_loss_single(predicted_outputs[i], target_outputs[i])
        return loss

    def _content_loss(self, predicted_outputs, target_outputs):
        """
        Calculate the content loss between a set of predicted outputs and a set of target content outputs.
        """
        losses = torch.zeros((len(predicted_outputs)))
        target_output = target_outputs[0]
        for i in range(len(predicted_outputs)):
            losses[i] = self.mse_loss(target_output, predicted_outputs[i])
        loss = torch.mean(losses)
        return loss

    def _style_loss_single(self, predicted_outputs, target_outputs):
        losses = torch.zeros((len(predicted_outputs)))
        target_gram = LossNetwork._gram_matrix(target_outputs[0])
        for i in range(len(predicted_outputs)):
            predicted_gram = LossNetwork._gram_matrix(predicted_outputs[i])
            losses[i] = self.mse_loss(predicted_gram, target_gram)
        loss = torch.mean(losses)
        return loss

    @staticmethod
    def _gram_matrix(m):
        # Reshape target outs from c x h x w to c x hw
        shape = torch.tensor(m.shape)
        m1 = m.reshape([shape[0], shape[1] * shape[2]])
        # Calculate gram matrix
        return m1.mm(m1.t()).div(shape.prod())
