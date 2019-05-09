"""
Loss network implementation.

Uses a pre-trained network (VGG-16 trained on ImageNet) to evaluate the loss of a transformed image with reference
style and content images.

Author: Joseph Early (je5g15@soton.ac.uk)
Created: 17/05/19
"""

import torch
from torch import nn
from torchvision.models import vgg16

from image_handler import load_image_as_tensor, transform_256


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

    def calculate_image_loss(self, transformed_tensor, style_tensor, content_tensor):
        """
        Calculate the style loss and content loss of an input image compared to a target style image and a
         target content image.
        """
        # Forward pass each image tensor and extract outputs
        predicted_outputs = self.model(transformed_tensor)
        style_target_outputs = self.model(style_tensor)
        content_target_outputs = self.model(content_tensor)

        # Compute and return style loss and content loss
        style_loss = self._style_loss(predicted_outputs, style_target_outputs)
        content_loss = self._content_loss(predicted_outputs, content_target_outputs)
        return style_loss, content_loss

    def _style_loss(self, predicted_outputs, target_outputs):
        """
        Calculate the style loss between a set of predicted outputs and a set of target style outputs.
        """

        def gram_matrix(m):
            # Reshape target outs from c x h x w to c x hw
            shape = torch.tensor(m.shape)
            m1 = m.reshape([shape[0], shape[1] * shape[2]])
            # Calculate gram matrix
            return m1.mm(m1.t()).div(shape.prod())

        # Sum over all of the target outputs
        loss = 0
        for i in range(len(target_outputs)):
            predicted_output = predicted_outputs[i]
            target_output = target_outputs[i]

            # Reduce from singleton 4D tensors to 3D tensors
            predicted_output = predicted_output[0]
            target_output = target_output[0]

            # Calculate gram matrices
            predicted_gram = gram_matrix(predicted_output)
            target_gram = gram_matrix(target_output)

            # Calculate Frobenius norm of gram matrices
            loss += self.mse_loss(predicted_gram, target_gram)

        return loss

    def _content_loss(self, predicted_outputs, target_outputs):
        """
        Calculate the content loss between a set of predicted outputs and a set of target content outputs.
        """
        # Use output from relu3_3 (third item in target outputs from TruncatedVgg16 model)
        return self.mse_loss(target_outputs[2], predicted_outputs[2])


if __name__ == '__main__':
    """
    Runs some simple tests to ensure everything is working
    """
    # Load style image tensor
    style_path = '../data/images/style/Van_Gogh_Starry_Night.jpg'
    style_tensor = load_image_as_tensor(style_path, transform=transform_256)

    # Load content image tensor
    content_path = '../data/images/content/Landscape.jpeg'
    content_tensor = load_image_as_tensor(content_path, transform=transform_256)

    # Style loss should be zero since it's comparing to itself
    print('Testing style tensor match')
    test_loss = LossNetwork().calculate_image_loss(style_tensor, style_tensor, content_tensor)
    assert test_loss[0] == 0
    assert test_loss[1] > 0

    # Content loss should be zero since it's comparing to itself
    print('Testing content tensor match')
    test_loss = LossNetwork().calculate_image_loss(content_tensor, style_tensor, content_tensor)
    assert test_loss[0] > 0
    assert test_loss[1] == 0
