"""
Loss network implementation.

Uses a pre-trained network (VGG-16 trained on ImageNet) to evaluate the loss of a transformed image with reference
style and content images.
"""

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import vgg16

from image_handler import load_image_as_tensor


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
        features = list(vgg16(pretrained=True).features)[:self.target_layers[-1]]
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        outputs = []
        for layer_index, feature in enumerate(self.features):
            x = feature(x)
            if layer_index in self.target_layers:
                outputs.append(x)
        return outputs


class LossNetwork:

    def __init__(self):
        self.model = TruncatedVgg16()
        # Lock model down
        self.model.eval()
        # Don't need to use gradients as not doing back prop.
        for p in self.model.parameters():
            p.require_grads = False

    def calculate_image_loss(self, transformed_tensor, style_tensor, content_tensor):
        # Don't need to use gradients as not doing back prop.
        with torch.no_grad():
            predicted_outputs = self.model(transformed_tensor)
            style_loss = self.style_loss(predicted_outputs, style_tensor)
            content_loss = self.content_loss(predicted_outputs, content_tensor)
            return style_loss, content_loss

    def style_loss(self, predicted_outputs, target_tensor):
        target_outputs = self.model(target_tensor)
        return 0

    def content_loss(self, predicted_outputs, target_tensor):
        target_outputs = self.model(target_tensor)

        # Use output from relu2_2 (second item in target outputs from TruncatedVgg16 model)
        shape = torch.tensor(target_outputs[1].shape)
        dist = torch.norm(predicted_outputs[1] - target_outputs[1], 2)
        dist = dist.pow(2)
        loss = dist.div(shape.prod(0))
        return loss


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()  # convert to tensor
    ])

    style_path = '../data/images/style/Van_Gogh_Starry_Night.jpg'
    style_tensor = load_image_as_tensor(style_path, transform=transform)

    content_path = '../data/images/content/Landscape.jpeg'
    content_tensor = load_image_as_tensor(content_path, transform=transform)

    print(LossNetwork().calculate_image_loss(content_tensor, style_tensor, content_tensor))
