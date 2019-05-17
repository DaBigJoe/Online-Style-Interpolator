import os

import torch

from data_manager import StyleManager
from image_handler import load_image_as_tensor, save_tensor_as_image, save_tensors_as_grid
from transfer_network import TransferNetwork


class Renderer:

    def __init__(self, model_path, save_dir, num_styles):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.save_dir = save_dir
        self.num_styles = num_styles
        self.style_manager = StyleManager('../data/images/style/', self.device)

        self.transfer_network = TransferNetwork(num_styles).to(self.device)
        if self.device == "cpu":
            self.transfer_network.load_state_dict(torch.load(model_path, map_location='cpu'))
        else:
            self.transfer_network.load_state_dict(torch.load(model_path))
        self.transfer_network.eval()

    def render(self, content_image, style_idx):
        content_image = content_image.to(self.device)
        output = self.transfer_network(content_image, style_idx=style_idx)
        output = output.detach().cpu()
        return output[0]

    def render_grid(self, content_dir, style_idx, style_id):
        rendered_images = [self.style_manager[style_id]]
        save_path = os.path.join(self.save_dir, 'style_grid_%d.png' % style_idx)
        for content_path in os.listdir(content_dir):
            content_image = load_image_as_tensor(os.path.join(content_dir, content_path)).unsqueeze(0)
            rendered_images.append(self.render(content_image, style_idx))
        save_tensors_as_grid(rendered_images, save_path, nrow=4)

    def render_all(self, content_image):
        for style_idx in range(self.num_styles):
            self.render_single(content_image, style_idx)

    def render_single(self, content_image, style_idx):
        save_path = os.path.join(self.save_dir, 'style_%d.png' % style_idx)
        output = self.render(content_image, style_idx)
        save_tensor_as_image(output, save_path)


if __name__ == "__main__":
    model_id = '0004'
    _model_path = '../data/networks/model_parameters/' + model_id
    _content_dir = '../data/images/content/'
    _save_dir = '../data/rendered/' + model_id
    if not os.path.exists(_save_dir):
        os.makedirs(_save_dir)
    _num_styles = 10
    style_ids = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]  # [22, 11, 23, 17, 24]
    renderer = Renderer(_model_path, _save_dir, _num_styles)
    for i in range(len(style_ids)):
        print('Rendering grid for style', i)
        renderer.render_grid(_content_dir, i, style_ids[i])
