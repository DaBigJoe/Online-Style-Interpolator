import os
import torch

from image_handler import load_image_as_tensor, save_tensor_as_image
from transfer_network import TransferNetwork


class Renderer:

    def __init__(self, image_path, model_path, save_dir, num_styles):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.image_tensor = load_image_as_tensor(image_path).unsqueeze(0).to(self.device)
        self.save_dir = save_dir
        self.num_styles = num_styles

        self.transfer_network = TransferNetwork(num_styles).to(self.device)
        self.transfer_network.load_state_dict(torch.load(model_path))
        self.transfer_network.eval()

    def render_all(self):
        for style_idx in range(self.num_styles):
            self.render_single(style_idx)

    def render_single(self, style_idx):
        save_path = os.path.join(self.save_dir, 'style_%d.png' % style_idx)
        output = self.transfer_network(self.image_tensor, style_idx=style_idx)
        save_tensor_as_image(output.detach().cpu()[0], save_path)


if __name__ == "__main__":
    model_id = '0002'
    _image_path = '../data/images/content/venice.jpeg'
    _model_path = '../data/networks/model_parameters/' + model_id
    _save_dir = '../data/rendered/' + model_id
    if not os.path.exists(_save_dir):
        os.makedirs(_save_dir)
    _num_styles = 2
    renderer = Renderer(_image_path, _model_path, _save_dir, _num_styles)
    renderer.render_all()
