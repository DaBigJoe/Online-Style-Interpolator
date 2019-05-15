import torch

from image_handler import load_image_as_tensor, transform_256, save_tensors_as_grid, plot_image_tensor
from transfer_network import TransferNetwork
from torchvision import transforms
import os


class TwoStyleInterpolator:

    def __init__(self, total_num_styles, network_parameter_path, device):
        print("Setting up interpolator")
        print(" Using network parameters from", network_parameter_path)
        self.transfer_network = TransferNetwork(total_num_styles)
        self.transfer_network.load_state_dict(torch.load(network_parameter_path))
        self.transfer_network.to(device).eval()
        print(" Ready")

    def interpolate(self, style_parameters_a, style_parameters_b, alpha):
        interpolated_weights = []
        interpolated_biases = []
        weights_a, biases_a = style_parameters_a
        weights_b, biases_b = style_parameters_b
        for i in range(len(weights_a)):
            interpolated_weights.append(weights_a[i].mul(alpha) + weights_b[i].mul(1 - alpha))
            interpolated_biases.append(biases_a[i].mul(alpha) + biases_b[i].mul(1 - alpha))
        return interpolated_weights, interpolated_biases

    def run_interpolation(self, style_num_a, style_num_b, step=0.2):
        assert (1/step).is_integer()
        print('Interpolating styles %d and %d with step size %.2f' % (style_num_a, style_num_b, step))

        style_parameters_a = self.transfer_network.get_style_parameters(style_num_a)
        style_parameters_b = self.transfer_network.get_style_parameters(style_num_b)

        interpolated_style_parameters_list = []
        alpha = 0
        while alpha <= 1:
            interpolated_style_parameters = self.interpolate(style_parameters_a, style_parameters_b, alpha)
            interpolated_style_parameters_list.append(interpolated_style_parameters)
            alpha += step

        return interpolated_style_parameters_list

    def render_interpolated_image(self, interpolated_style_parameters, test_image_tensor, style_idx=0):
        self.transfer_network.set_style_parameters(interpolated_style_parameters, style_idx)
        return self.transfer_network(test_image_tensor, style_idx)

    def produce_interpolated_grid(self, interpolated_style_parameters_list, test_image_tensor, run_id):
        print('Rendering images into grid')
        output_images = []
        for interpolated_style_parameters in interpolated_style_parameters_list:
            output_images.append(self.render_interpolated_image(interpolated_style_parameters, test_image_tensor))
        interpolation_dir = '../data/interpolation/'
        if not os.path.exists(interpolation_dir):
            os.makedirs(interpolation_dir)
        path = os.path.join(interpolation_dir, run_id + '.png')
        print(' Saving to', path)
        save_tensors_as_grid(output_images, path, nrow=len(output_images))
        plot_image_tensor(load_image_as_tensor(path, transform=transforms.ToTensor()))
        print(' Done')


if __name__ == '__main__':
    _device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _run_id = '0002'

    _test_image_tensor = load_image_as_tensor('../data/images/content/venice.jpeg', transform=transform_256)
    _test_image_tensor = _test_image_tensor.unsqueeze(0).to(_device)

    _total_num_styles = 2
    _network_parameter_path = '../data/networks/model_parameters/' + _run_id
    _interpolator = TwoStyleInterpolator(_total_num_styles, _network_parameter_path, _device)
    _interpolated_style_parameters_list = _interpolator.run_interpolation(0, 1)
    _interpolator.produce_interpolated_grid(_interpolated_style_parameters_list, _test_image_tensor, _run_id)
