import torch

from image_handler import load_image_as_tensor, transform_256, save_tensors_as_grid, plot_image_tensor
from transfer_network import TransferNetwork
from torchvision import transforms
import os
from abc import ABC
from torch.nn import Softmax
from data_manager import StyleManager

class Interpolator(ABC):

    def __init__(self, total_num_styles, network_parameter_path, device):
        print("Setting up interpolator")
        print(" Using network parameters from", network_parameter_path)
        self.transfer_network = TransferNetwork(total_num_styles)
        self.transfer_network.load_state_dict(torch.load(network_parameter_path))
        self.transfer_network.to(device).eval()
        print(" Ready")

    def render_interpolated_image(self, interpolated_style_parameters, test_image_tensor, style_idx=0):
        self.transfer_network.set_style_parameters(interpolated_style_parameters, style_idx)
        return self.transfer_network(test_image_tensor, style_idx)


class FourStyleInterpolator(Interpolator):

    def __init__(self, total_num_styles, network_parameter_path, device):
        super(FourStyleInterpolator, self).__init__(total_num_styles, network_parameter_path, device)
        self.softax = Softmax()

    def interpolate(self, style_parameters_a, style_parameters_b, style_parameters_c, style_parameters_d, alphas):
        interpolated_weights = []
        interpolated_biases = []
        weights_a, biases_a = style_parameters_a
        weights_b, biases_b = style_parameters_b
        weights_c, biases_c = style_parameters_c
        weights_d, biases_d = style_parameters_d
        #norm_distances = self.softax(torch.tensor(distances))
        #print(norm_distances)
        for i in range(len(weights_a)):
            weights_sum = weights_a[i].mul(alphas[0]) + \
                          weights_b[i].mul(alphas[1]) + \
                          weights_c[i].mul(alphas[2]) + \
                          weights_d[i].mul(alphas[3])
            biases_sum = biases_a[i].mul(alphas[0]) + \
                         biases_b[i].mul(alphas[1]) + \
                         biases_c[i].mul(alphas[2]) + \
                         biases_d[i].mul(alphas[3])
            interpolated_weights.append(weights_sum)
            interpolated_biases.append(biases_sum)
        return interpolated_weights, interpolated_biases

    def run_interpolation(self, style_num_a, style_num_b, style_num_c, style_num_d, grid_dim=5):
        print('Interpolating four styles with grid size %d' % grid_dim)

        style_parameters_a = self.transfer_network.get_style_parameters(style_num_a)
        style_parameters_b = self.transfer_network.get_style_parameters(style_num_b)
        style_parameters_c = self.transfer_network.get_style_parameters(style_num_c)
        style_parameters_d = self.transfer_network.get_style_parameters(style_num_d)

        def dist(x1, y1, x2, y2):
            # Very high distance if on opposite edge
            if abs(x1 - x2) >= (grid_dim - 1) or abs(y1 - y2) >= (grid_dim - 1):
                d = 0
            # Otherwise l1 distance
            else:
                d = float(abs(x1 - x2) + abs(y1 - y2))
                d = 1.0 if d == 0.0 else 1.0/d
            return d

        interpolated_style_parameters_list = []
        for y in range(grid_dim):
            for x in range(grid_dim):
                dist_a = dist(x, y, 0, 0)                        # A in top left
                dist_b = dist(x, y, grid_dim - 1, 0)             # B in top right
                dist_c = dist(x, y, grid_dim - 1, grid_dim - 1)  # C in lower right
                dist_d = dist(x, y, 0, grid_dim - 1)             # D in lower left
                distances = [dist_a, dist_b, dist_c, dist_d]
                interpolated_style_parameters = self.interpolate(style_parameters_a, style_parameters_b,
                                                                 style_parameters_c, style_parameters_d,
                                                                 distances)
                interpolated_style_parameters_list.append(interpolated_style_parameters)

        return interpolated_style_parameters_list

    def produce_interpolated_grid(self, style_tensors, interpolated_style_parameters_list,
                                  test_image_tensor, run_id, grid_dim=5):
        print('Rendering images into grid')

        # Load interpolated images
        output_images = []
        for interpolated_style_parameters in interpolated_style_parameters_list:
            output_images.append(self.render_interpolated_image(interpolated_style_parameters, test_image_tensor))

        # Insert style images
        black_image = torch.zeros([3, 256, 256]).to(_device)
        grid = []
        for y in range(grid_dim):
            for x in range(grid_dim + 2):
                if x == 0:
                    if y == 0:
                        grid.append(style_tensors[0])
                    elif y == (grid_dim - 1):
                        grid.append(style_tensors[3])
                    else:
                        grid.append(black_image)
                elif x == grid_dim + 1:
                    if y == 0:
                        grid.append(style_tensors[1])
                    elif y == (grid_dim - 1):
                        grid.append(style_tensors[2])
                    else:
                        grid.append(black_image)
                else:
                    grid.append(output_images[(x - 1) + y * grid_dim])

        interpolation_dir = '../data/interpolation/'
        if not os.path.exists(interpolation_dir):
            os.makedirs(interpolation_dir)

        path = os.path.join(interpolation_dir, run_id + '_four.png')
        print(' Saving to', path)
        save_tensors_as_grid(grid, path, nrow=grid_dim+2)
        plot_image_tensor(load_image_as_tensor(path, transform=transforms.ToTensor()))
        print(' Done')


class TwoStyleInterpolator(Interpolator):

    def __init__(self, total_num_styles, network_parameter_path, device):
        super(TwoStyleInterpolator, self).__init__(total_num_styles, network_parameter_path, device)

    def interpolate(self, style_parameters_a, style_parameters_b, alpha):
        interpolated_weights = []
        interpolated_biases = []
        weights_a, biases_a = style_parameters_a
        weights_b, biases_b = style_parameters_b
        for i in range(len(weights_a)):
            interpolated_weights.append(weights_a[i].mul(1 - alpha) + weights_b[i].mul(alpha))
            interpolated_biases.append(biases_a[i].mul(1 - alpha) + biases_b[i].mul(alpha))
        return interpolated_weights, interpolated_biases

    def run_interpolation(self, style_num_a, style_num_b, step=0.25):
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

    def produce_interpolated_grid(self, interpolated_style_parameters_list, test_image_tensor, run_id, grid_dim=5):
        print('Rendering images into grid')
        output_images = []
        for interpolated_style_parameters in interpolated_style_parameters_list:
            output_images.append(self.render_interpolated_image(interpolated_style_parameters, test_image_tensor))

        interpolation_dir = '../data/interpolation/'
        if not os.path.exists(interpolation_dir):
            os.makedirs(interpolation_dir)
        path = os.path.join(interpolation_dir, run_id + '_two.png')
        print(' Saving to', path)
        save_tensors_as_grid(output_images, path, nrow=len(output_images))
        plot_image_tensor(load_image_as_tensor(path, transform=transforms.ToTensor()))
        print(' Done')


if __name__ == '__main__':
    _device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _run_id = '0006'

    _test_image_tensor = load_image_as_tensor('../data/images/content/venice.jpeg', transform=transform_256)
    _test_image_tensor = _test_image_tensor.unsqueeze(0).to(_device)

    _total_num_styles = 5
    _network_parameter_path = '../data/networks/model_parameters/' + _run_id

    # Two style
    _interpolator = TwoStyleInterpolator(_total_num_styles, _network_parameter_path, _device)
    _interpolated_style_parameters_list = _interpolator.run_interpolation(1, 2)
    _interpolator.produce_interpolated_grid(_interpolated_style_parameters_list, _test_image_tensor, _run_id)

    # Four style
    style_dir = '../data/images/style/'
    style_manager = StyleManager(style_dir, _device)
    style_idxs = [11, 23, 17, 24]
    network_style_idxs = [1, 2, 3, 4]
    _style_tensors = style_manager.get_style_tensor_subset(style_idxs)
    _interpolator = FourStyleInterpolator(_total_num_styles, _network_parameter_path, _device)
    _interpolated_style_parameters_list = _interpolator.run_interpolation(*network_style_idxs)
    _interpolator.produce_interpolated_grid(_style_tensors, _interpolated_style_parameters_list,
                                            _test_image_tensor, _run_id)
