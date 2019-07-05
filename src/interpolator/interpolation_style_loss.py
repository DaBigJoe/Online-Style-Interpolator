"""
Find the style loss due to interpolation of images.
Produces a plot comparing the style loss when interpolating between two images.

Author: Joseph Early (je5g15@soton.ac.uk)
Created: 16/05/19
"""

import numpy as np
import torch
from src.interpolator.data_manager import StyleManager
from src.interpolator.interpolate_styles import TwoStyleInterpolator
from src.interpolator.loss_network import LossNetwork
from matplotlib import pyplot as plt

from src.interpolator.image_handler import load_image_as_tensor, transform_256

if __name__ == '__main__':
    _device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _run_id = '0006'

    _test_image_tensor = load_image_as_tensor('../data/images/content/venice.jpeg', transform=transform_256)
    _test_image_tensor = _test_image_tensor.unsqueeze(0).to(_device)

    _total_num_styles = 5
    _network_parameter_path = '../data/networks/model_parameters/' + _run_id

    _interpolator = TwoStyleInterpolator(_total_num_styles, _network_parameter_path, _device)
    style_a = 2
    style_b = 4
    step = 0.05
    steps = np.linspace(0, 1, int(1/step), endpoint=True)
    _interpolated_style_parameters_list = _interpolator.run_interpolation(style_a, style_b, step=0.05)

    style_dir = '../data/images/style/'
    style_manager = StyleManager(style_dir, _device)
    style_idxs = [22, 11, 23, 17, 24]
    style_tensors = style_manager.get_style_tensor_subset(style_idxs)
    loss_network = LossNetwork(style_tensors, _device)

    losses_a = []
    losses_b = []
    for i in range(len(_interpolated_style_parameters_list)):
        interpolated_style_parameters = _interpolated_style_parameters_list[i]
        image_tensor = _interpolator.render_interpolated_image(interpolated_style_parameters, _test_image_tensor)
        image_features = loss_network.model(image_tensor)
        loss_a = loss_network.style_loss(image_features, style_a).detach().cpu().numpy() * 1e10
        loss_b = loss_network.style_loss(image_features, style_b).detach().cpu().numpy() * 1e10
        losses_a.append(loss_a)
        losses_b.append(loss_b)

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['font.size'] = 12
    fig, axis = plt.subplots(1, 1, figsize=(4, 5))
    axis.plot(steps, losses_a, label='Style A')
    axis.plot(steps, losses_b, label='Style B')
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.2e15)
    axis.set_xlabel(r'$\alpha$', fontsize=16)
    axis.set_ylabel('Interpolated Style Loss')
    axis.legend(loc='best')
    plt.tight_layout()
    plt.show()

    fig.savefig('../data/interpolation_style_loss.eps', format='eps', dpi=1000, bbox_inches='tight')
