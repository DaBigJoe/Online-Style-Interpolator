

import torch
from image_handler import load_image_as_tensor, save_tensor_as_image, plot_image_tensor, transform_256
from transfer_network_single import TransferNetworkSingle



model_save_path = '../data/networks/model_parameters/transfer_network_single.dat'
test_image_path = '../data/images/content/Landscape.jpeg'

model = TransferNetworkSingle()
model.load_state_dict(torch.load(model_save_path))

test_image = load_image_as_tensor(test_image_path, transform=transform_256)
test_output = model(test_image)

plot_image_tensor(test_image)
plot_image_tensor(test_output.detach())
