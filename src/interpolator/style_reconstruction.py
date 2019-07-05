"""
Style Reconstruction implementation.

Create an image that satisfies the stylistic constraints pulled by VGG16 at layers
0,5,10,17,24 : Conv1_1 (0), Conv2_1 (5), Conv3_1 (10), Conv4_1 (17), Conv5_1 (24)

Author: Jamie Sian (js17g15@soton.ac.uk)
Created: 10/05/19
"""

import datetime
import os

from tqdm import tqdm

import csv
import torch
from torch import optim, randn
from src.interpolator.loss_network import LossNetwork
from src.interpolator.image_handler import save_tensor_as_image, plot_image_tensor, load_image_as_tensor, transform_256


class StyleLearnerSingle:

    def __init__(self, layer_num, style_path, learn_rate, num_epochs):
        super(StyleLearnerSingle, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.layer_idx = layer_num
        self.style_path = style_path
        self.learn_rate = learn_rate
        self.num_epochs = num_epochs

        # Load style as tensor
        self.style_tensor = load_image_as_tensor(self.style_path).unsqueeze(0).to(self.device)
        # Load loss network
        self.loss_network = LossNetwork(self.style_tensor, self.device)

    def train(self):
        # 3 channel, 256^2px image
        noise = torch.randn(1, 3, 256, 256, device=self.device, requires_grad=False)

        # Scale and shift standard normal distribution to speed training
        noise = torch.mul(noise, 0.5)
        noise = torch.add(noise, 0.5)
        noise = torch.tensor(noise, requires_grad=True)

        optimiser = optim.Adam([noise], lr=self.learn_rate)
        print("Here I go training again!")

        min_loss = 1e10 + 1
        min_noise = noise
        interrupt_flag = False

        # Allow for trial interrupt and save of best
        try:
            # For each epoch
            with tqdm(range(self.num_epochs)) as progress_bar:
                for _ in progress_bar:
                    optimiser.zero_grad()
                    noise_features = self.loss_network.model(noise)
                    loss = self.loss_network.style_loss_single(noise_features, 0, self.layer_idx) * 1e12

                    # Backprop step
                    loss.backward()
                    optimiser.step()
                    loss_str = "%.5f" % loss
                    progress_bar.set_postfix(error=loss_str)

                    if loss < min_loss:
                        min_loss = loss
                        min_noise = noise.clone()

        except KeyboardInterrupt:
            print('interrupted, wait for save dialogue')
            interrupt_flag = True
        finally:
            save_directory = "../data/images/style_analysis/" + str(self.layer_idx)
            print("Final loss for this run is:", str(min_loss.item()))
            input_image = min_noise.detach()

            # Save image
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            file_name_pre = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            file_name_post = '-loss-' + str(min_loss.item())
            file_name = file_name_pre + file_name_post + '.jpg'
            file_path = os.path.join(save_directory, file_name)
            save_tensor_as_image(input_image, file_path)

            # Check if finally is invoked by interrupt
            if interrupt_flag:
                exit(0)


class StyleLearnerCumulative:

    def __init__(self, layer_idx, style_path, learn_rate, num_epochs):
        super(StyleLearnerCumulative, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.layer_idx = layer_idx
        self.style_path = style_path
        self.learn_rate = learn_rate
        self.num_epochs = num_epochs
        self.plots = []

        for i in range(layer_idx + 2):
            self.plots.append([])

        # Load style as tensor
        self.style_tensor = load_image_as_tensor(self.style_path).unsqueeze(0).to(self.device)
        # Load loss network
        self.loss_network = LossNetwork(self.style_tensor, self.device)

    def train(self):
        # 3 channel, 256^2px image
        noise = torch.randn(1, 3, 256, 256, device=self.device, requires_grad=False)

        # Scale and shift standard normal distribution to speed training
        noise = torch.mul(noise, 0.5)
        noise = torch.add(noise, 0.5)
        noise = torch.tensor(noise, requires_grad=True)

        optimiser = optim.Adam([noise], lr=self.learn_rate)
        print("Here i go training again!")

        min_loss = 1e10 + 1
        min_noise = noise
        interrupt_flag = False

        # Allow for trial interrupt and save of best
        try:
            # For each epoch
            with tqdm(range(self.num_epochs)) as progress_bar:
                for _ in progress_bar:
                    # for epoch in range(num_epochs):
                    optimiser.zero_grad()
                    style_target_output = self.loss_network.model(self.style_tensor.cuda())
                    noise_copy = noise
                    predicted_output = self.loss_network.model(noise_copy)

                    optimiser.zero_grad()
                    noise_features = self.loss_network.model(noise)

                    loss = 0
                    # Add loss of style layers, and store for plotting
                    for x_i in range(self.layer_idx + 1):
                        layer_loss = self.loss_network.style_loss_single(noise_features, 0, x_i) * 1e12
                        self.plots[x_i].append(layer_loss.item())
                        loss += layer_loss

                    # And add cumulative loss
                    self.plots[self.layer_idx + 1].append(loss.item())

                    # Backprop step
                    loss.backward()
                    optimiser.step()
                    loss_str = "%.5f" % loss
                    progress_bar.set_postfix(error=loss_str)

                    if loss < min_loss:
                        min_loss = loss
                        min_noise = noise_copy.clone()

        except KeyboardInterrupt:
            print('interrupted, wait for save dialogue')
            interrupt_flag = True
        finally:
            save_directory = "../data/images/style_analysis/" + str(self.layer_idx)
            print("final loss for this run is:", str(min_loss.item()))
            input_image = min_noise.detach()

            # Save image
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            file_name_pre = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            file_name_post = '-loss-' + str(min_loss.item())
            file_name = file_name_pre + file_name_post + '.jpg'
            csv_name = file_name_pre + '-' + str(self.layer_idx) + '.csv'

            file_path = os.path.join(save_directory, file_name)
            csv_path = os.path.join(save_directory, csv_name)

            print("Writing image")
            save_tensor_as_image(input_image, file_path)

            print("Writing csv")
            with open(csv_path, "w+") as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerows(self.plots)

            # Check if finally is invoked by interrupt
            if interrupt_flag:
                exit(0)


if __name__ == '__main__':
    style_path = '../data/images/style/Van_Gogh_Starry_Night.jpg'

    # learn_rates = [0.001, 0.005, 0.005, 0.01]
    learn_rates = [0.001, 0.001, 0.001, 0.001, 0.001]
    num_epochs = 5000
    # learn_rates = [0.1, 0.1, 0.1, 0.1]

    for i in range(5):
        if i != 3:
            continue

        print("i =", i)
        StyleLearnerCumulative(i, style_path, learn_rates[i], num_epochs).train()