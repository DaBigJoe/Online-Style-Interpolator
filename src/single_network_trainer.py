"""

Author: Alexander Stonard (ads1g15@soton.ac.uk)
Created: 23/05/19
"""

from os import listdir

import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils import data
from torch.utils.data import DataLoader
from torch import optim

import torchbearer

from loss_network import TruncatedVgg16
from image_handler import load_image_as_tensor, save_tensor_as_image, plot_image_tensor, transform_256
from transfer_network_single import TransferNetworkSingle
import torch.multiprocessing as mp

class LossNetwork:

    def __init__(self):
        self.model = TruncatedVgg16().cuda()

        self.model.eval()
        for p in self.model.parameters():
            p.require_grads = False

    def get_style(self, tensor):
        return self.model(tensor)

    def calculate_image_loss(self, transformed_tensor, original_tensor, style_tensor):
        """
        Calculate the style loss and content loss of an input image compared to a target style image and a
         target content image.
        """
        # Forward pass each image tensor and extract outputs
        predicted_outputs = self.model(transformed_tensor)
        style_target_outputs = style_tensor
        content_target_outputs = self.model(original_tensor)

        # Compute and return style loss and content loss
        style_loss = self._style_loss(predicted_outputs, style_target_outputs)
        content_loss = self._content_loss(predicted_outputs, content_target_outputs)
        return style_loss, content_loss

    def _content_loss(self, predicted_outputs, target_outputs):
        """
        Calculate the content loss between a set of predicted outputs and a set of target content outputs.
        """
        # Use output from relu2_2 (second item in target outputs from TruncatedVgg16 model)
        shape = torch.tensor(target_outputs[1].shape)
        dist = torch.norm(predicted_outputs[1] - target_outputs[1])
        loss = dist.pow(2).sum().div(shape.prod())
        return loss


    def _style_loss(self, predicted_outputs, target_outputs):
        """
        Calculate the style loss between a set of predicted outputs and a set of target style outputs.
        """

        def gram_matrix(m):
            m = m.squeeze(0)
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
            dist = torch.norm(predicted_gram - target_gram, 'fro')
            shape = torch.tensor(predicted_gram.shape)
            loss += dist.pow(2) / shape.prod()

        return loss

def loss_calculator(x, y):
    global loss_net
    original_tens, target_style_tens = y

    new_tensor = x.add(x)
    new_tensor = torch.clamp(new_tensor, min=0, max=255)

    reg_loss = (
            torch.sum(torch.abs(new_tensor[:, :, :, :-1] - new_tensor[:, :, :, 1:])) +
            torch.sum(torch.abs(new_tensor[:, :, :-1, :] - new_tensor[:, :, 1:, :]))
    )

    style_loss, content_loss = loss_net.calculate_image_loss(new_tensor, original_tens, target_style_tens)

    l1 = torch.tensor([1.0])
    l2 = torch.tensor([1E4])
    l3 = torch.tensor([1E-6])

    content_loss = content_loss.detach() * l1
    style_loss = style_loss.detach() * l2
    reg_loss = reg_loss.detach() * l3

    loss = content_loss.add(style_loss.add(reg_loss))
    return loss.requires_grad_(True)

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, dir, target_style_tensor):
        self.dir = dir
        self.items = listdir(dir)

        self.target_style_tensor = target_style_tensor

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.items)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        path = self.items[index]

        # Load data and get label
        X = load_image_as_tensor(self.dir + path, transform=transform_256).squeeze(0)

        if not (len(X) == 3):
            _X = torch.zeros(3, X.shape[1], X.shape[2])
            _X[0] = X
            _X[1] = X
            _X[2] = X
            X = _X

        y = (X, self.target_style_tensor)

        return X, y

class SINGLE_TRAINER:

    def __init__(self, training_dir, style_image_dir, model_save_path):
        self.training_dir = training_dir
        self.style_image = load_image_as_tensor(style_image_dir, transform=transform_256).to("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_save_path = model_save_path

        self.model = TransferNetworkSingle()

        global loss_net

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mp.set_start_method('spawn')

        target_style_tensor = loss_net.get_style(self.style_image)
        training_set = Dataset(self.training_dir, target_style_tensor)

        train_loader = DataLoader(training_set, batch_size=4, shuffle=True)

        loss_function = loss_calculator
        optimiser = optim.Adam(self.model.parameters(), lr=1E-3)

        self.trial = torchbearer.Trial(self.model, optimiser, loss_function, metrics=[]).to(device)
        self.trial.with_generators(train_loader)

    def train(self):
        print('Running Training')
        self.trial.run(epochs=2)
        print('Finished Training')

        print('Saving')
        torch.save(self.model.state_dict(), self.model_save_path)
        print('Saved')

    def forward(self, img):
        return self.model(img).detach()

    def evaluate(self):
        print('Evaluating using training data')
        results = self.trial.evaluate(data_key=torchbearer.TRAIN_DATA)
        print(results)
        print('Evaluated')


training_images_dir = '../data/coco/'
style_image_dir = '../data/images/style/Van_Gogh_Starry_Night.jpg'
model_save_path = '../data/networks/model_parameters/transfer_network_single.dat'

print('INIT')
loss_net = LossNetwork()

st = SINGLE_TRAINER(training_images_dir, style_image_dir, model_save_path)

print('STARTING TRAINING')
st.train()

print('STARTING EVALUATION')
st.evaluate()

print('ALL DONE')