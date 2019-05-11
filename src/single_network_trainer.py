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

from tqdm import tqdm
import torchbearer

from loss_network import TruncatedVgg16
from loss_network import LossNetwork
from image_handler import load_image_as_tensor, save_tensor_as_image, plot_image_tensor, transform_256
from transfer_network_single import TransferNetworkSingle
import torch.multiprocessing as mp

# This acts as the loss function in torchbearer
def loss_calculator(x, y):
    global loss_net # Global variable

    # y is not really a target, it holds the tesnors
    #  that are used to calculate the loss instead
    original_tens, target_style_tens = y

    style_loss, content_loss = loss_net.calculate_image_loss(x, target_style_tens, original_tens)

    style_weight = 1e12
    content_weight = 1e5
    #print((style_loss, content_loss))

    style_loss = style_loss.mul(style_weight)
    content_loss = content_loss.mul(content_weight)

    loss = content_loss.add(style_loss)

    return loss

# Reads the data from the data source. This is either directly
#  from files, or from memory. We haven't decided the best
#  way to do it yet
class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'

  def __init__(self, dir, target_style_tensor):
        self.data = []

        items = listdir(dir)

        self.dir = dir
        self.data = items

        #ignore_runner = 0

        # Variable step, only sample 1 in check_step images TODO remove for final
        #check_step = 100
        #print("Check interval is", check_step)

        #for _, path in enumerate(tqdm(items)):
        #    if(ignore_runner % check_step == 0):
        #        self.data.append(load_image_as_tensor(dir + path, transform=transform_256).squeeze(0))
        #    ignore_runner += 1

        self.target_style_tensor = target_style_tensor


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # path = self.items[index]

        # Load data and get label
        #X = self.data[index]
        X = load_image_as_tensor(self.dir + self.data[index], transform=transform_256).squeeze(0)

        # load_image_as_tensor(self.dir + path, transform=transform_256).squeeze(0)
        if not (len(X) == 3):
            _X = torch.zeros((3, X.shape[1], X.shape[2]))
            _X[0] = X[0]
            _X[1] = X[0]
            _X[2] = X[0]
            X = _X

        y = (X, self.target_style_tensor.squeeze(0))

        return X, y

class SINGLE_TRAINER:

    def __init__(self, training_dir, style_image_dir, model_save_path):
        self.training_dir = training_dir
        self.style_image = load_image_as_tensor(style_image_dir, transform=transform_256).to("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_save_path = model_save_path

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Store model on GPU for better performance
        self.model = TransferNetworkSingle().to(device)

        global loss_net

        mp.set_start_method('spawn')
        self.model.share_memory()

        target_style_tensor = self.style_image

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


training_images_dir = '../data/coco/' #'/home/data/train2014/'
style_image_dir = '../data/images/style/Van_Gogh_Starry_Night.jpg'
model_save_path = '../data/networks/model_parameters/transfer_network_single.dat'

print('INIT')
loss_net = LossNetwork()

st = SINGLE_TRAINER(training_images_dir, style_image_dir, model_save_path)

print('STARTING TRAINING')
st.train()

#print('STARTING EVALUATION')
#st.evaluate()

print('ALL DONE')