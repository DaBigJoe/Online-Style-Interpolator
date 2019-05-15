import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data_manager import Dataset, StyleManager
from src.image_handler import load_image_as_tensor, normalise_batch
from src.loss_network import LossNetwork
from src.transfer_network import TransferNetwork
from tqdm import tqdm


def train():
    # Args
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    image_dir = '/home/data/train2014/'
    style_dir = '../data/images/style/'
    batch_size = 4
    num_parameter_updates = 100  # TODO change to parameter updates
    content_weight = 1e5
    style_weight = 1e10
    style_idxs = [0, 3]

    # Load dataset
    train_dataset = Dataset(image_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # to provide a batch loader

    # Load styles
    style_manager = StyleManager(style_dir, device)
    style_tensors = style_manager.get_style_tensor_subset(style_idxs)
    style_num = len(style_idxs)

    # Setup transfer network
    transfer_network = TransferNetwork(style_num).to(device)
    transfer_network.train()
    optimizer = Adam(transfer_network.parameters(), lr=1e-3)

    # Setup loss network
    loss_network = LossNetwork(normalise_batch(style_tensors), device)

    update_count = 0  # Number of parameter updates that have occurred
    with tqdm(total=num_parameter_updates, ncols=120) as progress_bar:
        while update_count < num_parameter_updates:
            for _, x in enumerate(train_loader):
                if update_count >= num_parameter_updates:
                    break

                # Begin optimisation step
                optimizer.zero_grad()

                # Get style for this step
                style_idx = update_count % style_num

                # Perform image transfer and normalise
                y = transfer_network(x.to(device), style_idx=style_idx)
                x = normalise_batch(x).to(device)
                y = normalise_batch(y).to(device)

                # Calculate loss
                content_loss, style_loss = loss_network.calculate_loss(x, y, style_idx)
                content_loss *= content_weight
                style_loss *= style_weight
                total_loss = content_loss + style_loss

                # Backprop
                total_loss.backward()
                optimizer.step()

                # Update tqdm bar
                progress_bar.update(1)
                progress_bar.set_postfix(style_loss="%.0f" % style_loss,
                                         content_loss="%.0f" % content_loss)

                # Step
                update_count += 1


# def stylize(args):
#     device = torch.device("cuda" if args.cuda else "cpu")
#
#     content_image = utils.load_image(args.content_image, scale=args.content_scale)
#     content_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: x.mul(255))
#     ])
#     content_image = content_transform(content_image)
#     content_image = content_image.unsqueeze(0).to(device)
#
#     with torch.no_grad():
#         style_model = TransformerNet(style_num=args.style_num)
#         state_dict = torch.load(args.model)
#         style_model.load_state_dict(state_dict)
#         style_model.to(device)
#         output = style_model(content_image, style_id=[args.style_id]).cpu()
#
#     utils.save_image('output/' + args.output_image + '_style' + str(args.style_id) + '.jpg', output[0])

if __name__ == "__main__":
    train()
