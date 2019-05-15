import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_manager import Dataset, StyleManager
from image_handler import normalise_batch, load_image_as_tensor, save_tensor_as_image
from loss_network import LossNetwork
from transfer_network import TransferNetwork


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def train():
    # Args
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    image_dir = '/home/data/train2014/'
    style_dir = '../data/images/style/'
    checkpoint_dir = '../data/checkpoints/'
    stats_dir = '../data/stats/'
    model_dir = '../data/networks/model_parameters'
    test_image_path = '../data/images/content/venice.jpeg'
    batch_size = 4
    num_parameter_updates = 40000
    content_weight = 1e5
    style_weight = 1e10
    style_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
    checkpoint_freq = 100

    # Ensure save directories exist
    check_dir(checkpoint_dir)
    check_dir(stats_dir)
    check_dir(model_dir)

    # Get unique run id
    unique_run_id = "{:04d}".format(len([i for i in os.listdir(checkpoint_dir)
                                         if os.path.isdir(os.path.join(checkpoint_dir, i))]) + 1)
    print('Starting run', unique_run_id)

    # Load dataset
    train_dataset = Dataset(image_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # to provide a batch loader
    test_image_tensor = load_image_as_tensor(test_image_path).unsqueeze(0).to(device)

    # Load styles
    style_manager = StyleManager(style_dir, device)
    style_tensors = style_manager.get_style_tensor_subset(style_idxs)
    style_num = len(style_idxs)
    print('Training on', style_num, 'styles for', num_parameter_updates, 'parameter updates')

    # Setup transfer network
    transfer_network = TransferNetwork(style_num).to(device)
    transfer_network.train()
    optimizer = Adam(transfer_network.parameters(), lr=1e-3)

    # Setup loss network
    loss_network = LossNetwork(normalise_batch(style_tensors), device)

    # Setup check pointing
    checkpoint_path = os.path.join(checkpoint_dir, unique_run_id)
    os.makedirs(checkpoint_path)
    print('Saving checkpoints to', checkpoint_path)

    # Setup model saving
    model_save_path = os.path.join(model_dir, unique_run_id)
    print('Saving model to', model_save_path)

    # Setup logging
    stats_path = stats_dir + 'stats' + unique_run_id + '.csv'
    stats_file = open(stats_path, 'w+')
    stats_file.write(str(num_parameter_updates) + ', ' + str(style_num) + '\n')
    stats_file.write(style_manager.get_style_tensor_name(style_idxs[0]))
    for i in range(1, style_num):
        stats_file.write(' ' + style_manager.get_style_tensor_name(style_idxs[0]))
    stats_file.write('\n')
    print('Saving stats to', stats_path)

    update_count = 0  # Number of parameter updates that have occurred
    checkpoint = 0  # Current checkpoint
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

                # Checkpoint
                if update_count % checkpoint_freq == 0:
                    checkpoint_file_path = os.path.join(checkpoint_path, str(checkpoint + 1) + '.jpeg')
                    test_output = transfer_network(test_image_tensor, 0)[0]
                    save_tensor_as_image(test_output.detach().cpu(), checkpoint_file_path)
                    checkpoint += 1

                # Update tqdm bar
                progress_bar.update(1)
                progress_bar.set_postfix(checkpoint=checkpoint,
                                         style_loss="%.0f" % style_loss,
                                         content_loss="%.0f" % content_loss)

                # Record loss in CSV file
                stats_file.write(str(update_count) + ', ' + str(style_loss.item()) + ', ' + str(content_loss.item()) + '\n')

                # Step
                update_count += 1

    # Finish stats
    stats_file.close()

    # Save model
    torch.save(transfer_network.state_dict(), model_save_path)


if __name__ == "__main__":
    train()
