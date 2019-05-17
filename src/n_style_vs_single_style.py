from matplotlib import pyplot as plt
import csv
import numpy as np


def get_final_losses_single_network(stats_file_path):
    with open(stats_file_path, 'r') as stats_file:
        reader = csv.reader(stats_file)
        # Skip headers
        next(reader)
        next(reader)
        for row in reader:
            final_style_loss = float(row[1])
            final_content_loss = float(row[2])
    return final_style_loss, final_content_loss


def get_final_losses_n_network(stats_file_path, num_styles):
    with open(stats_file_path, 'r') as stats_file:
        reader = csv.reader(stats_file)
        # Skip headers
        next(reader)
        next(reader)
        style_losses = np.zeros(num_styles)
        content_losses = np.zeros(num_styles)
        style_idx = 0
        for row in reader:
            style_losses[style_idx] = float(row[1])
            content_losses[style_idx] = float(row[2])
            style_idx += 1
            if style_idx == num_styles:
                style_idx = 0

    return style_losses, content_losses


if __name__ == '__main__':
    # Number of styles
    n = 5

    # Stats paths
    single_path_0 = '../data/stats/stats0007.csv'
    single_path_1 = '../data/stats/stats0008.csv'
    single_path_2 = '../data/stats/stats0009.csv'
    single_path_3 = '../data/stats/stats0002.csv'
    single_path_4 = '../data/stats/stats0010.csv'
    n_path = '../data/stats/stats0006.csv'
    single_paths = [single_path_0, single_path_1, single_path_2, single_path_3, single_path_4]

    # Get final losses for single style networks
    single_style_losses = []
    single_content_losses = []
    for single_path in single_paths:
        style_loss, content_loss = get_final_losses_single_network(single_path)
        single_style_losses.append(style_loss)
        single_content_losses.append(content_loss)

    # Get final losses for n style network
    n_style_losses, n_content_loss = get_final_losses_n_network(n_path, n)

    # Plot
    fig, axis = plt.subplots(1, 1, figsize=(8, 5))
    index = np.arange(n)
    bar_width = 0.35
    opacity = 0.4
    rects1 = axis.bar(index, single_style_losses, bar_width,
                      alpha=opacity, color='b',
                      label='Single Style Networks')

    rects2 = axis.bar(index + bar_width, n_style_losses, bar_width,
                      alpha=opacity, color='r',
                      label='n-Style Network')
    axis.set_xlabel('Style')
    axis.set_ylabel('Final Style Loss')
    axis.set_ylim(0, 600000)
    axis.set_xticks(index + bar_width / 2)
    axis.set_xticklabels(('A', 'B', 'C', 'D', 'E'))
    axis.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    axis.legend()

    fig.tight_layout()
    plt.show()
