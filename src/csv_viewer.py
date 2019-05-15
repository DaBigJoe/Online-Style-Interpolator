
import matplotlib.pyplot as plt

import numpy as np

import csv

class CsvViewer:

    def __init__(self, file_path, delimiter=','):
        self.data_points = []

        with open(file_path) as csv_file:
            read_csv = csv.reader(csv_file, delimiter=delimiter)

            # Gets core settings
            for idx, first_row in enumerate(read_csv):
                self.parameter_updates = int(first_row[0])
                self.style_num = int(first_row[1])
                break;

            # Gets styles used
            for idx, second_row in enumerate(read_csv):
                self.styles = []
                for i in range(self.style_num):
                    self.styles.append(second_row[i])
                break

            # For data points
            row = []
            row_styles = []
            row_contents = []

            for idx, row_raw in enumerate(read_csv):
                print(row_raw)

                if (not (idx % self.style_num)) and (not (0 == idx)):
                    _row = np.array([idx])
                    _row_styles = np.array(row_styles)
                    _row_content = np.array(row_contents)

                    new = np.concatenate((_row, _row_styles, _row_content))
                    self.data_points.append(new)

                    row = []
                    row_styles = []
                    row_contents = []

                row.append(float(idx))
                row_styles.append(float(row_raw[1]))
                row_contents.append(float(row_raw[2]))
            self.data_points = np.array(self.data_points)

    def plot_individual_data(self, style_idx=None, average_over=1):
        if None == style_idx:
            style_idx = []
            for i in range(self.style_num):
                style_idx.append(i)

        colors = ['r', 'g', 'b', 'y', 'm', 'black']
        convovle_op = np.array([1/average_over] * average_over)
        time_points = self.data_points[:, 0]

        style_points = []
        content_points = []
        for idx in style_idx:
            style = self.data_points[:, idx + 1]
            style = np.convolve(convovle_op, style)
            style_points.append(style)

            content = self.data_points[:, idx + 1 + len(self.styles)]
            content = np.convolve(convovle_op, content)
            content_points.append(content)

        for idx, style_pts in enumerate(style_points):
            plt.plot(time_points, style_pts, color=colors[idx % len(colors)], label=('Style Loss for ' + self.styles[idx]))
            plt.plot(time_points, content_points[idx], color=colors[len(colors) - ((idx % len(colors))+ 1)], label=('Content Loss for ' + self.styles[idx]))

        plt.legend()
        plt.show()

    def plot_total_data(self, style_idx=None, average_over=1):
        if None == style_idx:
            style_idx = []
            for i in range(self.style_num):
                style_idx.append(i)

        colors = ['r', 'g', 'b', 'y', 'm', 'black']
        convovle_op = np.array([1/average_over] * average_over)
        time_points = self.data_points[:, 0]

        style_points = np.zeros((self.data_points.shape[0]))
        content_points = np.zeros((self.data_points.shape[0]))
        for idx in style_idx:
            style_points += self.data_points[:, idx + 1]
            content_points += self.data_points[:, idx + 1 + len(self.styles)]
        style_points = np.convolve(convovle_op, style_points)
        content_points = np.convolve(convovle_op, content_points)

        plt.plot(time_points, style_points, color=colors[0], label=('Style Loss Total'))
        plt.plot(time_points, content_points, color=colors[1], label=('Content Loss Total'))

        plt.legend()
        plt.show()

cv = CsvViewer('../data/stats/stats2.csv')
cv.plot_individual_data()