
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
                    style = second_row[i]
                    print('IDX ' + str(i) + ', STYLE ' + str(style))

                    self.styles.append(style)
                break

            # For data points
            row = []
            row_styles = []
            row_contents = []

            for idx, row_raw in enumerate(read_csv):
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

            plt.rcParams['mathtext.fontset'] = 'stix'
            plt.rcParams['font.family'] = 'STIXGeneral'
            plt.rcParams['font.size'] = 12

    def plot_individual_data(self, style_idx=None, smooth_over=1, linewidth=1, show_content=True, show_style=True, show=True, alpha=1.0):
        if None == style_idx:
            style_idx = []
            for i in range(self.style_num):
                style_idx.append(i)

        convovle_op = np.array([1/smooth_over] * smooth_over)
        time_points = self.data_points[:, 0]
        time_points = np.convolve(convovle_op, time_points, mode='valid')

        style_points = []
        content_points = []
        for idx in style_idx:
            style = self.data_points[:, idx + 1]
            style = np.convolve(convovle_op, style, mode='valid')
            style_points.append(style)

            content = self.data_points[:, idx + 1 + len(self.styles)]
            content = np.convolve(convovle_op, content, mode='valid')
            content_points.append(content)

        for idx, style_pts in enumerate(style_points):
            if show_style:
                plt.plot(time_points, style_pts, label=('Style Loss for ' + self.styles[idx]), alpha=alpha, linewidth=linewidth)
            if show_content:
                plt.plot(time_points, content_points[idx], label=('Content Loss for ' + self.styles[idx]), alpha=alpha, linewidth=linewidth)

        if show:
            #plt.legend()
            plt.show()

    def plot_total_data(self, style_idx=None, smooth_over=1,linewidth=1, average_styles=False, show_style=True, show_content=True, show=True, alpha=1.0):
        if None == style_idx:
            style_idx = []
            for i in range(self.style_num):
                style_idx.append(i)

        convovle_op = np.array([1/smooth_over] * smooth_over)
        time_points = self.data_points[:, 0]
        time_points = np.convolve(convovle_op, time_points, mode='valid')

        style_points = np.zeros((self.data_points.shape[0]))
        content_points = np.zeros((self.data_points.shape[0]))
        for idx in style_idx:
            style_points += self.data_points[:, idx + 1]
            content_points += self.data_points[:, idx + 1 + len(self.styles)]

        if average_styles:
            style_points /= float(len(style_idx))
            content_points /= float(len(style_idx))

        style_points = np.convolve(convovle_op, style_points, mode='valid')
        content_points = np.convolve(convovle_op, content_points, mode='valid')

        if show_style:
            label = 'Style Loss Total'
            if average_styles:
                label = 'Style Loss Average'

            plt.plot(time_points, style_points, label=(label), alpha=alpha, linewidth=linewidth)
        if show_content:
            label = 'Content Loss Total'
            if average_styles:
                label = 'Content Loss Average'
            plt.plot(time_points, content_points,  label=(label), alpha=alpha, linewidth=linewidth)

        if show:
            #plt.legend()
            plt.show()


cv = CsvViewer('../data/stats/10Trained.csv')
cv.plot_total_data(show=False, show_style=False, show_content=True, average_styles=True, smooth_over=30, linewidth=2)
cv.plot_individual_data(show=False, show_style=False, show_content=True, smooth_over=30, alpha=0.5)

plt.xlim(left=0, right=40000)
plt.ylim(top=900000, bottom=100000)

#plt.title('Content Loss Over Time for 10 Styles')
plt.xlabel('Parameter Updates')
plt.ylabel('Content Loss')

plt.show()

cv.plot_total_data(show=False, show_style=True, show_content=False, average_styles=True, smooth_over=30, linewidth=2)
cv.plot_individual_data(show=False, show_style=True, show_content=False, smooth_over=30, alpha=0.4)

plt.xlim(left=0, right=40000)
plt.ylim(top=350000, bottom=0)

#plt.title('Style Loss Over Time for 10 Styles')
plt.xlabel('Parameter Updates')
plt.ylabel('Style Loss')

plt.show()