
import matplotlib.pyplot as plt

import numpy as np

import csv

data_points = []

with open('../data/stats/stats5.csv') as csv_file:
    read_csv = csv.reader(csv_file, delimiter=',')

    row = []
    row_styles = []
    row_contents = []
    for idx, row_raw in enumerate(read_csv):

        if (not (idx % 5)) and (not (0 == idx)):
            _row = np.array([row[0]])
            _row_styles = np.array(row_styles)
            _row_content = np.array(row_contents)

            new = np.concatenate((_row, _row_styles, _row_content))
            data_points.append(new)

            row = []
            row_styles = []
            row_contents = []

        row.append(float(idx))
        row_styles.append(float(row_raw[1]))
        row_contents.append(float(row_raw[2]))
        
data_points = np.array(data_points)

style_points = []
colors = ['r', 'g', 'b', 'y', 'm']

time_points = data_points[:,0]

for i in range(1, 6):
    style = data_points[:,i]
    style_points.append(style)

for idx, style_pts in enumerate(style_points):
    plt.plot(time_points, style_pts, color=colors[idx])

content_points = np.average(data_points[:,6:11], axis=-1)

plt.plot(time_points, content_points, color='black')

plt.show()