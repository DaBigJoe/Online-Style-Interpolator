"""
Style Reconstruction plotter.

Produce plot of relative cumulative loss in convolution block 4.
Thus show the layer-wise rate of optimisation by gradient descent.

Author: Jamie Sian (js17g15@soton.ac.uk)
Created: 16/05/19
"""
import matplotlib.pyplot as plt
from numpy import genfromtxt, divide
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 12

csv_file = '../data/images/style_analysis/3/2019-05-15_14:55:51-lay-3.csv'
data = genfromtxt(csv_file, delimiter=',')
fig, ax = plt.subplots()

domain_sz = 15000
plt.axis([0, domain_sz, 0, 1])

labels = ['Conv 1_1 (0)', 'Conv 2_1 (5)', 'Conv 3_1 (10)', 'Conv 4_1 (17)', 'Cumulative']

for i, data_i in enumerate(data):
    dat_slice = data_i[0:domain_sz]
    dat_slice = divide(dat_slice, dat_slice[0])
    ax.plot(range(domain_sz), dat_slice, label=labels[i])

plt.xlabel('Epochs')
plt.ylabel('Relative Loss')

plt.legend()
plt.rcParams['font.size'] = 14
ax.set_title('Constructing an image to minimise cumulative loss in Conv 4', y=1.03)
plt.rcParams['font.size'] = 12
plt.show()
fig.savefig('ConvGraph.eps', format='eps', dpi=1000)
