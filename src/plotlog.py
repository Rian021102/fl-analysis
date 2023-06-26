import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd

df=pd.read_csv('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/processed/dftestnew.csv')

#filter depth from 3000 to max depth
df = df[(df.DEPTH >= 8000) & (df.DEPTH <= df.DEPTH.max())]


# Define facies labels and colors
facies_labels = ['Non-SST', 'Gas', 'PosGas', 'Oil', 'PosOil', 'WTR', 'WtrRise']
facies_colors = ['#000000', '#990000', '#CC3333', '#006600', '#00CC00', '#000099', '#0000CC']


def label_facies(row, labels):
    fluid = int(row['FLUID'])
    prediction = int(row['PREDICTION'])
    if 0 <= fluid < len(labels) and 0 <= prediction < len(labels):
        return labels[fluid], labels[prediction]
    else:
        return 'Unknown', 'Unknown'

df['facies_labels'], df['prediction_labels'] = zip(*df.apply(lambda row: label_facies(row, facies_labels), axis=1))

def make_facies_log_plot(logs, facies_colors):
    # Make sure logs are sorted by depth
    logs = logs.sort_values(by='DEPTH')
    cmap_facies = colors.ListedColormap(facies_colors[0:len(facies_colors)], 'indexed')

    ztop = logs.DEPTH.min()
    zbot = logs.DEPTH.max()

    cluster_fluid = np.repeat(np.expand_dims(logs['FLUID'].values, 1), 100, 1)
    cluster_prediction = np.repeat(np.expand_dims(logs['PREDICTION'].values, 1), 100, 1)

    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(25, 20))
    ax[0].plot(logs.GR, logs.DEPTH, '-g')
    ax[1].plot(logs.RT, logs.DEPTH, '-')
    ax[2].plot(logs.NPHI, logs.DEPTH, '-', color='0.40')
    ax[3].plot(logs.RHOB, logs.DEPTH, '-', color='r')
    im_fluid = ax[4].imshow(cluster_fluid, interpolation='none', aspect='auto',
                            cmap=cmap_facies, vmin=0, vmax=10)
    im_prediction = ax[5].imshow(cluster_prediction, interpolation='none', aspect='auto',
                                cmap=cmap_facies, vmin=0, vmax=10)

    divider = make_axes_locatable(ax[5])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im_prediction, cax=cax)
    cbar.set_ticks(range(len(facies_labels)))
    cbar.set_ticklabels(facies_labels)

    # Adjust tick label properties
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.xticks(horizontalalignment='center')
    plt.yticks(horizontalalignment='center')
    plt.tick_params(axis='both', which='both', pad=20)

    # Set y-axis limits for RT plot
    ax[1].set_ylim(logs.DEPTH.min(), logs.DEPTH.max())

    for i in range(len(ax) - 2):
        ax[i].set_ylim(ztop, zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)

    ax[0].set_xlabel("GR")
    ax[1].set_xlabel("RT")
    ax[2].set_xlabel("NPHI")
    ax[3].set_xlabel("RHOB")
    ax[4].set_xlabel('Fluid')
    ax[5].set_xlabel('Prediction')

    ax[0].set_ylabel('Depth')

    # Set y-axis tick labels as depth values
    depth_ticks = np.arange(ztop, zbot + 1, 1000)
    ax[0].set_yticks(depth_ticks)
    ax[0].set_yticklabels(depth_ticks.astype(int))

    # Remove y-axis tick labels for all other plots
    for i in range(1, len(ax) - 1):
        ax[i].set_yticklabels([])

    f.suptitle('Well: %s' % logs.iloc[0]['Wells'], fontsize=20, y=0.94)

    plt.show()

# Call the function with your DataFrame
make_facies_log_plot(df, facies_colors)
