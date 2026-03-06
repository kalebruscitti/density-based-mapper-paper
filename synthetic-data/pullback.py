import numpy as np
import pandas as pd
import datamapplot
import sys
import os
import networkx as nx
import temporalmapper as tm
import temporalmapper.utilities_ as tmutils
import temporalmapper.weighted_clustering as tmwc

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN

data_time = np.load("data/density_test_data.npy")
data = data_time[:,0:2]
time = data_time[:,2]
sorted_indices = np.argsort(time)
time = time[sorted_indices]
data = data[sorted_indices]
N_data = np.size(time)


# Construct the temporal graph.
map_data = data
y_data = PCA(n_components=1).fit_transform(data)
clusterer = DBSCAN()
N_checkpoints = 20
kernel_params = (1),
TG = tm.TemporalMapper(
    time,
    map_data,
    clusterer,
    N_checkpoints = N_checkpoints,
    neighbours = 150,
    slice_method='time',
    overlap=1,
    rate_sensitivity=1,
    kernel=tmwc.square,
    verbose=True
)
TG.build()

idx=15
slice_ = (TG.weights[idx] >= 0.1).nonzero()
cp_with_ends = [np.amin(time)]+list(TG.checkpoints)+[np.amax(time)]
bin_width = (cp_with_ends[idx+1]-cp_with_ends[idx])
fig, (ax1,ax2) = plt.subplots(1,2)
fig.set_figwidth(10)
ax1.set_title(f"$f$-Density")
sca=ax1.scatter(time,y_data,s=1,c=TG.density)
ax2.scatter(time,y_data,s=1,c='grey')
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(sca, cax=cax, orientation='vertical')
ax2.scatter(time[slice_],y_data[slice_],s=1,c='red')
tstr = f'Pullback of $({TG.checkpoints[idx]-(bin_width/2)*(1+TG.g):.2f},{TG.checkpoints[idx]+bin_width/2*(1+TG.g):.2f})$'
ax2.set_title(tstr)
ax2.set_xlabel("Time")
ax1.set_xlabel("Time")
ax1.set_ylabel("PCA to 1d")
ax2.axvline(TG.checkpoints[idx]+(bin_width/2)*(1+TG.g),c='k')
ax2.axvline(TG.checkpoints[idx]-(bin_width/2)*(1+TG.g),c='k')
plt.savefig("td-verify.png")
plt.show()