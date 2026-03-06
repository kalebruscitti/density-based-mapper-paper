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
from sklearn.cluster import DBSCAN
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from tqdm import trange

data_time = np.load("data/genus1_demo.npy")
data_unsort = data_time[:,1].T
timestamps_unsort = data_time[:,0].T
sorted_indices = np.argsort(timestamps_unsort)
data = data_unsort[sorted_indices]
timestamps = timestamps_unsort[sorted_indices]
N_data = np.size(timestamps)
map_data = y_data = data

def generate_plot(TG, label_edges = True,ax=None, threshold = 0.2, vertices = None):
    self = TG
    if ax is None:
        ax = plt.gca()
    if type(vertices) == type(None):
        vertices = self.G.nodes()

    G = self.G.subgraph(vertices)
    pos = {}
    slice_no = nx.get_node_attributes(TG.G, 'slice_no')
    for node in vertices:
        t = slice_no[node]
        pt_idx = TG.get_vertex_data(node)
        w = TG.weights[t,pt_idx]
        node_ypos = np.average(np.squeeze(TG.data[pt_idx]),weights=w)
        node_xpos = t #np.average(TG.time[pt_idx],weights=w)
        pos[node] = (node_xpos, node_ypos)

    edge_width = np.array([d["weight"] for (u,v,d) in G.edges(data = True)])
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] >= threshold]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if 0.1< d["weight"] < threshold]
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=elarge, width=1, arrows=False)
    if label_edges:
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels)

    node_size = [np.log2(np.size(self.get_vertex_data(node))) for node in vertices]
    clr_dict = nx.get_node_attributes(self.G, 'cluster_no')
    node_clr = [clr_dict[node] for node in vertices]

    nx.draw_networkx_nodes(G, pos, ax=ax,node_size=node_size, node_color=node_clr)
    return ax

"""
Running standard mapper over a range of parameters, with DBSCAN.
"""

checkpoint_numbers = [6,12,18,24,30]
overlap_parameters = [0.2,0.4,0.6,0.8,1.]

fig, axes = plt.subplots(5,5)
fig.set_figwidth(11)
fig.set_figheight(8.5)
fig.dpi = 200
axes = axes.reshape(5*5)
clusterer = DBSCAN()
j = 0
for k in trange(25):
    TG = tm.TemporalMapper(
        timestamps,
        map_data,
        clusterer,
        N_checkpoints = checkpoint_numbers[k%5],
        neighbours = 50,
        overlap = overlap_parameters[j],
        slice_method='time',
        rate_sensitivity=0,
        kernel=tmwc.square,
        #kernel_params=(overlap_parameters[j],),
    )
    TG.build()
    generate_plot(TG,label_edges = False,ax=axes[k])
    xmin,xmax=axes[k].get_xlim()
    ymin,ymax=axes[k].get_ylim()
    axes[k].text(xmin+0.1,ymin+0.1,fr'$n$={TG.N_checkpoints}, $g$={TG.g}, $k$={50}')
    if k%5==4:
        j+=1
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig("genus1-regular-dbscan.png")
plt.show()

"""
Running fuzzy mapper over a range of parameters, with DBSCAN.
"""
checkpoint_numbers = [6,12,18,24,30]
overlap_parameters = [0.2,0.4,0.6,0.8,1.]

fig, axes = plt.subplots(5,5)
fig.set_figwidth(11)
fig.set_figheight(8.5)
fig.dpi = 200
axes = axes.reshape(5*5)
clusterer = DBSCAN()
j = 0
for k in trange(25):
    TG = tm.TemporalMapper(
        timestamps,
        map_data,
        clusterer,
        N_checkpoints = checkpoint_numbers[k%5],
        neighbours = 50,
        slice_method='time',
        overlap = overlap_parameters[j],
        rate_sensitivity=1,
        kernel=tmwc.square,
    )
    TG.build()
    generate_plot(TG,label_edges = False,ax=axes[k])
    xmin,xmax=axes[k].get_xlim()
    ymin,ymax=axes[k].get_ylim()
    axes[k].text(xmin+0.1,ymin+0.1,fr'$n$={checkpoint_numbers[k%5]}, $g$={overlap_parameters[j]}, $k$={50}')
    if k%5==4:
        j+=1
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig("genus1-db-dbscan.png")
plt.show()

"""
Running fuzzy mapper over a range of parameters, with HDBSCAN.
"""
checkpoint_numbers = [6,12,18,24,30]
overlap_parameters = [0.2,0.4,0.6,0.8,1.]

fig, axes = plt.subplots(5,5)
fig.set_figwidth(11)
fig.set_figheight(8.5)
fig.dpi = 200
axes = axes.reshape(5*5)
clusterer = HDBSCAN(min_cluster_size=50)
j = 0
for k in trange(25):
    TG = tm.TemporalMapper(
        timestamps,
        map_data,
        clusterer,
        N_checkpoints = checkpoint_numbers[k%5],
        neighbours = 50,
        slice_method='time',
        overlap = overlap_parameters[j],
        rate_sensitivity=1,
        kernel=tmwc.square,
    )
    TG.build()
    generate_plot(TG,label_edges = False,ax=axes[k])
    xmin,xmax=axes[k].get_xlim()
    ymin,ymax=axes[k].get_ylim()
    axes[k].text(xmin+0.1,ymin+0.1,fr'$n$={checkpoint_numbers[k%5]}, $g$={overlap_parameters[j]}, $k$={50}')
    if k%5==4:
        j+=1
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig("genus1-db-hdbscan.png")
plt.show()
              