import networkx as nx
from collections import defaultdict, deque
from toponymy.clustering import Clusterer, build_cluster_tree, centroids_from_labels, ClusterLayerText
from temporalmapper import TemporalMapper
from toponymy._utils import handle_verbose_params
from copy import deepcopy
import numpy as np
from scipy.sparse import issparse
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        else:
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1


def convert(node, layer):
    t,c = node.split(":")
    return (layer, int(c))

def merge_trees(topic_trees, graphs):
    """
    graphs[0] = leaves
    graphs[-1] = nodes just below root
    Parent of node in layer l is in layer l+1.

    Returns:
        {
            layer: {
                node: equivalence_class_id
            }
        }
    """

    # ---------------------------------------------
    # Build parent lookup from topic_trees
    # ---------------------------------------------
    parent_lookup = []

    for tree in topic_trees:
        parent = {}
        for p, children in tree.items():
            for c in children:
                parent[c] = p

        # roots get parent None
        for node in tree:
            if node not in parent:
                parent[node] = None

        parent_lookup.append(parent)

    n_layers = len(graphs)
    result = {}

    # ---------------------------------------------
    # Process layers top-down
    # ---------------------------------------------
    for l in reversed(range(n_layers)):
        G = graphs[l]
        uf = UnionFind()

        topics = nx.get_node_attributes(G, "topic")
        slice_no = nx.get_node_attributes(G, "slice_no")

        nodes = list(G.nodes())

        for node in nodes:
            uf.add(node)

        groups = defaultdict(list)

        for node in nodes:
            tree_index = slice_no[node]
            parent_node = parent_lookup[tree_index][convert(node,l)]

            # Determine parent equivalence class
            if parent_node == (n_layers, 0):
                parent_class = None
            else:
                _,pc = parent_node
                parent_class = result[l + 1][f'{tree_index}:{pc}']

            key = (topics[node], parent_class)
            groups[key].append(node)

        # Merge nodes within each structural group
        for group_nodes in groups.values():
            base = group_nodes[0]
            for other in group_nodes[1:]:
                uf.union(base, other)

        # Assign final class IDs for this layer
        rep_to_class = {}
        class_counter = 0
        layer_map = {}

        for node in nodes:
            rep = uf.find(node)
            if rep not in rep_to_class:
                rep_to_class[rep] = class_counter
                class_counter += 1
            layer_map[node] = rep_to_class[rep]
        # add the noise point possibilities
        for t in range(len(topic_trees)):
            layer_map[f'{t}:{-1}'] = -1

        result[l] = layer_map

    return result

class MapperClusterer(Clusterer):
    def __init__(
        self,
        base_clusterer: Clusterer,
        mapper_params: dict | None = None,
        verbose: bool = None,
        show_progress_bar: bool = None,
    ):
        self.base_clusterer = base_clusterer
        if mapper_params is None:
            mapper_params = {}
        self.mapper_params = mapper_params

        super().__init__()
        _, self.verbose = handle_verbose_params(
            verbose=verbose, show_progress_bar=show_progress_bar, default_verbose=False
        )

    def fit(
        self,
        clusterable_vectors: np.ndarray,
        embedding_vectors: np.ndarray,
        projection_index: int = -1,
        layer_class = ClusterLayerText,
        verbose: bool = None,
        show_progress_bar: bool = None,
        **layer_kwargs,
    ) -> Clusterer:
        _, verbose_output = handle_verbose_params(
            verbose=verbose if verbose is not None else self.verbose,
            show_progress_bar=show_progress_bar,
            default_verbose=False,
        )
        base_mapper = TemporalMapper(
            clusterer = None,
            **self.mapper_params,
        )
        lens = clusterable_vectors[:, projection_index]
        data = np.delete(
            clusterable_vectors,
            projection_index,
            axis=1
        )
        if issparse(clusterable_vectors):
            base_mapper._mapper.scaler_ = StandardScaler(copy=False, with_mean=False)
        else:
            base_mapper._mapper.scaler_ = StandardScaler(copy=False)
        base_mapper._mapper._compute_midpoints(lens)
        base_mapper._mapper._compute_density(data, lens)
        base_mapper._mapper._compute_weights(data, lens)
        n_layers = 0
        topic_trees = []
        graphs = []
        mappers = []
        slicewise_layers = []
        n_slices = len(base_mapper._mapper.slices_)
        for i, slice_ in enumerate(base_mapper._mapper.slices_):
            cvectors = data[slice_]
            evectors = embedding_vectors[slice_]
            cluster_layers, cluster_tree  = self.base_clusterer.fit_predict(
                clusterable_vectors = cvectors,
                embedding_vectors = evectors,
                layer_class=layer_class,
                verbose=verbose,
                show_progress_bar=show_progress_bar,
                **layer_kwargs,
            )
            if len(cluster_layers)>n_layers:
                n_layers = len(cluster_layers)
            topic_trees.append(cluster_tree)
            slicewise_layers.append(cluster_layers)
        print(f"Layers per slice: {[len(x) for x in slicewise_layers]}")
        for l in range(n_layers):
            sizes = []
            for clayers in slicewise_layers:
                n_clusters = np.unique(clayers[l].cluster_labels).size
                sizes.append(n_clusters)
            print(f"Layer {l} n_cluster: {sizes}")

        layer_clusters = []
        for l in range(n_layers):
            if l>=len(cluster_layers):
                break
            mapper = deepcopy(base_mapper)
            labels = np.full((n_slices, data.shape[0]), -2, dtype=int)
            for i, slice_ in enumerate(base_mapper._mapper.slices_):
                labels[i,slice_] = slicewise_layers[i][l].cluster_labels

            mapper._mapper.labels_ = np.array(labels)
            mapper._mapper._add_vertices()
            mapper._mapper._build_adjacency_matrix(lens)
            mapper._mapper._add_edges()
            mapper._mapper.is_fitted_ = True
            mapper.data = data
            mapper.time = lens
            mapper.n_samples = data.shape[0]
            mapper.n_components = data.shape[1]
            mapper.populate_node_attrs()
            t_attrs = nx.get_node_attributes(mapper.graph, "slice_no")
            mapper.populate_edge_attrs()
            mapper.is_fitted_ = True
            mapper.assign_topics()
            # Run the clustering logic from TemporalMapper.cluster
            dist = cdist(
                mapper._mapper.midpoints_.reshape(-1,1),
                lens.reshape(-1,1)
            )
            pt_max_cluster = np.argmin(
                dist,
                axis=0
            )
            topics = nx.get_node_attributes(mapper.graph, 'topic')
            clusters = []
            clrs = []
            for pt,t in enumerate(pt_max_cluster):
                c = mapper.clusters[t,pt]
                clrs.append(c)
                if c != -2:
                    clusters.append(f'{t}:{c}')
                elif c == -2:
                    clusters.append(f'{t}:{-1}')

            layer_clusters.append(clusters)
            mappers.append(mapper)
            graphs.append(mapper.graph)

        topic_map = merge_trees(topic_trees, graphs)
        # now assign each point its merged cluster val
        cluster_label_layers = []
        for l in range(n_layers):
            clusters = np.full(data.shape[0], -1, dtype=int)
            for node in graphs[l].nodes():
                indices = mappers[l].get_vertex_data(node)
                clusters[indices] = topic_map[l][node]
            cluster_label_layers.append(clusters)

        self.cluster_tree_ = build_cluster_tree(cluster_label_layers)
        self.cluster_layers_ = [
            layer_class(
                labels,
                centroids_from_labels(labels, embedding_vectors),
                layer_id=i,
                verbose=verbose,
                show_progress_bar=show_progress_bar,
                **layer_kwargs,
            )
            for i, labels in enumerate(cluster_label_layers)
        ]
        self.topic_map_ = topic_map
        self.mappers_ = mappers
        return self

    def fit_predict(
        self,
        clusterable_vectors: np.ndarray,
        embedding_vectors: np.ndarray,
        layer_class = ClusterLayerText,
        verbose: bool = None,
        show_progress_bar: bool = None,
        **layer_kwargs,
    ):
        self.fit(
            clusterable_vectors,
            embedding_vectors,
            layer_class=layer_class,
            verbose=verbose,
            show_progress_bar=show_progress_bar,
            **layer_kwargs,
        )
        return self.cluster_layers_, self.cluster_tree_