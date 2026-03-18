from collections import defaultdict, deque
from toponymy.clustering import Clusterer, build_cluster_tree, centroids_from_labels
from toponymy.cluster_layer import ClusterLayerText 
from temporalmapper import TemporalMapper
from toponymy._utils import handle_verbose_params
from copy import deepcopy
import networkx as nx
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
    t, c = node.split(":")
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
            parent_node = parent_lookup[tree_index][convert(node, l)]

            # Determine parent equivalence class
            if parent_node == (n_layers, 0):
                parent_class = None
            else:
                _, pc = parent_node
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


def _make_padding_tree(n_aligned_layers):
    """
    Build a cluster tree for a padding layer (all-noise layer).
    The padding layer sits at aligned index 0 and has no children in the
    layer below, so it only needs a root entry pointing to the sentinel.
    The tree is keyed by aligned layer index: {(layer, cluster): [children]}.
    Since the layer is all-noise there are no real clusters, so only the
    sentinel root (n_aligned_layers, 0) is needed.
    """
    return {(n_aligned_layers, 0): []}


def _align_slice_layers(slicewise_layers, topic_trees, n_layers):
    """
    Align per-slice layer lists to a common depth ``n_layers``, coarse-end
    aligned.  Slices with fewer than ``n_layers`` layers get padding inserted
    at the *fine* end (index 0) so that the coarsest layer in every slice
    always occupies index ``n_layers - 1``.

    Padding layers contain all-noise labels (-1 for every point in the
    slice) and a cluster tree with no children at the fine end.

    Parameters
    ----------
    slicewise_layers : list of list of ClusterLayer
        ``slicewise_layers[i]`` is the list of ClusterLayer objects produced
        by the base clusterer for slice ``i``, ordered fine → coarse.
    topic_trees : list of dict
        ``topic_trees[i]`` is the cluster tree returned alongside
        ``slicewise_layers[i]``.
    n_layers : int
        Target number of layers (== max across all slices).

    Returns
    -------
    aligned_layers : list of list of ClusterLayer
        Same structure as ``slicewise_layers`` but every inner list has
        length ``n_layers``.
    aligned_trees : list of dict
        Cluster trees whose layer indices have been shifted to match the
        aligned indexing.
    """
    aligned_layers = []
    aligned_trees = []

    for i, (layers, tree) in enumerate(zip(slicewise_layers, topic_trees)):
        k = len(layers)
        pad = n_layers - k  # number of padding layers needed at the fine end

        if pad == 0:
            aligned_layers.append(list(layers))
            aligned_trees.append(tree)
            continue

        # ------------------------------------------------------------------
        # Build padding ClusterLayer objects.
        # Each padding layer has all points in the slice labelled as noise
        # (-1).  We reuse the finest real layer's centroid_vectors shape but
        # fill labels with -1.
        # ------------------------------------------------------------------
        finest_real = layers[0]
        n_points = len(finest_real.cluster_labels)
        noise_labels = np.full(n_points, -1, dtype=int)
        # centroid_vectors must have shape (n_clusters, n_features); with
        # zero real clusters we use an empty array.
        empty_centroids = np.empty((0, finest_real.centroid_vectors.shape[1]))

        padding_layer_objects = [
            type(finest_real)(
                cluster_labels=noise_labels.copy(),
                centroid_vectors=empty_centroids.copy(),
                layer_id=j,
            )
            for j in range(pad)
        ]

        aligned_layers.append(padding_layer_objects + list(layers))

        # ------------------------------------------------------------------
        # Shift the cluster-tree keys/values so that the original layer
        # indices (0 … k-1) become (pad … n_layers-1).
        # Padding layers (0 … pad-1) have no children so they are omitted
        # from the tree entirely; the sentinel root stays at (n_layers, 0).
        # ------------------------------------------------------------------
        shifted_tree = {}
        for (pl, pc), children in tree.items():
            new_parent = (pl + pad if pl != n_layers else n_layers, pc)
            new_children = [
                (cl + pad, cc) for cl, cc in children
            ]
            shifted_tree[new_parent] = new_children

        aligned_trees.append(shifted_tree)

    return aligned_layers, aligned_trees


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
        layer_class=ClusterLayerText,
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
            clusterer=None,
            **self.mapper_params,
        )
        lens = clusterable_vectors[:, projection_index]
        data = np.delete(clusterable_vectors, projection_index, axis=1)

        if issparse(clusterable_vectors):
            base_mapper._mapper.scaler_ = StandardScaler(copy=False, with_mean=False)
        else:
            base_mapper._mapper.scaler_ = StandardScaler(copy=False)

        base_mapper._mapper._compute_midpoints(lens)
        base_mapper._mapper._compute_density(data, lens)
        base_mapper._mapper._compute_weights(data, lens)

        n_layers = 0
        topic_trees = []
        slicewise_layers = []
        n_slices = len(base_mapper._mapper.slices_)

        # ------------------------------------------------------------------
        # Cluster each slice independently
        # ------------------------------------------------------------------
        for i, slice_ in enumerate(base_mapper._mapper.slices_):
            cvectors = data[slice_]
            evectors = embedding_vectors[slice_]
            try:
                cluster_layers, cluster_tree = self.base_clusterer.fit_predict(
                    clusterable_vectors=cvectors,
                    embedding_vectors=evectors,
                    layer_class=layer_class,
                    verbose=verbose,
                    show_progress_bar=show_progress_bar,
                    **layer_kwargs,
                )
            except Exception as e:
                raise ValueError(
                    f"Base clusterer failed on slice {i} "
                    f"({len(slice_)} points). "
                    "Try adjusting mapper parameters (e.g. reducing n_slices "
                    "or increasing overlap) or clusterer parameters "
                    "(e.g. reducing base_min_cluster_size)."
                ) from e

            if len(cluster_layers) == 0:
                raise ValueError(
                    f"Base clusterer returned no layers for slice {i} "
                    f"({len(slice_)} points). "
                    "Try adjusting mapper parameters (e.g. reducing n_slices "
                    "or increasing overlap) or clusterer parameters "
                    "(e.g. reducing base_min_cluster_size)."
                )

            if len(cluster_layers) > n_layers:
                n_layers = len(cluster_layers)
            topic_trees.append(cluster_tree)
            slicewise_layers.append(cluster_layers)

        if verbose_output:
            print(f"Layers per slice (before alignment): {[len(x) for x in slicewise_layers]}")

        # ------------------------------------------------------------------
        # Align layers coarse-end first; pad fine end with noise layers
        # ------------------------------------------------------------------
        slicewise_layers, topic_trees = _align_slice_layers(
            slicewise_layers, topic_trees, n_layers
        )

        if verbose_output:
            for l in range(n_layers):
                sizes = []
                for clayers in slicewise_layers:
                    n_clusters = int(np.max(clayers[l].cluster_labels) + 1)
                    sizes.append(n_clusters)
                print(f"Layer {l} n_clusters (after alignment): {sizes}")

        # ------------------------------------------------------------------
        # Build one Mapper graph per aligned layer
        # ------------------------------------------------------------------
        mappers = []
        graphs = []

        # Precompute distance from each point to each midpoint for tie-breaking
        dist_to_midpoints = cdist(
            base_mapper._mapper.midpoints_.reshape(-1, 1),
            lens.reshape(-1, 1),
        )  # shape: (n_slices, n_points)

        for l in range(n_layers):
            mapper = deepcopy(base_mapper)
            labels = np.full((n_slices, data.shape[0]), -2, dtype=int)
            for i, slice_ in enumerate(base_mapper._mapper.slices_):
                labels[i, slice_] = slicewise_layers[i][l].cluster_labels

            mapper._mapper.labels_ = np.array(labels)
            mapper._mapper._add_vertices()
            mapper._mapper._build_adjacency_matrix(lens)  # fix: was `time`
            mapper._mapper._add_edges()
            mapper._mapper.is_fitted_ = True
            mapper.data = data
            mapper.time = lens
            mapper.n_samples = data.shape[0]
            mapper.n_components = data.shape[1]
            mapper.populate_node_attrs()
            mapper.populate_edge_attrs()
            mapper.is_fitted_ = True
            mapper.assign_topics()
            mappers.append(mapper)
            graphs.append(mapper.graph)

        # ------------------------------------------------------------------
        # Assign each point to its best slice, then read its cluster label.
        #
        # Primary key:   highest kernel weight  (weights_[t, pt])
        # Tiebreak:      closest midpoint in time (argmin dist_to_midpoints)
        #
        # A point whose best slice gives label -2 (not in slice at all) falls
        # back to noise (-1).
        # ------------------------------------------------------------------
        weights = base_mapper._mapper.weights_  # (n_slices, n_points)

        # Lexicographic maximisation: primary = weight, secondary = -dist.
        # We encode this as a single float: weight + tiny * (-dist_normalised)
        # so that ties in weight are broken by proximity.
        dist_norm = dist_to_midpoints / (dist_to_midpoints.max() + 1e-12)
        score = weights - 1e-6 * dist_norm  # (n_slices, n_points)
        best_slice = np.argmax(score, axis=0)  # (n_points,)

        for l in range(n_layers):
            clusters = []
            for pt, t in enumerate(best_slice):
                c = mappers[l].clusters[t, pt]
                if c != -2:
                    clusters.append(f'{t}:{c}')
                else:
                    clusters.append(f'{t}:{-1}')

        # ------------------------------------------------------------------
        # Merge the per-slice topic trees across the Mapper graph
        # ------------------------------------------------------------------
        topic_map = merge_trees(topic_trees, graphs)

        # ------------------------------------------------------------------
        # Produce final flat cluster label arrays, one per aligned layer
        # ------------------------------------------------------------------
        cluster_label_layers = []
        for l in range(n_layers):
            final_labels = np.full(data.shape[0], -1, dtype=int)
            for node in graphs[l].nodes():
                indices = mappers[l].get_vertex_data(node)
                final_labels[indices] = topic_map[l][node]
            cluster_label_layers.append(final_labels)

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
        layer_class=ClusterLayerText,
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