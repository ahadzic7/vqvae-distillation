import numpy as np
import networkx as nx
from src.EinsumNetwork.graph.VectorisedNodes import Product, DistributionVector
from src.EinsumNetwork.graph.helper_functions import get_distribution_nodes_by_scope, get_leaves

def cut_hypercube(hypercube, axis, pos):
    """
    Helper routine for Poon-Domingos (PD) structure. Cuts a discrete hypercube into two sub-hypercubes.

    A hypercube is represented as a tuple (l, r), where l and r are tuples of ints, representing discrete coordinates.
    For example ((0, 0), (10, 8)) represents a 2D hypercube (rectangle) whose upper-left coordinate is (0, 0) and its
    lower-right coordinate (10, 8). Note that upper, lower, left, right are arbitrarily assigned terms here.

    This function cuts a given hypercube in a given axis at a given position.

    :param hypercube: coordinates of the hypercube ((tuple of ints, tuple of ints))
    :param axis: in which axis to cut (int)
    :param pos: at which position to cut (int)
    :return: coordinates of the two hypercubes
    """
    if pos <= hypercube[0][axis] or pos >= hypercube[1][axis]:
        raise AssertionError

    coord_rigth = list(hypercube[1])
    coord_rigth[axis] = pos
    child1 = (hypercube[0], tuple(coord_rigth))

    coord_left = list(hypercube[0])
    coord_left[axis] = pos
    child2 = (tuple(coord_left), hypercube[1])

    return child1, child2


class HypercubeToScopeCache:
    """
    Helper class for Poon-Domingos (PD) structure. Represents a function cache, mapping hypercubes to their unrolled
    scope.

    For example consider the hypercube ((0, 0), (4, 5)), which is a rectangle with 4 rows and 5 columns. We assign
    linear indices to the elements in this rectangle as follows:
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]]
    Similarly, we assign linear indices to higher-dimensional hypercubes, where higher axes toggle faster than lower
    axes. The scope of sub-hypercubes are just the unrolled linear indices. For example, for the rectangle above,
    the sub-rectangle ((1, 2), (4, 5)) has scope (7, 8, 9, 12, 13, 14, 17, 18, 19).

    This class just represents a cached mapping from hypercubes to their scopes.
    """
    def __init__(self):
        self._hyper_cube_to_scope = {}

    def __call__(self, hypercube, shape):
        if hypercube in self._hyper_cube_to_scope:
            return self._hyper_cube_to_scope[hypercube]

        x1 = hypercube[0]
        x2 = hypercube[1]

        if len(x1) != len(x2) or len(x1) != len(shape):
            raise AssertionError
        for i in range(len(shape)):
            if x1[i] < 0 or x2[i] > shape[i]:
                raise AssertionError

        scope = np.zeros(tuple(x2[i] - x1[i] for i in range(len(shape))), np.int64)
        f = 1
        for i, c in enumerate(reversed(range(len(shape)))):
            range_to_add = f * np.array(range(x1[c], x2[c]), np.int64)
            scope += np.reshape(range_to_add, (len(range_to_add),) + i * (1,))
            f *= shape[c]

        scope = tuple(scope.reshape(-1))
        self._hyper_cube_to_scope[hypercube] = scope
        return scope



def poon_domingos_structure(shape, delta, axes=None, max_split_depth=None):
    """
    The PD structure was proposed in
        Sum-Product Networks: A New Deep Architecture
        Hoifung Poon, Pedro Domingos
        UAI 2011
    and generates a PC structure for random variables which can be naturally arranged on discrete grids, like images.

    This function implements PD structure, generalized to grids of arbitrary dimensions: 1D (e.g. sequences),
    2D (e.g. images), 3D (e.g. video), ...
    Here, these grids are called hypercubes, and represented via two coordinates, corresponding to the corner with
    lowest coordinates and corner with largest coordinates.
    For example,
        ((1,), (5,)) is a 1D hypercube ranging from 1 to 5
        ((2,3), (7,7)) is a 2D hypercube ranging from 2 to 7 for the first axis, and from 3 to 7 for the second axis.

    Each coordinate in a hypercube/grid corresponds to a random variable (RVs). The argument shape determines the
    overall hypercube. For example, shape = (28, 28) corresponds to a 2D hypercube containing 28*28 = 784 random
    variables. This would be appropriate, for example, to model MNIST images. The overall hypercube has coordinates
    ((0, 0), (28, 28)). We index the RVs with a linear index, which toggles fastest for higher axes. For example, a
    (5, 5) hypercube gets linear indices
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]  ->   (0, 1, 2, 3, ..., 21, 22, 23, 24)

    Sum nodes and leaves in PCs correspond to sub-hypercubes, and the corresponding unrolled linear indices serve as
    scope for these PC nodes. For example, the sub-hypercube ((1, 2), (4, 5)) of the (5, 5) hypercube above gets scope
        [[ 7  8  9]
         [12 13 14]
         [17 18 19]]   ->   (7, 8, 9, 12, 13, 14, 17, 18, 19)

    The PD structure starts with a single sum node corresponding to the overall hypercube. Then, it recursively splits
    the hypercube using axis-aligned cuts. A cut corresponds to a product node, and the split parts correspond again to
    sums or leaves.
    Regions are split in several ways, by displacing the cut point by some delta. Note that sub-hypercubes can
    typically be obtained by different ways to cut. For example, splitting

        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]

    into

    [[ 0  1]    |   [[ 2  3  4]
     [ 5  6]    |    [ 7  8  9]
     [10 11]    |    [12 13 14]
     [15 16]    |    [17 18 19]
     [20 21]]   |    [22 23 24]]

    and then splitting the left hypercube into

    [[ 0  1]
     [ 5  6]]
    ----------
    [[10 11]
     [15 16]
     [20 21]]

    Gives us the hypercube with scope (0, 1, 5, 6). Alternatively, we could also cut

    [[0 1 2 3 4]
     [5 6 7 8 9]]
    -------------------
    [[10 11 12 13 14]
     [15 16 17 18 19]
     [20 21 22 23 24]]

    and then cut the upper hypercube into

    [[0 1]   |  [[2 3 4]
     [5 6]]  |   [7 8 9]]

    which again gives us the hypercube with scope (0, 1, 5, 6). Thus, we obtained the same hypercube, (0, 1, 5, 6),
    via two (in in general more) alternative cutting processes. What is important is that this hypercube is
    *not duplicated*, but we re-use it when we re-encounter it. In PCs, this means that the sum node associated with
    (0, 1, 5, 6) becomes a shared child of many product nodes. This sharing yields PC structures, which resemble a bit
    convolutional structures. Thus, the PD structure has arguably a suitable inductive bias for array-shaped data.

    The displacement of the cutting points is governed via argument delta. We can also specify multiple deltas, and
    also different delta values for different axes. We first compute all cutting points on the overall hypercube, for
    each specified delta and each axis. When we encounter a hypercube in the recursive splitting process, we consider
    each axis and split it on all cutting points corresponding to the coarsest delta.

    :param shape: shape of the overall hypercube (tuple of ints)
    :param delta: determines the displacement of cutting points.
                  numerical: a single displacement value, applied to all axes.
                  list of numerical: several displacement values, applied to all axes.
                  list of list of numerical: several displacement values, specified for each individual axis.
                                             in this case, the outer list must be of same length as axes.
    :param axes: which axes are subject to cutting? (tuple of ints)
                 For example, if shape = (5, 5) (2DGrid), then axes = (0,) means that we only cut along the first axis.
                 Can be None, in which case all axes are subject to cutting.
    :param max_split_depth: maximal depth for the recursive split process (int)
    :return: PC graph (DiGraph)
    """
    shape = tuple(shape)
    if any([type(s) != int for s in shape]):
        raise TypeError("Elements in shape must be ints.")

    if axes is None:
        axes = list(range(len(shape)))

    try:
        delta = list(delta)
    except TypeError:
        delta = [delta]

    for c in range(len(delta)):
        try:
            delta[c] = list(delta[c])
            if len(delta[c]) != len(axes):
                raise AssertionError("Each delta must either be list of length len(axes), or numeric.")
        except TypeError:
            delta[c] = [float(delta[c])] * len(axes)

    if any([dd < 1. for d in delta for dd in d]):
        raise AssertionError('Any delta must be >= 1.0.')

    sub_shape = tuple(s for c, s in enumerate(shape) if c in axes)
    global_cut_points = []
    for dd in delta:
        cur_global_cur_points = []
        for s, d in zip(sub_shape, dd):
            num_cuts = int(np.floor(float(s - 1) / d))
            cps = [int(np.ceil((i + 1) * d)) for i in range(num_cuts)]
            cur_global_cur_points.append(cps)
        global_cut_points.append(cur_global_cur_points)

    hypercube_to_scope = HypercubeToScopeCache()
    hypercube = ((0,) * len(shape), shape)
    hypercube_scope = hypercube_to_scope(hypercube, shape)

    graph = nx.DiGraph()
    root = DistributionVector(hypercube_scope)

    graph.add_node(root)

    Q = [hypercube]
    depth_dict = {hypercube_scope: 0}

    while Q:
        hypercube = Q.pop(0)
        hypercube_scope = hypercube_to_scope(hypercube, shape)
        depth = depth_dict[hypercube_scope]
        if max_split_depth is not None and depth >= max_split_depth:
            continue

        node = get_distribution_nodes_by_scope(graph, hypercube_scope)
        if len(node) != 1:
            raise AssertionError("Node not found or duplicate.")
        node = node[0]

        found_cut_on_level = False
        for cur_global_cut_points in global_cut_points:
            if found_cut_on_level:
                break
            for ac, axis in enumerate(axes):
                cut_points = [c for c in cur_global_cut_points[ac] if hypercube[0][axis] < c < hypercube[1][axis]]
                if len(cut_points) > 0:
                    found_cut_on_level = True

                for idx in cut_points:
                    child_hypercubes = cut_hypercube(hypercube, axis, idx)
                    child_nodes = []
                    for c_cube in child_hypercubes:
                        c_scope = hypercube_to_scope(c_cube, shape)
                        c_node = get_distribution_nodes_by_scope(graph, c_scope)
                        if len(c_node) > 1:
                            raise AssertionError("Duplicate node.")
                        if len(c_node) == 1:
                            c_node = c_node[0]
                        else:
                            c_node = DistributionVector(c_scope)
                            depth_dict[c_scope] = depth + 1
                            Q.append(c_cube)
                        child_nodes.append(c_node)

                    product = Product(node.scope)
                    graph.add_edge(node, product)
                    for c_node in child_nodes:
                        graph.add_edge(product, c_node)

    for node in get_leaves(graph):
        node.einet_address.replica_idx = 0

    return graph