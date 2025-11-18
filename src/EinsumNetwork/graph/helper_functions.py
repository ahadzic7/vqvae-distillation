from src.EinsumNetwork.graph.VectorisedNodes import Product, DistributionVector
import networkx as nx
import numpy as np
import os

def check_if_is_partition(X, P):
    """
    Checks if P represents a partition of X.

    :param X: some iterable representing a set of objects.
    :param P: some iterable of iterables, representing a set of sets.
    :return: True of P is a partition of X
                 i) union over P is X
                 ii) sets in P are non-overlapping
    """
    P_as_sets = [set(p) for p in P]
    union = set().union(*[set(p) for p in P_as_sets])
    non_overlapping = len(union) == sum([len(p) for p in P_as_sets])
    return set(X) == union and non_overlapping


def check_graph(graph):
    """
    Check if a graph satisfies our requirements for PC graphs.

    :param graph:
    :return: True/False (bool), string description
    """

    contains_only_PC_nodes = all([type(n) == DistributionVector or type(n) == Product for n in graph.nodes()])

    is_DAG = nx.is_directed_acyclic_graph(graph)
    is_connected = nx.is_connected(graph.to_undirected())

    sums = get_sums(graph)
    products = get_products(graph)

    products_one_parents = all([len(list(graph.predecessors(p))) == 1 for p in products])
    products_two_children = all([len(list(graph.successors(p))) == 2 for p in products])

    sum_to_products = all([all([type(p) == Product for p in graph.successors(s)]) for s in sums])
    product_to_dist = all([all([type(s) == DistributionVector for s in graph.successors(p)]) for p in products])
    alternating = sum_to_products and product_to_dist

    proper_scope = all([len(n.scope) == len(set(n.scope)) for n in graph.nodes()])
    smooth = all([all([p.scope == s.scope for p in graph.successors(s)]) for s in sums])
    decomposable = all([check_if_is_partition(p.scope, [s.scope for s in graph.successors(p)]) for p in products])

    check_passed = contains_only_PC_nodes \
                   and is_DAG \
                   and is_connected \
                   and products_one_parents \
                   and products_two_children \
                   and alternating \
                   and proper_scope \
                   and smooth \
                   and decomposable

    msg = ''
    if check_passed:
        msg += 'Graph check passed.\n'
    if not contains_only_PC_nodes:
        msg += 'Graph does not only contain DistributionVector or Product nodes.\n'
    if not is_connected:
        msg += 'Graph not connected.\n'
    if not products_one_parents:
        msg += 'Products do not have exactly one parent.\n'
    if not products_two_children:
        msg += 'Products do not have exactly two children.\n'
    if not alternating:
        msg += 'Graph not alternating.\n'
    if not proper_scope:
        msg += 'Scope is not proper.\n'
    if not smooth:
        msg += 'Graph is not smooth.\n'
    if not decomposable:
        msg += 'Graph is not decomposable.\n'

    return check_passed, msg.rstrip()


def save_graph(graph, model_dir):
    import pickle
    graph_file = os.path.join(model_dir, "einet.pc")
    with open(graph_file, 'wb') as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
    #print(f"Saved PC graph to {graph_file}")



def get_roots(graph):
    return [n for n, d in graph.in_degree() if d == 0]


def get_sums(graph):
    return [n for n, d in graph.out_degree() if d > 0 and type(n) == DistributionVector]


def get_products(graph):
    return [n for n in graph.nodes() if type(n) == Product]


def get_leaves(graph):
    return [n for n, d in graph.out_degree() if d == 0]


def get_distribution_nodes_by_scope(graph, scope):
    scope = tuple(sorted(scope))
    return [n for n in graph.nodes if type(n) == DistributionVector and n.scope == scope]



def topological_layers(graph):
    """
    Arranging the PC graph in topological layers -- see Algorithm 1 in the paper.

    :param graph: the PC graph (DiGraph)
    :return: list of layers, alternating between DistributionVector and Product layers (list of lists of nodes).
    """
    visited_nodes = set()
    layers = []

    sums = list(sorted(get_sums(graph)))
    products = list(sorted(get_products(graph)))
    leaves = list(sorted(get_leaves(graph)))

    num_internal_nodes = len(sums) + len(products)

    while len(visited_nodes) != num_internal_nodes:
        sum_layer = [s for s in sums if s not in visited_nodes and all([p in visited_nodes for p in graph.predecessors(s)])]
        sum_layer = sorted(sum_layer)
        layers.insert(0, sum_layer)
        visited_nodes.update(sum_layer)

        product_layer = [p for p in products if p not in visited_nodes and all([s in visited_nodes for s in graph.predecessors(p)])]
        product_layer = sorted(product_layer)
        layers.insert(0, product_layer)
        visited_nodes.update(product_layer)

    layers.insert(0, leaves)
    return layers


def plot_graph(graph):
    """
    Plots the PC graph.

    :param graph: the PC graph (DiGraph)
    :return: None
    """
    pos = {}
    layers = topological_layers(graph)
    for i, layer in enumerate(layers):
        for j, item in enumerate(layer):
            pos[item] = np.array([float(j) - 0.25 + 0.5 * np.random.rand(), float(i)])

    distributions = [n for n in graph.nodes if type(n) == DistributionVector]
    products = [n for n in graph.nodes if type(n) == Product]
    node_sizes = [3 + 10 * i for i in range(len(graph))]

    nx.draw_networkx_nodes(graph, pos, distributions, node_shape='+', node_color='red')
    nx.draw_networkx_nodes(graph, pos, products, node_shape='x', node_color='blue')
    nx.draw_networkx_edges(graph, pos, node_size=node_sizes, arrowstyle='->', arrowsize=10, width=2)
