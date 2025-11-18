from src.EinsumNetwork.graph.VectorisedNodes import Product, DistributionVector
from src.EinsumNetwork.graph.helper_functions import check_if_is_partition
import networkx as nx
import numpy as np


def partition_on_node(graph, node, scope_partition):
    """
    Helper routine to extend the graph.

    Takes a node and adds a new product child to it. Furthermore, as children of the product, it adds new
    DistributionVector nodes with scopes as prescribed in scope_partition (must be a proper partition of the node's
    scope).

    :param graph: PC graph (DiGraph)
    :param node: node in the graph (DistributionVector)
    :param scope_partition: partition of the node's scope
    :return: the product and a list if the product's children
    """

    if not check_if_is_partition(node.scope, scope_partition):
        raise AssertionError("Not a partition.")

    product = Product(node.scope)
    graph.add_edge(node, product)
    product_children = [DistributionVector(scope) for scope in scope_partition]
    for c in product_children:
        graph.add_edge(product, c)

    return product, product_children


def randomly_partition_on_node(graph, node, num_parts=2, proportions=None, rand_state=None):
    """
    Calls partition_on_node with a random partition -- used for random binary trees (RAT-SPNs).

    :param graph: PC graph (DiGraph)
    :param node: node in the graph (DistributionVector)
    :param num_parts: number of parts in the partition (int)
    :param proportions: split proportions (list of numbers)
    :param rand_state: numpy random_state to use for random split; if None the default numpy random state is used
    :return: the product and a list if the products children
    """
    if proportions is not None:
        if num_parts is None:
            num_parts = len(proportions)
        else:
            if len(proportions) != num_parts:
                raise AssertionError("proportions should have num_parts elements.")
        proportions = np.array(proportions).astype(np.float64)
    else:
        proportions = np.ones(num_parts).astype(np.float64)

    if num_parts > len(node.scope):
        raise AssertionError("Cannot split scope of length {} into {} parts.".format(len(node.scope), num_parts))

    proportions /= proportions.sum()
    if rand_state is not None:
        permutation = list(rand_state.permutation(list(node.scope)))
    else:
        permutation = list(np.random.permutation(list(node.scope)))

    child_indices = []
    for p in range(num_parts):
        p_len = int(np.round(len(permutation) * proportions[0]))
        p_len = min(max(p_len, 1), p + 1 + len(permutation) - num_parts)
        child_indices.append(permutation[0:p_len])
        permutation = permutation[p_len:]
        proportions = proportions[1:]
        proportions /= proportions.sum()

    return partition_on_node(graph, node, child_indices)





def random_binary_trees(num_var, depth, num_repetitions):
    """
    Generate a PC graph via several random binary trees -- RAT-SPNs.

    See
        Random sum-product networks: A simple but effective approach to probabilistic deep learning
        Robert Peharz, Antonio Vergari, Karl Stelzner, Alejandro Molina, Xiaoting Shao, Martin Trapp, Kristian Kersting,
        Zoubin Ghahramani
        UAI 2019

    :param num_var: number of random variables (int)
    :param depth: splitting depth (int)
    :param num_repetitions: number of repetitions (int)
    :return: generated graph (DiGraph)
    """
    graph = nx.DiGraph()
    root = DistributionVector(range(num_var))
    graph.add_node(root)

    for repetition in range(num_repetitions):
        cur_nodes = [root]
        for d in range(depth):
            child_nodes = []
            for node in cur_nodes:
                _, cur_child_nodes = randomly_partition_on_node(graph, node, 2)
                child_nodes += cur_child_nodes
            cur_nodes = child_nodes
        for node in cur_nodes:
            node.einet_address.replica_idx = repetition

    return graph

