from src.EinsumNetwork.graph.helper_functions import plot_graph, check_graph
from src.EinsumNetwork.graph.random_binary_trees import random_binary_trees
from src.EinsumNetwork.graph.poon_domingos_structure import poon_domingos_structure

import matplotlib.pyplot as plt

# run to see some usage examples
if __name__ == '__main__':
    graph = random_binary_trees(7, 2, 3)
    _, msg = check_graph(graph)
    print(msg)

    plt.figure(1)
    plt.clf()
    plt.title("Random binary tree (RAT-SPN)")
    plot_graph(graph)
    plt.show()
    plt.savefig("random_binary_tree.png")

    print()

    graph = poon_domingos_structure((32, 32), delta=16, max_split_depth=None)
    _, msg = check_graph(graph)
    print(msg)
    plt.figure(1)
    plt.clf()
    plt.title("Poon-Domingos Structure")
    plot_graph(graph)
    plt.show()
    plt.savefig("poon_domingos.png")