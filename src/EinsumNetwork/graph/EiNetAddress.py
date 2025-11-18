class EiNetAddress:
    """
    Address of a PC node to its EiNet implementation.

    In EiNets, each layer implements a tensor of log-densities of shape
        (batch_size, vector_length, num_nodes)
    All DistributionVector's, which are either vectors of leaf distributions (exponential families) or vectors of
    sum nodes, uniquely correspond to some slice of the log-density tensor of some layer, where we slice the last axis.

    EiNetAddress stores the "address" of the implementation in the EinsumNetwork.
    """
    def __init__(self, layer=None, idx=None, replica_idx=None):
        """
        :param layer: which layer implements this node?
        :param idx: which index does the node have in the the layers log-density tensor?
        :param replica_idx: this is solely for the input layer -- see ExponentialFamilyArray and FactorizedLeafLayer.
                            These two layers implement all leaves in parallel. To this end we need "enough leaves",
                            which is achieved to make a sufficiently large "block" of input distributions.
                            The replica_idx indicates in which slice of the ExponentialFamilyArray a leaf is
                            represented.
        """
        self.layer = layer
        self.idx = idx
        self.replica_idx = replica_idx