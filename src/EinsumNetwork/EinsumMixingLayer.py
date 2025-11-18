import torch
from src.EinsumNetwork.SumLayer import SumLayer
from src.EinsumNetwork.utils import sample_matrix_categorical



class EinsumMixingLayer(SumLayer):
    """
    Implements the Mixing Layer, in order to handle sum nodes with multiple children.
    Recall Figure II from above:

           S          S
        /  |  \      / \
       P   P  P     P  P
      /\   /\  /\  /\  /\
     N  N N N N N N N N N

    Figure II


    We implement such excerpt as in Figure III, splitting sum nodes with multiple children in a chain of two sum nodes:

            S          S
        /   |  \      / \
       S    S   S    S  S
       |    |   |    |  |
       P    P   P    P  P
      /\   /\  /\  /\  /\
     N  N N N N N N N N N

    Figure III


    The input nodes N have already been computed. The product nodes P and the first sum layer are computed using an
    EinsumLayer, yielding a log-density tensor of shape
        (batch_size, vector_length, num_nodes).
    In this example num_nodes is 5, since the are 5 product nodes (or 5 singleton sum nodes). The EinsumMixingLayer
    then simply mixes sums from the first layer, to yield 2 sums. This is just an over-parametrization of the original
    excerpt.
    """

    def __init__(self, graph, nodes, einsum_layer, use_em):
        """
        :param graph: the PC graph (see Graph.py)
        :param nodes: the nodes of the current layer (see constructor of EinsumNetwork), which have multiple children
        :param einsum_layer:
        :param use_em:
        """

        self.nodes = nodes

        self.num_sums = set([n.num_dist for n in self.nodes])
        if len(self.num_sums) != 1:
            raise AssertionError("Number of distributions must be the same for all regions in one layer.")
        self.num_sums = list(self.num_sums)[0]

        self.max_components = max([len(graph.succ[n]) for n in self.nodes])
        # einsum_layer is actually the only layer which gives input to EinsumMixingLayer
        # we keep it in a list, since otherwise it gets registered as a torch sub-module
        self.layers = [einsum_layer]
        self.mixing_component_idx = einsum_layer.mixing_component_idx

        if einsum_layer.dummy_idx is None:
            raise AssertionError('EinsumLayer has not set a dummy index for padding.')

        param_shape = (self.num_sums, len(self.nodes), self.max_components)

        # The following code does some bookkeeping.
        # padded_idx indexes into the log-density tensor of the previous EinsumLayer, padded with a dummy input which
        # outputs constantly 0 (-inf in the log-domain), see class EinsumLayer.
        padded_idx = []
        params_mask = torch.ones(param_shape)
        for c, node in enumerate(self.nodes):
            num_components = len(self.mixing_component_idx[node])
            padded_idx += self.mixing_component_idx[node]
            padded_idx += [einsum_layer.dummy_idx] * (self.max_components - num_components)
            if self.max_components > num_components:
                params_mask[:, c, num_components:] = 0.0
            node.einet_address.layer = self
            node.einet_address.idx = c

        super(EinsumMixingLayer, self).__init__(param_shape,
                                                normalization_dims=(2,),
                                                use_em=use_em,
                                                params_mask=params_mask)

        self.register_buffer('padded_idx', torch.tensor(padded_idx))

    def _forward(self, params):
        self.child_log_prob = self.layers[0].prob[:, :, self.padded_idx]
        self.child_log_prob = self.child_log_prob.reshape((self.child_log_prob.shape[0],
                                                           self.child_log_prob.shape[1],
                                                           len(self.nodes),
                                                           self.max_components))

        max_p = torch.max(self.child_log_prob, 3, keepdim=True)[0]
        prob = torch.exp(self.child_log_prob - max_p)

        output = torch.einsum('bonc,onc->bon', prob, params)

        self.prob = torch.log(output) + max_p[:, :, :, 0]

    def _backtrack(self, dist_idx, node_idx, sample_idx, params, use_evidence=False, mode='sample', **kwargs):
        """Helper routine for backtracking in EiNets."""
        with torch.no_grad():
            if use_evidence:
                log_prior = torch.log(params[dist_idx, node_idx, :])
                log_posterior = log_prior + self.child_log_prob[sample_idx, dist_idx, node_idx, :]
                posterior = torch.exp(log_posterior - torch.logsumexp(log_posterior, 1, keepdim=True))
            else:
                posterior = params[dist_idx, node_idx, :]

            if mode == 'sample':
                idx = sample_matrix_categorical(posterior)
            elif mode == 'argmax':
                idx = torch.argmax(posterior, -1)
            dist_idx_out = dist_idx
            node_idx_out = [self.mixing_component_idx[self.nodes[i]][idx[c]] for c, i in enumerate(node_idx)]
            layers_out = [self.layers[0]] * len(node_idx)

        return dist_idx_out, node_idx_out, layers_out
