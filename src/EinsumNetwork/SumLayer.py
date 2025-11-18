import torch
import torch.nn.functional as F
from src.EinsumNetwork.Layer import Layer
import functools

class SumLayer(Layer):
    """
    Implements an abstract SumLayer class. Takes care of parameters and EM.
    EinsumLayer and MixingLayer are derived from SumLayer.
    """

    def __init__(self, params_shape, normalization_dims, use_em, params_mask=None):
        """
        :param params_shape: shape of tensor containing all sum weights (tuple of ints).
        :param normalization_dims: the dimensions (axes) of the sum-weights which shall be normalized
                                   (int of tuple of ints)
        :param use_em: use the on-board EM algorithm?
        :param params_mask: binary mask for masking out certain parameters (tensor of shape params_shape).
        """
        super(SumLayer, self).__init__(use_em=use_em)

        self.params_shape = params_shape
        self.params = None
        self.normalization_dims = normalization_dims
        if params_mask is not None:
            params_mask = params_mask.clone().detach()
        self.register_buffer('params_mask', params_mask)

        self.online_em_frequency = None
        self.online_em_stepsize = None
        self._online_em_counter = 0

    # --------------------------------------------------------------------------------
    # The following functions need to be implemented in derived classes.

    def _forward(self, params):
        """
        Implementation of the actual sum operation.

        :param params: sum-weights to use.
        :return: result of the sum layer. Must yield a (batch_size, num_dist, num_nodes) tensor of log-densities.
                 Here, num_dist is the vector length of vectorized sums (K in the paper), and num_nodes is the number
                 of sum nodes in this layer.
        """
        raise NotImplementedError

    def _backtrack(self, dist_idx, node_idx, sample_idx, params, use_evidence=False, mode='sample', **kwargs):
        """
        Helper routine to implement EiNet backtracking, for sampling or MPE approximation.

        dist_idx, node_idx, sample_idx are lists of indices, all of the same length.

        :param dist_idx: list of indices, indexing into vectorized sums.
        :param node_idx: list of indices, indexing into node list of this layer.
        :param sample_idx: list of sample indices; representing the identity of the samples the EiNet is about to
                           generate. We need this, since not every SumLayer necessarily gets selected in the top-down
                           sampling process.
        :param params: sum-weights to use (Tensor).
        :param use_evidence: incorporate the bottom-up evidence (Bool)? For conditional sampling.
        :param mode: 'sample' or 'argmax'; for sampling or MPE approximation, respectively.
        :param kwargs: Additional keyword arguments.
        :return: depends on particular implementation.
        """
        raise NotImplementedError

    # --------------------------------------------------------------------------------

    def default_initializer(self):
        """
        A simple initializer for normalized sum-weights.
        :return: initial parameters
        """
        params = 0.01 + 0.98 * torch.rand(self.params_shape)
        with torch.no_grad():
            if self.params_mask is not None:
                params.data *= self.params_mask
            params.data = params.data / (params.data.sum(self.normalization_dims, keepdim=True))
        return params

    def initialize(self, initializer='default'):
        """
        Initialize the parameters for this SumLayer.

        :param initializer: denotes the initialization method.
               If 'default' (str): use the default initialization, and store the parameters locally.
               If Tensor: provide custom initial parameters.
        :return: None
        """
        if initializer is None:
            self.params = None
        elif type(initializer) == str and initializer == 'default':
            if self._use_em:
                self.params = torch.nn.Parameter(self.default_initializer())
            else:
                self.params = torch.nn.Parameter(torch.randn(self.params_shape))
        elif type(initializer) == torch.Tensor:
            if initializer.shape != self.params_shape:
                raise AssertionError("Incorrect parameter shape.")
            self.params = torch.nn.Parameter(initializer)
        else:
            raise AssertionError("Unknown initializer.")

    def forward(self, x=None):
        """
        Evaluate this SumLayer.

        :param x: unused
        :return: tensor of log-densities. Must be of shape (batch_size, num_dist, num_nodes).
                 Here, num_dist is the vector length of vectorized sum nodes (K in the paper), and num_nodes is the
                 number of sum nodes in this layer.
        """
        if self._use_em:
            params = self.params
        else:
            reparam = self.reparam(self.params)
            params = reparam
        self._forward(params)

    def backtrack(self, dist_idx, node_idx, sample_idx, use_evidence=False, mode='sample', **kwargs):
        """
        Helper routine for backtracking in EiNets, see _sample(...) for details.
        """
        if mode != 'sample' and mode != 'argmax':
            raise AssertionError('Unknown backtracking mode {}'.format(mode))

        if self._use_em:
            params = self.params
        else:
            with torch.no_grad():
                params = self.reparam(self.params)
        return self._backtrack(dist_idx, node_idx, sample_idx, params, use_evidence, mode, **kwargs)

    def em_purge(self):
        """ Discard em statistics."""
        if self.params is not None:
            self.params.grad = None

    def em_process_batch(self):
        """
        Accumulate EM statistics of current batch. This should be called after call to backwards() on the output of
        the EiNet.
        """
        if not self._use_em:
            raise AssertionError("em_process_batch called while _use_em==False.")
        if self.params is None:
            return

        if self.online_em_frequency is not None:
            self._online_em_counter += 1
            if self._online_em_counter == self.online_em_frequency:
                self.em_update(True)
                self._online_em_counter = 0

    def em_update(self, _triggered=False):
        """
        Do an EM update. If the setting is online EM (online_em_stepsize is not None), then this function does nothing,
        since updates are triggered automatically. Thus, leave the private parameter _triggered alone.

        :param _triggered: for internal use, don't set
        :return: None
        """
        if not self._use_em:
            raise AssertionError("em_update called while _use_em==False.")
        if self.params is None:
            return

        if self.online_em_stepsize is not None and not _triggered:
            return

        with torch.no_grad():
            n = self.params.grad * self.params.data

            if self.online_em_stepsize is None:
                self.params.data = n
            else:
                s = self.online_em_stepsize
                p = torch.clamp(n, 1e-16)
                p = p / (p.sum(self.normalization_dims, keepdim=True))
                self.params.data = (1. - s) * self.params + s * p

            self.params.data = torch.clamp(self.params, 1e-16)
            if self.params_mask is not None:
                self.params.data *= self.params_mask
            self.params.data = self.params / (self.params.sum(self.normalization_dims, keepdim=True))
            self.params.grad = None

    def reparam(self, params_in):
        """
        Reparametrization function, transforming unconstrained parameters into valid sum-weight
        (non-negative, normalized).

        :params_in params: unconstrained parameters (Tensor) to be projected
        :return: re-parametrized parameters.
        """
        other_dims = tuple(i for i in range(len(params_in.shape)) if i not in self.normalization_dims)

        permutation = other_dims + self.normalization_dims
        unpermutation = tuple(c for i in range(len(permutation)) for c, j in enumerate(permutation) if j == i)

        numel = functools.reduce(lambda x, y: x * y, [params_in.shape[i] for i in self.normalization_dims])

        other_shape = tuple(params_in.shape[i] for i in other_dims)
        params_in = params_in.permute(permutation)
        orig_shape = params_in.shape
        params_in = params_in.reshape(other_shape + (numel,))
        out = F.softmax(params_in, -1)
        out = out.reshape(orig_shape).permute(unpermutation)
        return out

    def project_params(self, params):
        """Currently not required."""
        raise NotImplementedError
