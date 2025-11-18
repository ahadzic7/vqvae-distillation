import torch

class ExponentialFamilyArray(torch.nn.Module):
    """
    ExponentialFamilyArray computes log-densities of exponential families in parallel. ExponentialFamilyArray is
    abstract and needs to be derived, in order to implement a concrete exponential family.

    The main use of ExponentialFamilyArray is to compute the densities for FactorizedLeafLayer, which computes products
    of densities over single RVs. All densities over single RVs are computed in parallel via ExponentialFamilyArray.

    Note that when we talk about single RVs, these can in fact be multi-dimensional. A natural use-case is RGB image
    data: it is natural to consider pixels as single RVs, which are, however, 3-dimensional vectors each.

    Although ExponentialFamilyArray is not derived from class Layer, it implements a similar interface. It is intended
    that ExponentialFamilyArray is a helper class for FactorizedLeafLayer, which just forwards calls to the Layer
    interface.

    Best to think of ExponentialFamilyArray as an array of log-densities, of shape array_shape, parallel for each RV.
    When evaluated, it returns a tensor of shape (batch_size, num_var, *array_shape) -- for each sample in the batch and
    each RV, it evaluates an array of array_shape densities, each with their own parameters. Here, num_var is the number
    of random variables, i.e. the size of the set (boldface) X in the paper.

    The boolean use_em indicates if we want to use the on-board EM algorithm (alternatives would be SGD, Adam,...).

    After the ExponentialFamilyArray has been generated, we need to initialize it. There are several options for
    initialization (see also method initialize(...) below):
        'default': use the default initializer (to be written in derived classes).
        Tensor: provide a custom initialization.

    In order to implement a concrete exponential family, we need to derive this class and implement

        sufficient_statistics(self, x)
        log_normalizer(self, theta)
        log_h(self, x)

        expectation_to_natural(self, phi)
        default_initializer(self)
        project_params(self, params)
        reparam_function(self)
        _sample(self, *args, **kwargs)

    Please see docstrings of these functions below, for further details.
    """

    def __init__(self, num_var, num_dims, array_shape, num_stats, use_em):
        """
        :param num_var: number of random variables (int)
        :param num_dims: dimensionality of random variables (int)
        :param array_shape: shape of log-probability tensor, (tuple of ints)
                            log-probability tensor will be of shape (batch_size, num_var,) + array_shape
        :param num_stats: number of sufficient statistics of exponential family (int)
        :param use_em: use internal EM algorithm? (bool)
        """
        super(ExponentialFamilyArray, self).__init__()

        self.num_var = num_var
        self.num_dims = num_dims
        self.array_shape = array_shape
        self.num_stats = num_stats
        self.params_shape = (num_var, *array_shape, num_stats)

        self.params = None
        self.ll = None
        self.suff_stats = None

        self.marginalization_idx = None
        self.marginalization_mask = None

        self._use_em = use_em
        self._p_acc = None
        self._stats_acc = None
        self._online_em_frequency = None
        self._online_em_stepsize = None
        self._online_em_counter = 0

    # --------------------------------------------------------------------------------
    # The following functions need to be implemented to specify an exponential family.

    def sufficient_statistics(self, x):
        """
        The sufficient statistics function for the implemented exponential family (called T(x) in the paper).

        :param x: observed data (Tensor).
                  If self.num_dims == 1, this can be either of shape (batch_size, self.num_var, 1) or
                  (batch_size, self.num_var).
                  If self.num_dims > 1, this must be of shape (batch_size, self.num_var, self.num_dims).
        :return: sufficient statistics of the implemented exponential family (Tensor).
                 Must be of shape (batch_size, self.num_var, self.num_stats)
        """
        raise NotImplementedError

    def log_normalizer(self, theta):
        """
        Log-normalizer of the implemented exponential family (called A(theta) in the paper).

        :param theta: natural parameters (Tensor). Must be of shape (self.num_var, *self.array_shape, self.num_stats).
        :return: log-normalizer (Tensor). Must be of shape (self.num_var, *self.array_shape).
        """
        raise NotImplementedError

    def log_h(self, x):
        """
        The log of the base measure (called h(x) in the paper).

        :param x: observed data (Tensor).
                  If self.num_dims == 1, this can be either of shape (batch_size, self.num_var, 1) or
                  (batch_size, self.num_var).
                  If self.num_dims > 1, this must be of shape (batch_size, self.num_var, self.num_dims).
        :return: log(h) of the implemented exponential family (Tensor).
                 Can either be a scalar or must be of shape (batch_size, self.num_var)
        """
        raise NotImplementedError

    def expectation_to_natural(self, phi):
        """
        Conversion from expectations parameters phi to natural parameters theta, for the implemented exponential
        family.

        :param phi: expectation parameters (Tensor). Must be of shape (self.num_var, *self.array_shape, self.num_stats).
        :return: natural parameters theta (Tensor). Same shape as phi.
        """
        raise NotImplementedError

    def default_initializer(self):
        """
        Default initializer for params.

        :return: initial parameters for the implemented exponential family (Tensor).
                 Must be of shape (self.num_var, *self.array_shape, self.num_stats)
        """
        raise NotImplementedError

    def project_params(self, params):
        """
        Project onto parameters' constraint set.

        Exponential families are usually defined on a constrained domain, e.g. the second parameter of a Gaussian needs
        to be non-negative. The EM algorithm takes the parameters sometimes out of their domain. This function projects
        them back onto their domain.

        :param params: the current parameters, same shape as self.params.
        :return: projected parameters, same shape as self.params.
        """
        raise NotImplementedError

    def reparam(self, params):
        """
        Re-parameterize parameters, in order that they stay in their constrained domain.

        When we are not using the EM, we need to transform unconstrained (real-valued) parameters to the constrained set
        of the expectation parameter.
        This function should return such a function (i.e. the return value should not be
        a projection, but a function which does the projection).

        :param params: unconstrained parameters (Tensor) to be projected
        :return: re-parametrized parameters.
        """
        raise NotImplementedError

    def _sample(self, num_samples, params, **kwargs):
        """
        Helper function for sampling the exponential family.

        :param num_samples: number of samples to be produced
        :param params: expectation parameters (phi) of the exponential family, of shape
                       (self.num_var, *self.array_shape, self.num_stats)
        :param kwargs: keyword arguments
               Depending on the implementation, kwargs can also contain further arguments.
        :return: i.i.d. samples of the exponential family (Tensor).
                 Should be of shape (num_samples, self.num_var, self.num_dims, *self.array_shape)
        """
        raise NotImplementedError

    def _argmax(self, params, **kwargs):
        """
        Helper function for getting the argmax of the exponential family.

        :param params: expectation parameters (phi) of the exponential family, of shape
                       (self.num_var, *self.array_shape, self.num_stats)
        :param kwargs: keyword arguments
               Depending on the implementation, kwargs can also contain further arguments.
        :return: argmax of the exponential family (Tensor).
                 Should be of shape (self.num_var, self.num_dims, *self.array_shape)
        """
        raise NotImplementedError

    # --------------------------------------------------------------------------------

    def initialize(self, initializer='default'):
        """
        Initialize the parameters for this ExponentialFamilyArray.

        :param initializer: denotes the initialization method.
               If 'default' (str): use the default initialization, and store the parameters locally.
               If Tensor: provide custom initial parameters.
        :return: None
        """
        if type(initializer) == str and initializer == 'default':
            # default initializer; when em is switched off, we reparametrize and use Gaussian noise as init values.
            if self._use_em:
                self.params = torch.nn.Parameter(self.default_initializer())
            else:
                self.params = torch.nn.Parameter(torch.randn(self.params_shape))
        elif type(initializer) == torch.Tensor:
            # provided initializer
            if initializer.shape != self.params_shape:
                raise AssertionError("Incorrect parameter shape.")
            self.params = torch.nn.Parameter(initializer)
        else:
            raise AssertionError("Unknown initializer.")

    def forward(self, x):
        """
        Evaluate log-densities.

        Accepts input in several shapes, and normalizes to (B, num_var, num_dims) before computing:
        - (B, N_RV)                 -> only if num_dims == 1
        - (B, N_RV, num_dims)       -> already correct
        - (B, C, N_RV)              -> will be permuted to (B, N_RV, C) when C == num_dims
        - (B, C, H, W)              -> flattened to (B, N_RV, C) if H*W == num_var
        """
        # --- normalize input to (B, num_var, num_dims) ---
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")

        B = x.shape[0]
        nd = x.dim()

        if nd == 2:
            # (B, num_var)  -> only allowed if num_dims == 1
            assert self.num_dims == 1, (
                f"Received 2D input with shape {tuple(x.shape)}, "
                f"but self.num_dims={self.num_dims}. Provide shape (B, num_var, num_dims) or (B, C, N_RV)."
            )
            x = x.unsqueeze(-1)  # -> (B, num_var, 1)

        elif nd == 3:
            # Could be (B, num_var, num_dims) OR (B, C, num_var)
            b, a, c = x.shape
            # # self.num_var = 128**2
            # print(f"Num var {self.num_var}, num dims {self.num_dims}")
            # print(f"Input shape {(a, b, c)}")
            # exit()
            if a == self.num_var and c == self.num_dims:
                # already (B, num_var, num_dims)
                pass
            elif a == self.num_dims and c == self.num_var:
                # (B, C, N_RV) -> permute to (B, N_RV, C)
                x = x.permute(0, 2, 1).contiguous()
            elif a == self.num_var and self.num_dims == 1 and c == 1:
                # (B, num_var, 1) -> already OK
                pass
            else:
                raise ValueError(
                    f"Unrecognized 3D input shape {tuple(x.shape)}."
                    f"Expected one of: (B, num_var, num_dims) or (B, num_dims, num_var)."
                )
        elif nd == 4:
            # (B, C, H, W) -> flatten spatial dims to N_RV and permute to (B, N_RV, C)
            b, c, h, w = x.shape
            n = h * w
            assert n == self.num_var, (
                f"Flattened spatial resolution H*W={n} does not match self.num_var={self.num_var}."
            )
            assert c == self.num_dims, (
                f"Channel count C={c} does not match self.num_dims={self.num_dims}."
            )
            x = x.view(b, c, n).permute(0, 2, 1).contiguous()  # -> (B, N_RV, C)

        else:
            raise ValueError(f"Unsupported input tensor dimension: {nd}. Expected 2, 3 or 4 dims.")

        # Now x is guaranteed to be (B, num_var, num_dims)
        # ----------------------------------------------------------------
        # Existing logic (unchanged except using the normalized x)
        if self._use_em:
            with torch.no_grad():
                theta = self.expectation_to_natural(self.params)
        else:
            phi = self.reparam(self.params)
            theta = self.expectation_to_natural(phi)

        # suff_stats: (batch_size, self.num_var, self.num_stats)
        self.suff_stats = self.sufficient_statistics(x)

        # reshape for broadcasting to (B, num_var, *array_shape, num_stats)
        shape = self.suff_stats.shape
        shape = shape[0:2] + (1,) * len(self.array_shape) + (shape[2],)
        self.suff_stats = self.suff_stats.reshape(shape)

        # log_normalizer: (self.num_var, *self.array_shape)
        log_normalizer = self.log_normalizer(theta)

        # log_h: scalar, or (batch_size, self.num_var)
        log_h = self.log_h(x)
        if isinstance(log_h, torch.Tensor) and len(log_h.shape) > 0:
            log_h = log_h.reshape(log_h.shape[0:2] + (1,) * len(self.array_shape))

        # compute the exponential family tensor (B, num_var, *array_shape)
        self.ll = log_h + (theta.unsqueeze(0) * self.suff_stats).sum(-1) - log_normalizer

        if self._use_em:
            # If EM needs gradients wrt self.ll itself (but not necessarily wrt params),
            # keep this behavior; but see comments in earlier analysis about no_grad + requires_grad_.
            self.ll.requires_grad_()

        # Marginalization mask (cache/register_buffer recommended elsewhere)
        if self.marginalization_idx is not None:
            with torch.no_grad():
                mask = torch.ones(self.num_var, dtype=self.ll.dtype, device=self.ll.device)
                mask[self.marginalization_idx] = 0.0
                shape = (1, self.num_var) + (1,) * len(self.array_shape)
                self.marginalization_mask = mask.view(shape)
                self.marginalization_mask.requires_grad_(False)
        else:
            self.marginalization_mask = None

        if self.marginalization_mask is not None:
            output = self.ll * self.marginalization_mask
        else:
            output = self.ll

        return output

    def sample(self, num_samples=1, **kwargs):
        if self._use_em:
            params = self.params
        else:
            with torch.no_grad():
                params = self.reparam(self.params)
        return self._sample(num_samples, params, **kwargs)

    def argmax(self, **kwargs):
        if self._use_em:
            params = self.params
        else:
            with torch.no_grad():
                params = self.reparam(self.params)
        return self._argmax(params, **kwargs)

    def em_set_hyperparams(self, online_em_frequency, online_em_stepsize, purge=True):
        """Set new setting for online EM."""
        if purge:
            self.em_purge()
            self._online_em_counter = 0
        self._online_em_frequency = online_em_frequency
        self._online_em_stepsize = online_em_stepsize

    def em_purge(self):
        """ Discard em statistics."""
        if self.ll is not None and self.ll.grad is not None:
            self.ll.grad.zero_()
        self._p_acc = None
        self._stats_acc = None

    def em_process_batch(self):
        """
        Accumulate EM statistics of current batch. This should typically be called via EinsumNetwork.em_process_batch().
        """
        if not self._use_em:
            raise AssertionError("em_process_batch called while _use_em==False.")
        if self.params is None:
            return
        
        with torch.no_grad():
            p = self.ll.grad
            weighted_stats = (p.unsqueeze(-1) * self.suff_stats).sum(0)
            p = p.sum(0)

            if self._p_acc is None:
                self._p_acc = torch.zeros_like(p)
                
            self._p_acc += p
            
            if self._stats_acc is None:
                self._stats_acc = torch.zeros_like(weighted_stats)
                
            self._stats_acc += weighted_stats

            self.ll.grad.zero_()

            if self._online_em_frequency is not None:
                self._online_em_counter += 1
                
                if self._online_em_counter == self._online_em_frequency:
                    self.em_update(True)
                    self._online_em_counter = 0


    def em_update(self, _triggered=False):
        """
        Do an EM update. If the setting is online EM (online_em_stepsize is not None), then this function does nothing,
        since updates are triggered automatically. (Thus, leave the private parameter _triggered alone)

        :param _triggered: for internal use, don't set
        :return: None
        """
        if not self._use_em:
            raise AssertionError("em_update called while _use_em==False.")
        if self._online_em_stepsize is not None and not _triggered:
            return
        eps = 1e-12
        with torch.no_grad():
            d = (self._p_acc.unsqueeze(-1) + eps)
            if self._online_em_stepsize is None:
                self.params.data = self._stats_acc / d
            else:
                s = self._online_em_stepsize  
                self.params.data = (1. - s) * self.params + s * self._stats_acc / d
            
            self.params.data = self.project_params(self.params.data)

        self._p_acc = None
        self._stats_acc = None

    def set_marginalization_idx(self, idx):
        """Set indicices of marginalized variables."""
        self.marginalization_idx = idx

    def get_marginalization_idx(self):
        """Set indicices of marginalized variables."""
        return self.marginalization_idx


def shift_last_axis_to(x, i):
    """This takes the last axis of tensor x and inserts it at position i"""
    num_axes = len(x.shape)
    return x.permute(tuple(range(i)) + (num_axes - 1,) + tuple(range(i, num_axes - 1)))


# def forward(self, x):
    #     """
    #     Evaluates the exponential family, in log-domain. For a single log-density we would compute
    #         log_h(X) + <params, T(X)> + A(params)
    #     Here, we do this in parallel and compute an array of log-densities of shape array_shape, for each sample in the
    #     batch and each RV.

    #     :param x: input data (Tensor).
    #               If self.num_dims == 1, this can be either of shape (batch_size, self.num_var, 1) or
    #               (batch_size, self.num_var).
    #               If self.num_dims > 1, this must be of shape (batch_size, self.num_var, self.num_dims).
    #     :return: log-densities of implemented exponential family (Tensor).
    #              Will be of shape (batch_size, self.num_var, *self.array_shape)
    #     """
    #     exit(x.shape)
        
    #     if self._use_em:
    #         with torch.no_grad():
    #             theta = self.expectation_to_natural(self.params)
    #     else:
    #         phi = self.reparam(self.params)
    #         theta = self.expectation_to_natural(phi)

    #     # suff_stats: (batch_size, self.num_var, self.num_stats)
    #     self.suff_stats = self.sufficient_statistics(x)
        
    #     # reshape for broadcasting
    #     shape = self.suff_stats.shape
    #     # print(shape)
    #     shape = shape[0:2] + (1,) * len(self.array_shape) + (shape[2],)
    #     # print(shape)
    #     # exit()
    #     self.suff_stats = self.suff_stats.reshape(shape)

    #     # log_normalizer: (self.num_var, *self.array_shape)
    #     log_normalizer = self.log_normalizer(theta)
        
    #     # log_h: scalar, or (batch_size, self.num_var)
    #     log_h = self.log_h(x)
    #     if len(log_h.shape) > 0:
    #         # reshape for broadcasting
    #         log_h = log_h.reshape(log_h.shape[0:2] + (1,) * len(self.array_shape))

    #     # compute the exponential family tensor
    #     # (batch_size, self.num_var, *self.array_shape)
    #     # print(theta.unsqueeze(0).shape)
    #     # print(self.suff_stats.shape)
        
    #     #print((theta.unsqueeze(0) * self.suff_stats).shape)
    #     # exit()
    #     # print(log_h.shape)
        
    #     self.ll = log_h + (theta.unsqueeze(0) * self.suff_stats).sum(-1) - log_normalizer
        
    #     if self._use_em:
    #         # EM needs the gradient with respect to self.ll
    #         self.ll.requires_grad_()
    #     # Marginalization in PCs works by simply setting leaves corresponding to marginalized variables to 1 (0 in
    #     # (log-domain). We achieve this by a simple multiplicative 0-1 mask, generated here.
    #     # TODO: the marginalization mask doesn't need to be computed every time; only when marginalization_idx changes.
    #     if self.marginalization_idx is not None:
    #         with torch.no_grad():
    #             self.marginalization_mask = torch.ones(self.num_var, dtype=self.ll.dtype, device=self.ll.device)
    #             self.marginalization_mask.data[self.marginalization_idx] = 0.0
    #             shape = (1, self.num_var) + (1,) * len(self.array_shape)
    #             self.marginalization_mask = self.marginalization_mask.reshape(shape)
    #             self.marginalization_mask.requires_grad_(False)
    #     else:
    #         self.marginalization_mask = None
    #     if self.marginalization_mask is not None:
    #         output = self.ll * self.marginalization_mask
    #     else:
    #         output = self.ll
    #     return output