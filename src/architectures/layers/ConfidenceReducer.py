import torch
import torch.nn as nn

class ConfidenceReducer(nn.Module):
    def __init__(self, alpha=0.1, axis=-1, kernel_size=3):
        """
        Parameters:
            alpha (float): Fraction of the maximum probability to reduce.
            axis (int): The axis along which the categorical distribution is defined.
            kernel_size (int): Size of the smoothing kernel (must be an odd number).
                               This defines how many neighbors will receive mass.
        """
        super(ConfidenceReducer, self).__init__()
        self.alpha = alpha
        self.axis = axis
        self.kernel_size = kernel_size
        assert kernel_size > 1 and kernel_size % 2 == 1, "Kernel size must be odd and bigger than 1."
        center = kernel_size // 2

        # Create a kernel for neighbor redistribution.
        # We set the center weight to 0 (because that mass is taken away)
        # and assign weights to neighbors based on inverse distance.
        weights = []
        for i in range(kernel_size):
            if i == center:
                weights.append(0.0)
            else:
                weights.append(1.0 / abs(i - center))
        total = sum(weights)
        normalized_weights = [w / total for w in weights]
        # Save the kernel as a buffer so it moves with the model (e.g., to GPU)
        self.register_buffer('neighbor_weights', torch.tensor(normalized_weights))
        
    def forward(self, x):
        """
        x: Tensor of shape (..., num_categories) containing probability distributions.
           It is assumed that the probabilities along the 'axis' sum to 1.
        """
        # 1. Find the max indices and values along the specified axis.
        max_idx = x.argmax(dim=self.axis, keepdim=True)
        max_vals = torch.gather(x, self.axis, max_idx)
        
        # 2. Compute the reduction amount from the maximum element.
        reduction = self.alpha * max_vals
        
        # 3. Reduce the maximum element.
        x_new = x.clone()
        x_new.scatter_(self.axis, max_idx, max_vals - reduction)
        
        # 4. Redistribute the removed mass to neighbors.
        # We'll add an "add_mass" tensor to accumulate contributions.
        add_mass = torch.zeros_like(x)
        center = self.kernel_size // 2
        
        # For each neighbor offset (skip offset 0 since that’s our center)
        for offset in range(-center, center + 1):
            if offset == 0:
                continue  # Skip the center.
            weight = self.neighbor_weights[offset + center]
            if weight == 0.0:
                continue

            # For non-circular distributions, we must be careful not to wrap around.
            # We use slicing to add mass only where the neighbor index is valid.
            if offset > 0:
                # For positive offset, destination indices are [ : -offset ]
                dest_slice = [slice(None)] * x.ndim
                src_slice  = [slice(None)] * x.ndim
                dest_slice[self.axis] = slice(0, -offset)
                src_slice[self.axis]  = slice(offset, None)
            else:  # offset < 0
                dest_slice = [slice(None)] * x.ndim
                src_slice  = [slice(None)] * x.ndim
                dest_slice[self.axis] = slice(-offset, None)
                src_slice[self.axis]  = slice(0, offset)
                
            # Create a mask for where the maximum values were located.
            # Extract only the reduction mass from the max positions.
            mask = torch.zeros_like(x)
            mask.scatter_(self.axis, max_idx, 1.0)
            reduction_mass = reduction * mask
            
            # Shift the reduction_mass by the offset using slicing.
            shifted_reduction = reduction_mass[src_slice]
            add_mass[dest_slice] += weight * shifted_reduction

        x_new += add_mass

        # The argmax will remain the same if the added neighbor mass is not enough
        # to surpass the reduced max value.
        return x_new.softmax(dim=self.axis)


# Example usage:
# logits = torch.tensor([0, 1.])  # Logits input

# logits = torch.tensor([0.9999995, 0.0000001, 0.0000001, 0.0000001, 0.0000001, 0.0000001])

# #print(torch.softmax(logits, dim=0))
# cr = ConfidenceReducer(alpha=.1, axis=0)
# adjusted_probs = cr(logits)
# print(adjusted_probs)
# print(adjusted_probs.sum())
