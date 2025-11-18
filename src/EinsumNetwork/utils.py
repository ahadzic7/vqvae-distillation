import numpy as np
import os
import torch
from PIL import Image
from torchvision.utils import save_image

def save_model(einet, model_dir, epochs=None, removing=True):
    if epochs is None:
        model_file = os.path.join(model_dir, f"einet.mdl")
    else:
        ep, old = epochs
        model_file = os.path.join(model_dir, f"einet_epoch={ep}.mdl")
        old_file = os.path.join(model_dir, f"einet_epoch={old}.mdl")
        if removing and os.path.exists(old_file):
            os.remove(old_file)
    torch.save(einet, model_file)
    #print(f"Saved model to {model_file}")

def save_image_stack(samples, num_rows, num_columns, filename, margin=5, margin_gray_val=1., frame=0, frame_gray_val=0.0):
    """Save image stack in a tiled image"""

    # for gray scale, convert to rgb
    if len(samples.shape) == 3:
        samples = np.stack((samples,) * 3, -1)

    height = samples.shape[1]
    width = samples.shape[2]

    samples -= samples.min()
    samples /= samples.max()

    img = margin_gray_val * np.ones((height*num_rows + (num_rows-1)*margin, width*num_columns + (num_columns-1)*margin, 3))
    for h in range(num_rows):
        for w in range(num_columns):
            img[h*(height+margin):h*(height+margin)+height, w*(width+margin):w*(width+margin)+width, :] = samples[h*num_columns + w, :]

    framed_img = frame_gray_val * np.ones((img.shape[0] + 2*frame, img.shape[1] + 2*frame, 3))
    framed_img[frame:(frame+img.shape[0]), frame:(frame+img.shape[1]), :] = img

    img = Image.fromarray(np.round(framed_img * 255.).astype(np.uint8))

    img.save(filename)


def sample_matrix_categorical(p):
    """Sample many Categorical distributions represented as rows in a matrix."""
    with torch.no_grad():
        cp = torch.cumsum(p[:, 0:-1], -1)
        rand = torch.rand((cp.shape[0], 1), device=cp.device)
        rand_idx = torch.sum(rand > cp, -1).long()
        return rand_idx


# def sampling(einet, samples_dir, dims, test_x):
#     height, width = dims
#     samples = einet.sample(num_samples=25).reshape((25, 1, 28, 28))
#     f = os.path.join(samples_dir, "samples.png")
#     save_image(samples, f, nrow=5)

#     # Draw conditional samples for reconstruction
#     image_scope = np.array(range(height * width)).reshape(height, width)
#     marginalize_idx = list(image_scope[0:round(height/2), :].reshape(-1))
#     einet.set_marginalization_idx(marginalize_idx)

#     num_samples = 100
#     samples = einet.sample(x=test_x[0:25, :])
#     for _ in range(num_samples-1):
#         samples += einet.sample(x=test_x[0:25, :])
#     samples /= num_samples
#     samples = samples.reshape((25, 1, 28, 28))
#     save_image(samples, "sample_reconstruction.png", nrow=5)
     

# def mpe_reconstructions(einet, samples_dir, dims, test_x):
#     mpe = einet.mpe().reshape((1, 28, 28))
#     f = os.path.join(samples_dir, "mpe.png")
#     save_image(mpe, f, nrow=5)

#     # Draw conditional samples for reconstruction
#     height, width = dims
#     image_scope = np.array(range(height * width)).reshape(height, width)
#     marginalize_idx = list(image_scope[0:round(height/2), :].reshape(-1))
#     einet.set_marginalization_idx(marginalize_idx)

#     mpe_rec = einet.mpe(x=test_x[0:25, :]).reshape((25, 1, 28, 28))
#     f = os.path.join(samples_dir, "mpe_reconstruction.png")
#     save_image(mpe_rec, f, nrow=5)


# from src.EinsumNetwork.distributions.NormalArray import NormalArray
# from src.EinsumNetwork.distributions.BinomialArray import BinomialArray
# from src.EinsumNetwork.distributions.CategoricalArray import CategoricalArray

# def dist_params(dfamily):
#     if dfamily == BinomialArray:
#         return {'N': 255}
#     elif dfamily == CategoricalArray:
#         return {'K': 256}
#     elif dfamily == NormalArray:
#         return {'min_var': 1e-6, 'max_var': 1}
#     return {}