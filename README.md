# VQ-VAE with PixelCNN and Continuous Mixtures

## Complete Guide to Reproducing Results

This repository implements a Vector Quantized Variational Autoencoder (VQ-VAE) with PixelCNN as a prior model
It also includes Continuous Mixtures and EinSum networks as baseline SOTA models for tractable probabilistic modeling. 
This comprehensive guide will walk you through every step needed to reproduce the results from scratch.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Project Structure](#project-structure)
5. [Configuration](#configuration)
6. [Training Models](#training-models)
7. [Evaluating Models](#evaluating-models)
8. [Advanced Operations](#advanced-operations)
9. [Troubleshooting](#troubleshooting)
10. [Expected Results](#expected-results)

---

## Overview

### What This Repository Does

This project implements three key models for generative modeling:

1. **VQ-VAE (Vector Quantized Variational Autoencoder)**: Learns discrete latent representations of images by encoding them into a quantized latent space
2. **PixelCNN**: An autoregressive model that learns the prior distribution over the discrete latent codes produced by VQ-VAE
3. **Continuous Mixtures (CM)**: A tractable probabilistic model that creates exact mixture models over the latent space
4. **EinSum Networks (EiNets)**

### Key Features

- Complete training pipeline for all models
- Exact marginalization of latent space for mixture models distilled from VQ-VAEs
- Inpainting capabilities for image completion tasks
- Configurable hyperparameters via JSON configuration files

---

---

## Environment Setup

### Step 1: Install Conda

### Step 2: Create Conda Environment

Create a new isolated environment with Python 3.10:

```bash
conda create --name vqvae-tpm python=3.10
```

This creates an environment named `vqvae-tpm` (VQ-VAE Tractable Probabilistic Modeling).

### Step 3: Activate Environment

Activate the newly created environment:

```bash
# On Linux/macOS
source activate vqvae-tpm

# Alternative (all platforms)
conda activate vqvae-tpm
```

You should see `(vqvae-tpm)` prefix in your terminal prompt.

### Step 4: Install PyTorch with CUDA Support

Install PyTorch 2.5.1 with CUDA 12.4 support:

```bash
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
```

**Note:** If you have a different CUDA version or want CPU-only installation:
- For CUDA 11.8: Replace `cu124` with `cu118`
- For CPU only: Use `--index-url https://download.pytorch.org/whl/cpu`


### Step 6: Install Additional Dependencies

Install PyTorch Lightning (training framework):

```bash
pip install lightning
```

Install Matplotlib (plotting and visualization):

```bash
pip install matplotlib
```

### Step 7: Clone Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/ahadzic7/test-vqvae.git
cd test-vqvae
```

### Step 8: Verify Installation

Check that all packages are installed correctly:

```bash
python -c "import torch; import lightning; import tensorboard; import matplotlib; print('All dependencies installed successfully!')"
```

---

## Project Structure

Understanding the repository structure is crucial for navigating and modifying the code:

```
test-vqvae/
├── train.py                    # Main training script for all models
├── eval.py                     # Evaluation script for trained models
├── marginalize.py              # Script for exact latent space enumeration
├── pcnn_cdf.py                 # PixelCNN CDF plotting utility
├── inpaint.py                  # Single-query inpainting script
├── inpaint_batched.py          # Batch inpainting script
├── config/                     # Configuration directory
│   ├── train/                  # Training configurations
│   │   ├── vqvae.json         # VQ-VAE training hyperparameters
│   │   ├── pcnn.json          # PixelCNN training hyperparameters
│   │   └── cm.json            # Continuous Mixtures hyperparameters
│   └── eval/                   # Evaluation configurations
│       ├── vqvae.json         # VQ-VAE evaluation settings
│       ├── pcnn.json          # PixelCNN evaluation settings
│       └── cm.json            # Continuous Mixtures evaluation settings
├── models/                     # Model architecture definitions
├── data/                       # Dataset loaders and preprocessing
├── utils/                      # Utility functions
└── checkpoints/               # Saved model checkpoints (created during training)
```

---

## Configuration

### Understanding Configuration Files

All training and evaluation settings are controlled through JSON configuration files located in `config/train/` and `config/eval/` directories.

### VQ-VAE Configuration (`config/train/vqvae.json`)

Key hyperparameters you can adjust:

```json
{
  "model": {
    "embedding_dim": 64,        # Dimensionality of codebook vectors
    "num_embeddings": 512,      # Size of the codebook (number of discrete codes)
    "commitment_cost": 0.25,    # Weight for commitment loss
    "decay": 0.99               # Exponential moving average decay for codebook updates
  },
  "training": {
    "batch_size": 128,          # Number of images per batch
    "learning_rate": 2e-4,      # Adam optimizer learning rate
    "num_epochs": 100,          # Total training epochs
    "save_interval": 10         # Save checkpoint every N epochs
  },
  "data": {
    "dataset": "cifar10",       # Dataset name (cifar10, mnist, etc.)
    "image_size": 32,           # Input image resolution
    "num_workers": 4            # DataLoader worker processes
  }
}
```

### PixelCNN Configuration (`config/train/pcnn.json`)

Key hyperparameters:

```json
{
  "model": {
    "num_layers": 15,           # Number of PixelCNN layers
    "hidden_dim": 128,          # Hidden dimension size
    "num_classes": 512,         # Should match VQ-VAE num_embeddings
    "kernel_size": 7            # Convolutional kernel size
  },
  "training": {
    "batch_size": 64,
    "learning_rate": 3e-4,
    "num_epochs": 200
  }
}
```

### Continuous Mixtures Configuration (`config/train/cm.json`)

Configuration for the exact mixture model:

```json
{
  "marginalization": {
    "batch_size": 1000,         # Batch size for enumeration
    "save_interval": 100        # Save progress every N batches
  }
}
```

### Modifying Configurations

To modify any setting:

1. Open the relevant JSON file in a text editor
2. Modify the desired parameter
3. Save the file
4. Run training with the updated configuration

**Example:** To increase VQ-VAE embedding dimension:

```bash
# Edit config/train/vqvae.json
# Change "embedding_dim": 64 to "embedding_dim": 128
nano config/train/vqvae.json
```

---

## Training Models

### Training Workflow Overview

The typical workflow is:

1. **Train VQ-VAE** → Learn discrete latent representations
2. **Train PixelCNN** (optional) → Model the prior over latent codes
3. **Create Continuous Mixtures** (optional) → Generate exact mixture model

### Training VQ-VAE

The VQ-VAE is the foundation model that must be trained first.

#### Command

```bash
python train.py --model_type vqvae
```

#### What Happens During Training

1. **Initialization**: Model architecture is created with parameters from `config/train/vqvae.json`
2. **Data Loading**: Dataset is loaded and preprocessed (default: CIFAR-10)
3. **Training Loop**: 
   - Images are encoded into latent space
   - Latent codes are quantized to nearest codebook vectors
   - Decoder reconstructs images from quantized codes
   - Loss is computed (reconstruction loss + commitment loss)
   - Model parameters are updated via backpropagation
4. **Checkpointing**: Model checkpoints saved every N epochs
5. **Logging**: Training metrics logged to TensorBoard

#### Monitoring Training

Launch TensorBoard to monitor training progress:

```bash
tensorboard --logdir=./lightning_logs
```

Then open your browser to `http://localhost:6006`

#### Expected Training Time

- **CIFAR-10 on RTX 3070**: ~2-3 hours for 100 epochs
- **CIFAR-10 on RTX 4090**: ~1-1.5 hours for 100 epochs
- **CPU training**: Not recommended (would take 20+ hours)

#### Training Output

During training, you'll see output like:

```
Epoch 1/100: 100%|████████| 391/391 [02:15<00:00, 2.88it/s, loss=0.0234, recon_loss=0.0220, vq_loss=0.0014]
Epoch 2/100: 100%|████████| 391/391 [02:14<00:00, 2.91it/s, loss=0.0198, recon_loss=0.0186, vq_loss=0.0012]
```

Key metrics to monitor:
- **loss**: Total combined loss
- **recon_loss**: Reconstruction quality (lower is better)
- **vq_loss**: Vector quantization loss

#### Checkpoint Location

Trained models are saved in:
```
checkpoints/vqvae/
├── epoch=10.ckpt
├── epoch=20.ckpt
└── best_model.ckpt
```

### Training PixelCNN

After VQ-VAE training completes, train the PixelCNN prior.

#### Prerequisites

You must have a trained VQ-VAE model checkpoint.

#### Command

```bash
python train.py --model_type pcnn
```

#### What Happens During Training

1. **VQ-VAE Loading**: Previously trained VQ-VAE is loaded (in eval mode)
2. **Latent Extraction**: Training images are encoded to latent codes using VQ-VAE
3. **PixelCNN Training**: PixelCNN learns to model the distribution of these latent codes
4. **Autoregressive Modeling**: Each latent code is predicted conditioned on previous codes

#### Expected Training Time

- **CIFAR-10 on RTX 3070**: ~4-5 hours for 200 epochs
- **CIFAR-10 on RTX 4090**: ~2-3 hours for 200 epochs

#### Monitoring PixelCNN Training

Key metrics:
- **nll_loss**: Negative log-likelihood (lower is better)
- **bits_per_dim**: Information-theoretic measure of model quality
- **perplexity**: Average uncertainty in predictions

### Training Continuous Mixtures

This step creates an exact mixture model through latent space enumeration.

#### Command

```bash
python train.py --model_type cm
```

**Note:** This doesn't involve traditional training but rather exact enumeration of the latent space.

#### What Happens

1. **Complete Enumeration**: All possible latent code combinations are enumerated
2. **Probability Computation**: Exact probabilities are computed for each configuration
3. **Batch Saving**: Results are saved in batches to manage memory

#### Expected Time

- Depends heavily on latent space size
- For standard CIFAR-10 setup: 1-2 hours
- Larger latent spaces: can take several hours

---

## Evaluating Models

### Evaluation Overview

Evaluation assesses trained model performance on held-out test data.

### Evaluating VQ-VAE

#### Command

```bash
python eval.py --model_type vqvae
```

#### Metrics Computed

1. **Reconstruction Error**: MSE between input and reconstructed images
2. **SSIM (Structural Similarity Index)**: Perceptual quality metric
3. **Codebook Usage**: Percentage of codebook vectors actually used
4. **Latent Space Visualizations**: t-SNE/UMAP plots of learned codes

#### Expected Output

```
=== VQ-VAE Evaluation Results ===
Test Reconstruction Error: 0.0156
Test SSIM: 0.892
Codebook Usage: 87.3%
Generating sample reconstructions...
Saved reconstructions to: results/vqvae/reconstructions.png
```

#### Interpreting Results

- **Good Reconstruction Error**: < 0.02 for CIFAR-10
- **Good SSIM**: > 0.85
- **Good Codebook Usage**: > 80% (indicates most codes are being utilized)

### Evaluating PixelCNN

#### Command

```bash
python eval.py --model_type pcnn
```

#### Metrics Computed

1. **Negative Log-Likelihood**: Lower is better
2. **Sample Generation**: Generate new images from the prior
3. **Conditional Sampling**: Test conditional generation capabilities

#### Expected Output

```
=== PixelCNN Evaluation Results ===
Test NLL: 2.143
Bits per dimension: 3.09
Generating samples...
Saved samples to: results/pcnn/samples.png
```

### Evaluating Continuous Mixtures

#### Command

```bash
python eval.py --model_type cm
```

#### Metrics Computed

1. **Exact Log-Likelihood**: Tractable probability computation
2. **Mixture Component Analysis**: Analysis of learned mixture components
3. **Conditional Probabilities**: Test exact inference capabilities

---

## Advanced Operations

### Latent Space Marginalization

Create an exact mixture model through complete enumeration:

#### Command

```bash
python marginalize.py
```

#### What This Does

1. Enumerates all possible latent configurations
2. Computes exact probabilities for each configuration
3. Saves results in batches for memory efficiency
4. Creates a tractable mixture model for exact inference

#### Output

```
Marginalizing latent space...
Batch 1/1000: 100%|████████| [00:15<00:00]
Batch 2/1000: 100%|████████| [00:15<00:00]
...
Marginalization complete!
Saved mixture model batches to: results/mixtures/
```

#### Use Cases

- Exact probabilistic inference
- Inpainting with guaranteed optimal solutions
- Uncertainty quantification

### PixelCNN CDF Visualization

Visualize the cumulative distribution function learned by PixelCNN:

#### Command

```bash
python pcnn_cdf.py
```

#### Output

Generates plots showing:
- CDF curves for each latent position
- Distribution shape analysis
- Comparison with empirical distributions

Saved to: `results/pcnn/cdf_plots.png`

### Inpainting Operations

#### Single Query Inpainting

Fill in missing regions of an image:

```bash
python inpaint.py --image_path path/to/image.png --mask_path path/to/mask.png
```

**Parameters:**
- `--image_path`: Path to input image with missing regions
- `--mask_path`: Binary mask indicating regions to inpaint (1 = inpaint, 0 = keep)

#### Batch Inpainting

Process multiple inpainting queries efficiently:

```bash
python inpaint_batched.py --image_dir path/to/images/ --mask_dir path/to/masks/
```

#### Inpainting Output

```
Processing inpainting query...
Computing optimal latent configuration...
Decoding result...
Saved inpainted image to: results/inpainting/result.png
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: CUDA Out of Memory

**Error Message:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**
1. Reduce batch size in configuration file
2. Reduce model dimensions (embedding_dim, hidden_dim)
3. Use gradient accumulation
4. Clear CUDA cache: `torch.cuda.empty_cache()`

#### Issue: Poor Reconstruction Quality

**Symptoms:**
- Blurry reconstructions
- High reconstruction error
- Low SSIM scores

**Solutions:**
1. Train for more epochs
2. Increase codebook size (num_embeddings)
3. Increase embedding dimension
4. Reduce commitment cost (allows more flexibility)
5. Check learning rate (try 1e-4 to 3e-4 range)

#### Issue: Low Codebook Usage

**Symptoms:**
- Codebook usage < 50%
- Many unused codes

**Solutions:**
1. Increase decay parameter (0.99 → 0.995)
2. Adjust commitment cost (try 0.1 to 0.5)
3. Increase batch size for better statistics
4. Add codebook reset mechanism

#### Issue: PixelCNN Not Converging

**Symptoms:**
- NLL not decreasing
- Poor sample quality

**Solutions:**
1. Verify VQ-VAE is properly trained first
2. Increase number of layers (15 → 20)
3. Adjust learning rate
4. Train for more epochs (200 → 400)

#### Issue: Training Crashes or Hangs

**Solutions:**
1. Check GPU memory usage: `nvidia-smi`
2. Reduce num_workers in DataLoader
3. Update CUDA drivers
4. Verify PyTorch installation

### Getting Help

If issues persist:
1. Check GitHub Issues: https://github.com/ahadzic7/test-vqvae/issues
2. Provide full error message and stack trace
3. Include configuration files used
4. Specify system specifications (GPU, CUDA version, PyTorch version)

---

## Expected Results

### VQ-VAE Results on CIFAR-10

After 100 epochs, you should expect:

| Metric | Expected Value |
|--------|---------------|
| Reconstruction Error | 0.015 - 0.020 |
| SSIM | 0.85 - 0.92 |
| Codebook Usage | 75% - 90% |
| Training Time | 2-3 hours (RTX 3070) |

**Visual Quality:**
- Reconstructions should preserve main objects and colors
- Fine details may be slightly blurred
- No major artifacts or distortions

### PixelCNN Results

After 200 epochs:

| Metric | Expected Value |
|--------|---------------|
| Test NLL | 2.0 - 2.5 |
| Bits per Dim | 2.8 - 3.5 |
| Training Time | 4-5 hours (RTX 3070) |

**Sample Quality:**
- Generated samples should be recognizable objects
- Color distributions match training data
- Some samples may have artifacts (normal for autoregressive models)

### Continuous Mixtures Results

| Metric | Expected Value |
|--------|---------------|
| Exact Log-Likelihood | Available for all test points |
| Enumeration Time | 1-2 hours |
| Inpainting Quality | Optimal given the model |

---

## Additional Tips

### Speeding Up Training

1. **Use Mixed Precision Training**: Add to config
2. **Multi-GPU Training**: Distribute across GPUs
3. **Optimized Data Loading**: Increase num_workers
4. **Smaller Validation Sets**: Validate less frequently

### Improving Results

1. **Data Augmentation**: Add random flips, crops
2. **Learning Rate Scheduling**: Cosine annealing, step decay
3. **Architecture Modifications**: Deeper networks, residual connections
4. **Ensemble Methods**: Train multiple models

### Experiment Tracking

Use TensorBoard effectively:
```bash
# Compare multiple runs
tensorboard --logdir=./lightning_logs --port=6006
```

Log additional metrics:
- Gradient norms
- Learning rate schedules
- Custom visualizations

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{vqvae-tpm,
  author = {Hadzic, A.},
  title = {VQ-VAE with PixelCNN and Continuous Mixtures},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/ahadzic7/test-vqvae}
}
```

---

## License

Please refer to the LICENSE file in the repository.

---

## Support

For questions, issues, or contributions:
- **GitHub Issues**: https://github.com/ahadzic7/test-vqvae/issues
- **Pull Requests**: Contributions welcome!

---

## Acknowledgments

This implementation builds on research in:
- Vector Quantized Variational Autoencoders (VQ-VAE)
- PixelCNN autoregressive models
- Tractable probabilistic models

---

**Last Updated**: November 2024  
**Repository**: https://github.com/ahadzic7/test-vqvae  
**Status**: Active Development