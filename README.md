# Distilling VQ-VAE into Tractable Probabilistic Models

## Overview

The goal of this repository is to show that **VQ-VAE models with PixelCNN priors can be distilled into tractable probabilistic models** without sacrificing expressiveness.

This repository contains the code for the paper: *Hadžić A, Papež M, Šmídl V, Pevný T. [Distillation of a tractable model from the VQ-VAE](https://arxiv.org/abs/2509.01400).*




We:
1. Train a **VQ-VAE** with a **PixelCNN prior** over its discrete latent codes.
2. **Distill** this model by selecting a subset of high-probability latent configurations.
3. Represent the distilled model as a **probabilistic circuit**, enabling **exact probabilistic inference** (e.g., exact marginals and conditionals).
4. Compare against **state-of-the-art tractable baselines**:
   - **Continuous Mixtures (CM)**  
   - **Einsum Networks (Einets)**  

Our experiments evaluate **density estimation** and **conditional generation**, demonstrating that distilled VQ-VAE models can compete with, and in some cases rival, these tractable baselines—challenging the view of VQ-VAE as inherently intractable.

---

## Implemented Models

This repository includes:

- **VQ-VAE + PixelCNN prior**
  - Vector-quantized autoencoder with discrete latent space.
  - PixelCNN as an autoregressive prior over discrete codes.
  - Serves as the *teacher* model for distillation.

- **Distilled Mixture model as a Probabilistic Circuit**
  - Obtained by selecting high-probability latent assignments.
  - Structured as a probabilistic circuit to support:
    - Exact marginalization
    - Exact conditioning
    - Efficient likelihood queries

- **Baselines (State-of-the-art Tractable Models)**
  - **Continuous Mixtures (CM)**
  - **Einets**

---

## Experiments for:
  - Density estimation
  - Conditional generation (e.g., inpainting)

---


## Environment Setup

### Step 1: Clone Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/ahadzic7/test-vqvae.git
cd test-vqvae
```

### Step 2: Create Conda Environment from YAML

Create the environment from the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

### Step 3: Activate Environment

Activate the newly created environment:

```bash
conda activate vqvae-tpm
```

### Step 4: Install Python Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 5: Verify Installation

Check that all packages are installed correctly:

```bash
python -c "import torch; import lightning; import tensorboard; import matplotlib; print('All dependencies installed successfully!')"
```

---

# Reproducing Experiments

This repository contains all models and scripts necessary to reproduce the experiments from the paper.

## Model Configurations

All models compared in the paper are located in the `experimental_settings` folder as JSON configuration files. Each config defines a model and its hyperparameters.

2. To run all experiments from the paper sequentially:
   ```bash
   python recreate.py
   ```
   This runs the full pipelines for every config in `experimental_settings`.
3. To train a single model:
   ```bash
   python train.py -mt [cm, einet, vqvae-pcnn-dm] -c PATH_TO_CONFIG_FILE
   ```
4. Trained models are saved to the `models` folder; pipeline outputs are saved to `results`.

5. The `results` folder contains for each model:
   - images of:
    - samples,
    - inpaints,
    - means of Gaussian mixtures,
   - performance results as CSV files on the test sets.
6. To evaluate an individual model:
   ```bash
   python eval.py -mt [cm, einet, vqvae-pcnn-dm] -c PATH_TO_CONFIG_FILE
   ```
7. To plot and compare model performance:
   ```bash
   python plot.py
   ```

---

## License

Please refer to the LICENSE file in the repository.


