# vqvae-tpm

Set up the environment.
```
conda create --name vqvae-tpm python=3.10

source activate vqvae-tpm

torch==2.5.1+cu124
torchaudio==2.5.1+cu124
torchvision==0.20.1+cu124

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install lightning
pip install tensorboard
pip install matplotlib
```

To run the training execute the following command:

`python train.py --model_type [vqvae, pcnn, cm]`

- vqvae stands for the Vector Quantized Variational Autoencoder.
- pcnn stands for the PixelCNN.
- cm stands for the Continuous Mixtures 

To evaluate a given model run the following command:

`python eval.py --model_type [vqvae, pcnn, cm]`

Training settings and hyperparameters are defined in `config/train` and `config/eval` folders as json files.

`marginalize.py` runs the enumeration of the latent space and creates an exact mixture model saved in batches.

`pcnn_cdf.py` plots the CDF function of a trained PixelCNN.

`inpaint.py` and `inpaint_batched.py` run inpainting querries on mixtures.


