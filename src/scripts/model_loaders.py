import os
import torch
import numpy as np
from src.utilities import get_device, latent_dim, rb_path
from src.mm.GaussianMixture import GaussianMixture as GMM
from src.scripts.PathBuilder import PathBuilder


class ModelLoader:
    """Centralized model loading with consistent naming and path handling."""
    
    def __init__(self, models_dir, config: dict):
        self.models_dir = models_dir
        self.config = config
        self.device = get_device()
        
        # Extract common config elements
        self.data = config["input_data"]
        self.dataset = self.data["dataset"]
        self.pixel_type = self.data["pixel_representation"]["type"]
        self.seed = config["seed"]
        
        # Initialize path builder
        self.path_builder = PathBuilder(models_dir, self.dataset, self.pixel_type)
    
    def _format_params(self, vqvae_config: dict):
        """Extract and format VQ-VAE parameters."""
        arch = vqvae_config["architecture"]
        if "layers" in vqvae_config:
            layers = f'{vqvae_config["layers"]}l'
        else:
            layers = f'{len(vqvae_config["filters"])+1}l'

        ld = latent_dim(vqvae_config["latent_shape"])
        codebook = f'cb_size-{vqvae_config.get("codebook_size", None)}'
        beta = f'b_{vqvae_config.get("beta", None)}'
        seed = f'seed_{self.seed}'
        return arch, layers, ld, codebook, beta, seed
    
    def get_training_paths(self, model_key = "vqvae"):
        """
        Get training path components for PyTorch Lightning trainer.
        
        This method reuses the existing path building logic and splits it into
        the components needed by PyTorch Lightning's trainer.
        
        Args:
            model_key: Config key for model (e.g., 'vqvae', 'vae', 'pcnn', etc.)
        
        Returns:
            Tuple of (log_folder, name, version) for trainer initialization
        
        Example:
            loader = ModelLoader(models_dir, config)
            log_folder, name, version = loader.get_training_paths("vae")
            trainer = vae_trainer(log_folder, epochs, name=name, version=version)
            trainer.fit(model, trainl, validl)
        """
        # Get the full path using existing methods
        if model_key == "vqvae":
            vqvae_config = self.config["vqvae"]
            arch, layers, ld, cb, beta, seed = self._format_params(vqvae_config)
            full_path = self.path_builder.vqvae_path(arch, layers, ld, cb, beta, seed)

        elif model_key == "vqvae_rec":
            vqvae_config = self.config["vqvae_rec"]
            arch, layers, ld, cb, beta, seed = self._format_params(vqvae_config)
            full_path = self.path_builder.vqvae_rec_path(arch, layers, ld, cb, beta, seed)

        elif model_key == "vae":
            vae_config = self.config["vae"]
            arch = vae_config["architecture"]
            
            if "layers" in vae_config:
                layers = f'{vae_config["layers"]}l'
            else:
                layers = f'{len(vae_config["filters"])+1}l'

            ld = latent_dim(vae_config["latent_shape"])
            beta = f'b_{vae_config["beta"]}'
            seed = f'seed_{self.seed}'
            full_path = self.path_builder.vae_path(arch, layers, ld, beta, seed)

        elif model_key == "ae":
            ae_config = self.config["ae"]
            arch = ae_config["architecture"]
            
            if "layers" in ae_config:
                layers = f'{ae_config["layers"]}l'
            else:
                layers = f'{len(ae_config["filters"])+1}l'

            ld = latent_dim(ae_config["latent_shape"])
            seed = f'seed_{self.seed}'
            full_path = self.path_builder.ae_path(arch, layers, ld, seed)

        elif model_key in ["pcnn", "pixel_cnn"]:
            vqvae_config = self.config["vqvae"]
            arch, layers, ld, cb, beta, seed = self._format_params(vqvae_config)
            labeling = self.data["supervised"]["use_labeling"]
            model_type = "cond_pixel_cnn" if labeling else "pixel_cnn"
            full_path = self.path_builder.pcnn_path(model_type, arch, layers, ld, cb, beta, seed)
            
        elif model_key in ["psnail", "pixel_snail"]:
            vqvae_config = self.config["vqvae"]
            arch, layers, ld, cb, _, _ = self._format_params(vqvae_config)
            labeling = self.data["supervised"]["use_labeling"]
            model_type = "cond_pixel_snail" if labeling else "pixel_snail"
            full_path = self.path_builder.psnail_path(model_type, arch, layers, ld, cb)
            
        elif model_key == "cm":
            cm_config = self.config["model"]
            lat_shape = cm_config["latent_shape"]
            ld = latent_dim(lat_shape)
            nb = cm_config["n_bins"]
            nl = cm_config["layers"]
            seed = self.config["seed"]
            full_path = self.path_builder.cm_path(nl, ld, nb, seed)
            
        elif model_key == "classifier":
            full_path = self.path_builder.classifier_path()
            
        else:
            raise ValueError(f"Unknown model_key: {model_key}. "
                           f"Supported: vqvae, vae, pcnn, psnail, cm, classifier")
        
        # Split the full path into trainer components
        return PathBuilder.split_training_path(full_path)
    
    def _ensure_directory(self, file_path) -> None:
        """Create directory structure if it doesn't exist."""
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
    
    def load_vqvae(self):
        """Load VQ-VAE model."""
        vqvae_config = self.config.get("vqvae", self.config.get("model", None))
        arch, layers, ld, cb, beta, seed = self._format_params(vqvae_config)
        K = vqvae_config["codebook_size"]
        
        model_path = self.path_builder.vqvae_path(arch, layers, ld, cb, beta, seed)
        vqvae = torch.load(model_path, weights_only=False).to(self.device)
        vqvae.eval()
        
        model_name = f"{self.pixel_type}_{ld}_{K}_{layers}_{arch}_{beta}_{seed}"
        return vqvae, model_name
    
    def load_vqvae_rec(self):
        """Load VQ-VAE_rec model."""
        vqvae_config = self.config.get("vqvae_rec", self.config.get("model", None))
        arch, layers, ld, cb, beta, seed = self._format_params(vqvae_config)
        K = vqvae_config["codebook_size"]
        
        model_path = self.path_builder.vqvae_rec_path(arch, layers, ld, cb, beta, seed)
        vqvae = torch.load(model_path, weights_only=False).to(self.device)
        vqvae.eval()
        
        model_name = f"{self.pixel_type}_{ld}_{K}_{layers}_{arch}_{beta}_{seed}"
        return vqvae, model_name
    

    def load_vqvae_rec(self):
        """Load VQ-VAE_rec model."""
        vqvae_config = self.config.get("vqvae_rec", self.config.get("model", None))
        arch, layers, ld, cb, beta, seed = self._format_params(vqvae_config)
        K = vqvae_config["codebook_size"]
        
        model_path = self.path_builder.vqvae_path(arch, layers, ld, cb, beta, seed)
        vqvae = torch.load(model_path, weights_only=False).to(self.device)
        vqvae.eval()
        
        model_name = f"{self.pixel_type}_{ld}_{K}_{layers}_{arch}_{beta}_{seed}"
        return vqvae, model_name
    

    def load_pcnn(self):
        """Load PixelCNN model (conditional or standard)."""
        vqvae_config = self.config.get("vqvae", self.config.get("model", None))
        arch, layers, ld, cb, beta, seed = self._format_params(vqvae_config)
        K = vqvae_config["codebook_size"]
        
        # Determine if conditional
        labeling = self.data["supervised"]["use_labeling"]
        model_type = "cond_pixel_cnn" if labeling else "pixel_cnn"
        
        model_path = self.path_builder.pcnn_path(model_type, arch, layers, ld, cb, beta, seed)
        pcnn = torch.load(model_path, weights_only=False).to(self.device)
        pcnn.eval()
        
        model_name = f"{self.pixel_type}_{ld}_{K}_{arch}_{model_type}_{beta}_{seed}"
        return pcnn, model_name
    
    def load_psnail(self):
        """Load PixelSNAIL model."""
        vqvae_config = self.config.get("vqvae", self.config.get("model", None))
        arch, layers, ld, cb, _, _ = self._format_params(vqvae_config)
        K = vqvae_config["codebook_size"]
        
        # Determine if conditional
        labeling = self.data["supervised"]["use_labeling"]
        model_type = "cond_pixel_snail" if labeling else "pixel_snail"
        
        model_path = self.path_builder.psnail_path(model_type, arch, layers, ld, cb)
        psnail = torch.load(model_path, weights_only=False).to(self.device)
        psnail.eval()
        
        model_name = f"{self.pixel_type}_{ld}_{K}_{arch}_{model_type}"
        return psnail, model_name


    def load_theta(self, seed= None, n= None):
        """Load theta parameters from latent codes."""
        config = self.config.get("model", self.config.get("vqvae", self.config.get("vae", self.config.get("ae", None))))
        arch, layers, ld, cb, beta, config_seed = self._format_params(config)
        
        dt = self.config["distillation_type"]
        prior = self.config["prior"]
        n_components = self.config["n_components"]
        
        # Build latent file path
        if prior == "vae":
            # arch, layers, ld, beta, seed, dt, prior
            dm_base = self.path_builder.continuous_dm_path(arch, layers, ld, beta, config_seed, dt, prior)

            if seed is not None:
                latent_file = f'{dm_base}/latents-{n_components}-{0}.ckpt.npy'
            else:
                latent_file = f'{dm_base}/latents-{n_components}.ckpt.npy'
            
            # Load latents and compute theta
            vae, _ = self.load_vae()
            arr = np.load(latent_file)
            latents = torch.from_numpy(arr).to(vae.device)
            
            if n is not None:
                print(f"Limiting latents to first {n} samples")
                print(f"Original shape: {latents.shape} -> Sliced shape: {latents[:n].shape}")
                latents = latents[:n]
            
            print(f"Final latents shape: {latents.shape}")
            theta = vae.params(latent=latents)
            
            model_name = self.path_builder.continuous_dm_name(arch, layers, ld, beta, seed, dt, prior)
        elif prior == "ae":
            # arch, layers, ld, seed, dt, prior
            dm_base = self.path_builder.continuous_dm_path_ae(arch, layers, ld, config_seed, dt, prior)

            if seed is not None:
                latent_file = f'{dm_base}/latents-{n_components}-{0}.ckpt.npy'
            else:
                latent_file = f'{dm_base}/latents-{n_components}.ckpt.npy'
            
            # Load latents and compute theta
            ae, _ = self.load_ae()
            arr = np.load(latent_file)
            latents = torch.from_numpy(arr).to(ae.device)
            
            if n is not None:
                print(f"Limiting latents to first {n} samples")
                print(f"Original shape: {latents.shape} -> Sliced shape: {latents[:n].shape}")
                latents = latents[:n]
            
            print(f"Final latents shape: {latents.shape}")
            theta = ae.params(latent=latents)
            
            model_name = self.path_builder.continuous_dm_name_ae(arch, layers, ld, seed, dt, prior)
        else:
            path_fun = self.path_builder.discrete_dm_path

            dm_base = path_fun(arch, layers, ld, cb, beta, config_seed, dt, prior)
            if seed is not None:
                latent_file = f'{dm_base}/latents-{n_components}-{0}.ckpt.npy'
            else:
                latent_file = f'{dm_base}/latents-{n_components}.ckpt.npy'
            
            # Load latents and compute theta
            vqvae, _ = self.load_vqvae()
            print(vqvae.encoder)
            print("/////////////////////")
            print(vqvae.decoder)
            
            arr = np.load(latent_file)
            latents = torch.from_numpy(arr).to(vqvae.device)
            
            if n is not None:
                print(f"Limiting latents to first {n} samples")
                print(f"Original shape: {latents.shape} -> Sliced shape: {latents[:n].shape}")
                latents = latents[:n]
            
            print(f"Final latents shape: {latents.shape}")
            theta = vqvae.params(latents)
            
            model_name = self.path_builder.discrete_dm_name(arch, layers, ld, cb, beta, seed, dt, prior)

        return theta, model_name


    def load_dm(self, seed= None):
        """Load Gaussian Mixture Model from theta parameters."""
        theta, model_name = self.load_theta(seed=seed)
        dm = GMM(theta).to(self.device)
        return dm, model_name
    
    def load_cm(self):
        """Load Categorical Mixture model."""
        cm_config = self.config["model"]
        lat_shape = cm_config["latent_shape"]
        ld = latent_dim(lat_shape)
        nb = cm_config["n_bins"]
        nl = cm_config["layers"]
        seed = self.config["seed"]
        
        model_path = self.path_builder.cm_path(nl, ld, nb, seed)
        cm = torch.load(model_path, weights_only=False).to(self.device)
        
        # Configure model
        cm.n_chunks = 2**10
        cm.sampler.n_bins = 2**14
        cm.eval()
        
        model_name = self.path_builder.cm_name(nl, ld, nb, seed)
        return cm, model_name
    
    def load_vae(self):
        """Load VAE model."""
        vae_config = self.config.get("vae", self.config.get("model", None))
        arch = vae_config["architecture"]
        layers = f'{vae_config["layers"]}l'
        ld = latent_dim(vae_config["latent_shape"])
        beta = f'b_{vae_config["beta"]}'
        seed = f'seed_{self.seed}'
        
        model_path = self.path_builder.vae_path(arch, layers, ld, beta, seed)
        vae = torch.load(model_path, weights_only=False).to(self.device)
        vae.eval()
        
        model_name = f"{self.pixel_type}_{ld}_{layers}_{arch}_{beta}_{seed}"
        return vae, model_name
    
    def load_ae(self):
        """Load AE model."""
        ae_config = self.config.get("ae", self.config.get("model", None))
        arch = ae_config["architecture"]
        layers = f'{ae_config["layers"]}l'
        ld = latent_dim(ae_config["latent_shape"])
        seed = f'seed_{self.seed}'
        
        model_path = self.path_builder.ae_path(arch, layers, ld, seed)
        ae = torch.load(model_path, weights_only=False).to(self.device)
        ae.eval()
        
        model_name = f"{self.pixel_type}_{ld}_{layers}_{arch}_{seed}"
        return ae, model_name
    

    def load_classifier(self):
        """Load classifier model."""
        model_path = self.path_builder.classifier_path()
        classifier = torch.load(model_path, weights_only=False).to(self.device)
        classifier.eval()
        
        model_name = f"classifier_{self.pixel_type}_{self.dataset}"
        return classifier, model_name
    
    def load_einet(self):
        """Load EinsumNet model with memory checks."""
        result_path, model_name = rb_path(self.models_dir, self.config)
        print(f"Loading EiNet from: {result_path}\n")
        
        model_file = os.path.join(result_path, 'einet.mdl')
        record_file = os.path.join(result_path, 'record.pkl')
        
        if not (os.path.isfile(model_file) and os.path.isfile(record_file)):
            print("Model not trained or files are missing!")
            return None, None
        
        # Memory check
        if not self._check_memory(model_file):
            print("Insufficient memory to load model safely.")
            return None, None
        
        try:
            spn = torch.load(model_file, weights_only=False)
            print("Successfully loaded EiNet model")
            return spn, model_name
        except RuntimeError as e:
            print(f"RuntimeError while loading model: {e}")
            return None, None
    
    @staticmethod
    def _check_memory(model_file) -> bool:
        """Check if there's enough memory to load the model."""
        import psutil
        
        file_size_mb = os.path.getsize(model_file) / (1024**2)
        available_mb = psutil.virtual_memory().available / (1024**2)
        
        print(f"Model size: {file_size_mb:.1f} MB | Available RAM: {available_mb:.1f} MB")
        
        # Require 20% buffer
        return file_size_mb < available_mb * 0.8


# Convenience functions for backward compatibility
def load_vqvae(models_dir, config):
    return ModelLoader(models_dir, config).load_vqvae()

def load_vqvae_rec(models_dir, config):
    return ModelLoader(models_dir, config).load_vqvae_rec()



def load_pcnn(models_dir, config):
    return ModelLoader(models_dir, config).load_pcnn()


def load_psnail(models_dir, config):
    return ModelLoader(models_dir, config).load_psnail()


def load_theta(models_dir, config, seed = None, n = None):
    return ModelLoader(models_dir, config).load_theta(seed=seed, n=n)


def load_dm(models_dir, config, seed = None):
    return ModelLoader(models_dir, config).load_dm(seed=seed)


def load_cm(models_dir, config):
    return ModelLoader(models_dir, config).load_cm()



def load_ae(models_dir, config):
    return ModelLoader(models_dir, config).load_ae()



def load_vae(models_dir, config):
    return ModelLoader(models_dir, config).load_vae()


def load_classifier(models_dir, config):
    return ModelLoader(models_dir, config).load_classifier()


def load_einet(models_dir, config):
    return ModelLoader(models_dir, config).load_einet()


def load_vqvae(models_dir, config):
    return ModelLoader(models_dir, config).load_vqvae()

