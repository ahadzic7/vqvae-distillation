import torch
import os
import numpy as np
from src.scripts.PathBuilder import PathBuilder
from src.utilities import latent_dim, rb_path


class ModelSaver:
    """Centralized model saving with identical interface to ModelLoader."""
    
    def __init__(self, models_dir: str, config: dict):
        self.models_dir = models_dir
        self.config = config
        
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
        codebook = f'cb_size-{vqvae_config["codebook_size"]}'
        beta = f'b_{vqvae_config["beta"]}'
        seed = f'seed_{self.seed}'
        return arch, layers, ld, codebook, beta, seed
    
    def _ensure_directory(self, file_path: str) -> None:
        """Create directory structure if it doesn't exist."""
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
    
    def save_vqvae(self, model: torch.nn.Module):
        """Save VQ-VAE model."""
        vqvae_config = self.config["vqvae"]
        arch, layers, ld, cb, beta, seed = self._format_params(vqvae_config)
        K = vqvae_config["codebook_size"]
        
        model_path = self.path_builder.vqvae_path(arch, layers, ld, cb, beta, seed)
        self._ensure_directory(model_path)
        torch.save(model, model_path)
        
        model_name = f"{self.pixel_type}_{ld}_{K}_{layers}_{arch}_{beta}_{seed}"
        return model_path, model_name
    

    def save_vqvae_rec(self, model: torch.nn.Module):
        """Save VQ-VAE_rec model."""
        vqvae_config = self.config["vqvae_rec"]
        arch, layers, ld, cb, beta, seed = self._format_params(vqvae_config)
        K = vqvae_config["codebook_size"]
        
        model_path = self.path_builder.vqvae_path(arch, layers, ld, cb, beta, seed)
        self._ensure_directory(model_path)
        torch.save(model, model_path)
        
        model_name = f"{self.pixel_type}_{ld}_{K}_{layers}_{arch}_{beta}_{seed}"
        return model_path, model_name
    
    def save_pcnn(self, model: torch.nn.Module):
        """Save PixelCNN model (conditional or standard)."""
        vqvae_config = self.config["vqvae"]
        arch, layers, ld, cb, beta, seed = self._format_params(vqvae_config)
        K = vqvae_config["codebook_size"]
        
        # Determine if conditional
        labeling = self.data["supervised"]["use_labeling"]
        model_type = "cond_pixel_cnn" if labeling else "pixel_cnn"
        
        model_path = self.path_builder.pcnn_path(model_type, arch, layers, ld, cb, beta, seed)
        self._ensure_directory(model_path)
        torch.save(model, model_path)
        
        model_name = f"{self.pixel_type}_{ld}_{K}_{arch}_{model_type}_{beta}_{seed}"
        return model_path, model_name
    
    def save_psnail(self, model: torch.nn.Module):
        """Save PixelSNAIL model."""
        vqvae_config = self.config["vqvae"]
        arch, layers, ld, cb, _, _ = self._format_params(vqvae_config)
        K = vqvae_config["codebook_size"]
        
        # Determine if conditional
        labeling = self.data["supervised"]["use_labeling"]
        model_type = "cond_pixel_snail" if labeling else "pixel_snail"
        
        model_path = self.path_builder.psnail_path(model_type, arch, layers, ld, cb)
        self._ensure_directory(model_path)
        torch.save(model, model_path)
        
        model_name = f"{self.pixel_type}_{ld}_{K}_{arch}_{model_type}"
        return model_path, model_name
    
    
    def save_discrete_latents(self, latents, seed = None):
        vqvae_config = self.config.get("vqvae", self.config.get("model", None))

        arch, layers, ld, cb, _, config_seed = self._format_params(vqvae_config)
        K = vqvae_config["codebook_size"]
        beta = f'b_{vqvae_config["beta"]}'

        dt = self.config["distillation_type"]
        prior = self.config["prior"]
        n_components = self.config["n_components"]
        
        # Build latent file path
        dm_base = self.path_builder.discrete_dm_path(arch, layers, ld, cb, beta, config_seed, dt, prior)
        if seed is not None:
            latent_file = f'{dm_base}/latents-{n_components}-{seed}.ckpt.npy'
        else:
            latent_file = f'{dm_base}/latents-{n_components}.ckpt.npy'
        
        self._ensure_directory(latent_file)
        
        np.save(latent_file, latents.cpu().numpy())
        
        model_name = f"{self.pixel_type}_{ld}_{K}_{layers}_{arch}_{beta}_{prior}_{dt}_{config_seed}"
        return latent_file, model_name
    
    def save_continuous_latents(self, latents, seed = None):
        vae_config = self.config.get("vae", self.config.get("model", self.config.get("ae", None)))
        arch = vae_config["architecture"]

        if "layers" in vae_config:
            layers = f'{vae_config["layers"]}l'
        else:
            layers = f'{len(vae_config["filters"])+1}l'

        ld = latent_dim(vae_config["latent_shape"])
        config_seed = f'seed_{self.seed}'
        dt = self.config["distillation_type"]
        prior = self.config["prior"]
        n_components = self.config["n_components"]
        
        # Build latent file path
        if prior == "ae":
            dm_base = self.path_builder.continuous_dm_path_ae(arch, layers, ld, config_seed, dt, prior)
            model_name = f"{self.pixel_type}_{ld}_{layers}_{arch}_{prior}_{dt}_{config_seed}"
        else:    
            beta = f'b_{vae_config["beta"]}'
            dm_base = self.path_builder.continuous_dm_path(arch, layers, ld, beta, config_seed, dt, prior)
            model_name = f"{self.pixel_type}_{ld}_{layers}_{arch}_{beta}_{prior}_{dt}_{config_seed}"

        
        if seed is not None:
            latent_file = f'{dm_base}/latents-{n_components}-{seed}.ckpt.npy'
        else:
            latent_file = f'{dm_base}/latents-{n_components}.ckpt.npy'
        
        self._ensure_directory(latent_file)
        
        np.save(latent_file, latents.cpu().numpy())
        
        return latent_file, model_name
    
    def save_cm(self, model: torch.nn.Module):
        cm_config = self.config["model"]
        lat_shape = cm_config["latent_shape"]
        ld = latent_dim(lat_shape)
        nb = cm_config["n_bins"]
        nl = cm_config["layers"]
        seed = self.config["seed"]
        
        model_path = self.path_builder.cm_path(nl, ld, nb, seed)
        self._ensure_directory(model_path)
        torch.save(model, model_path)
        
        model_name = f"{self.pixel_type}_{ld}_{nl}_{nb}"
        return model_path, model_name
    
    def save_vae(self, model: torch.nn.Module):
        """Save VAE model."""
        vae_config = self.config["vae"]
        arch = vae_config["architecture"]

        if "layers" in vae_config:
            layers = f'{vae_config["layers"]}l'
        else:
            layers = f'{len(vae_config["filters"])+1}l'

        ld = latent_dim(vae_config["latent_shape"])
        beta = f'b_{vae_config["beta"]}'
        seed = f'seed_{self.seed}'
        
        model_path = self.path_builder.vae_path(arch, layers, ld, beta, seed)
        self._ensure_directory(model_path)
        torch.save(model, model_path)
        
        model_name = f"{self.pixel_type}_{ld}_{layers}_{arch}_{beta}_{seed}"
        return model_path, model_name
    
    def save_ae(self, model: torch.nn.Module):
        """Save AE model."""
        ae_config = self.config["ae"]
        arch = ae_config["architecture"]

        if "layers" in ae_config:
            layers = f'{ae_config["layers"]}l'
        else:
            layers = f'{len(ae_config["filters"])+1}l'

        ld = latent_dim(ae_config["latent_shape"])
        seed = f'seed_{self.seed}'
        
        model_path = self.path_builder.ae_path(arch, layers, ld, seed)
        self._ensure_directory(model_path)
        torch.save(model, model_path)
        
        model_name = f"{self.pixel_type}_{ld}_{layers}_{arch}_{seed}"
        return model_path, model_name
    
    def save_classifier(self, model: torch.nn.Module):
        """Save classifier model."""
        model_path = self.path_builder.classifier_path()
        self._ensure_directory(model_path)
        torch.save(model, model_path)
        
        model_name = f"classifier_{self.pixel_type}_{self.dataset}"
        return model_path, model_name
    
    def save_einet(self, model: torch.nn.Module, record = None):
        """Save EinsumNet model and optional training record."""
        result_path, model_name = rb_path(self.models_dir, self.config)
        print(f"Saving EiNet to: {result_path}\n")
        
        # Ensure directory exists
        os.makedirs(result_path, exist_ok=True)
        
        model_file = os.path.join(result_path, 'einet.mdl')
        record_file = os.path.join(result_path, 'record.pkl')
        
        # Save model
        torch.save(model, model_file)
        print(f"Model saved to: {model_file}")
        
        # Save record if provided
        if record is not None:
            import pickle
            with open(record_file, 'wb') as f:
                pickle.dump(record, f)
            print(f"Record saved to: {record_file}")
        
        return model_file, model_name


# Convenience functions for backward compatibility

# Savers
def save_vqvae(models_dir, config, model):
    return ModelSaver(models_dir, config).save_vqvae(model)

def save_vqvae_rec(models_dir, config, model):
    return ModelSaver(models_dir, config).save_vqvae_rec(model)

def save_vqvae_rec(models_dir, config, model):
    return ModelSaver(models_dir, config).save_vqvae_rec(model)


def save_pcnn(models_dir, config, model):
    saver = ModelSaver(models_dir, config)
    return saver.save_pcnn(model)


def save_psnail(models_dir, config, model):
    return ModelSaver(models_dir, config).save_psnail(model)


def save_discrete_latents(models_dir, config, latents, seed = None):
    return ModelSaver(models_dir, config).save_discrete_latents(latents, seed=seed)

def save_continuous_latents(models_dir, config, latents, seed = None):
    return ModelSaver(models_dir, config).save_continuous_latents(latents, seed=seed)


def save_cm(models_dir, config, model):
    return ModelSaver(models_dir, config).save_cm(model)


def save_vae(models_dir, config, model):
    return ModelSaver(models_dir, config).save_vae(model)

def save_ae(models_dir, config, model):
    return ModelSaver(models_dir, config).save_ae(model)

def save_classifier(models_dir, config, model):
    return ModelSaver(models_dir, config).save_classifier(model)


def save_einet(models_dir, config, model, record = None):
    return ModelSaver(models_dir, config).save_einet(model, record)