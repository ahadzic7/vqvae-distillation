from dataclasses import dataclass

@dataclass
class PathBuilder:
    """Helper class to build model paths consistently."""
    
    models_dir:str
    dataset:str
    pixel_type:str
    
    def base_path(self):
        """Returns the base path for all models."""
        return f'{self.models_dir}/{self.dataset}/{self.pixel_type}'
    
    def vqvae_name(self, arch, layers, ld, cb, beta, seed):
        return f"{self.pixel_type}_{ld}_{cb}_{layers}_{arch}_{beta}_{seed}"
    
    def vqvae_path(self, arch, layers, ld, cb, beta, seed):
        return f'{self.base_path()}/vqvae/{arch}/{layers}/{ld}/{cb}/{beta}/{seed}/checkpoints/model.ckpt'
    

    def pcnn_name(self, model, arch, layers, ld, cb, beta, seed):
        return f"{self.pixel_type}_{model}_{ld}_{cb}_{layers}_{arch}_{beta}_{seed}"

    def pcnn_path(self, model, arch, layers, ld, cb, beta, seed):
        return f'{self.base_path()}/{model}/{arch}/{layers}/{ld}/{cb}/{beta}/{seed}/checkpoints/model.ckpt'
    

    def psnail_name(self, model, arch, layers, ld, cb):
        return f"{self.pixel_type}_{model}_{ld}_{cb}_{layers}_{arch}"

    def psnail_path(self, model, arch, layers, ld, cb):
        return f'{self.base_path()}/{model}/{arch}/{layers}/{ld}/{cb}/checkpoints/model.ckpt'
    


    def discrete_dm_name(self, arch, layers, ld, cb, beta, seed, dt, prior):
        return f"{self.pixel_type}_{ld}_{cb}_{layers}_{arch}_{beta}_{dt}_{prior}_{seed}"

    def discrete_dm_path(self, arch, layers, ld, cb, beta, seed, dt, prior):
        return f'{self.base_path()}/dm/{arch}/{layers}/{ld}/{cb}/{beta}/{seed}/{dt}/{prior}'


    def continuous_dm_name(self, arch, layers, ld, beta, seed, dt, prior):
        return f"{self.pixel_type}_{ld}_{layers}_{arch}_{beta}_{dt}_{prior}_{seed}"
    
    def continuous_dm_path(self, arch, layers, ld, beta, seed, dt, prior):
        return f'{self.base_path()}/dm/{arch}/{layers}/{ld}/{beta}/{seed}/{dt}/{prior}'
    


    def continuous_dm_name_ae(self, arch, layers, ld, seed, dt, prior):
        return f"{self.pixel_type}_{ld}_{layers}_{arch}_{dt}_{prior}_{seed}"
    
    def continuous_dm_path_ae(self, arch, layers, ld, seed, dt, prior):
        return f'{self.base_path()}/dm/{arch}/{layers}/{ld}/{seed}/{dt}/{prior}'
    


    def cm_name(self, nl, ld, nb, seed):
        return f"{self.pixel_type}_cm_{nl}_{ld}_nBins{nb}_seed_{seed}"

    def cm_path(self, nl, ld, nb, seed):
        return f'{self.base_path()}/cm/layers_{nl}/{ld}/n_bins-{nb}/seed_{seed}/checkpoints/model.ckpt'
    

    def vae_path(self, arch, layers, ld, beta, seed):
        return f'{self.base_path()}/vae/{arch}/{layers}/{ld}/{beta}/{seed}/checkpoints/model.ckpt'
    
    def ae_path(self, arch, layers, ld, seed):
        return f'{self.base_path()}/ae/{arch}/{layers}/{ld}/{seed}/checkpoints/model.ckpt'
    
    def vae_name(self, arch, layers, ld, beta, seed):
        return f"{self.pixel_type}_{ld}_{layers}_{arch}_{beta}_{seed}"


    def classifier_path(self):
        return f'{self.base_path()}/classifier/{self.dataset}/0/checkpoints/model.ckpt'
    
    @staticmethod
    def split_training_path(full_path):
        """
        Split a full checkpoint path into PyTorch Lightning trainer components.
        
        Given: /path/to/dataset/pixel/model/arch/layers/ld/name/version/checkpoints/model.ckpt
        Returns: (log_folder, name, version)
        Where log_folder = /path/to/dataset/pixel/model/arch/layers/ld
        
        Args:
            full_path: Complete checkpoint path
            
        Returns:
            Tuple of (log_folder, name, version)
        """
        # Remove /checkpoints/model.ckpt from the end
        path_without_checkpoint = full_path.rsplit('/checkpoints/', 1)[0]
        
        # Split into parts
        parts = path_without_checkpoint.split('/')
        
        # Last part is version, second to last is name
        version = parts[-1]
        name = parts[-2]
        log_folder = '/'.join(parts[:-2])
        
        return log_folder, name, version


    def vqvae_rec_name(self, arch, layers, ld, cb, beta, seed):
        return f"{self.pixel_type}_{ld}_{cb}_{layers}_{arch}_{beta}_{seed}"
    
    def vqvae_rec_path(self, arch, layers, ld, cb, beta, seed):
        return f'{self.base_path()}/vqvae_rec/{arch}/{layers}/{ld}/{cb}/{beta}/{seed}/checkpoints/model.ckpt'
