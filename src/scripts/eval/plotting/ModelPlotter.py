import numpy as np

# Central registry for all models - each model has consistent visual identity
MODEL_REGISTRY = {

    "DM": {
        'marker': 'o', 
        'color': "#030305", 
        'label': 'DM',
        'linestyle': ':'
    },

    "DM best fid": {
        'marker': 'o', 
        'color': "#0202BD", 
        'label': 'Lowest FID 8x8x256 - beta=1.0 ML',
        'linestyle': '-'
    },

    "DM best bpd": {
        'marker': 'o', 
        'color': "#FFE600", 
        'label': 'Lowest BPD 1x1x256 - beta=50.0',
        'linestyle': '-' 
    },

    "DM best lsm": {
        'marker': 'o', 
        'color': "#1D9E0C", 
        'label': 'Lowest Sum of Ranks 1x1x1024 - beta=50.0',
        'linestyle': ':' 
    },

    "DM best ns": {
        'marker': 'o', 
        'color': "#740FC7", 
        'label': 'Lowest Norm. Score 8x8x256 - beta=50.0',
        'linestyle': ':'
    },



    "dm-RS (old loss)": {
        'marker': 'o', 
        'color': '#000080', 
        'label': 'DM-RS (old loss) 8x8x128 - 256',
        'linestyle': ':'
    },
    "dm-RS (new loss)": {
        'marker': 'o', 
        'color': "#0059FF", 
        'label': 'DM-RS (new loss) 8x8x128 - 256',
        'linestyle': '-'
    },
    "dm-ML (old loss)": {
        'marker': 'o', 
        'color': "#356504", 
        'label': 'DM-ML (old loss) 8x8x128 - 256',
        'linestyle': ':'
    },

    "dm-ML (old loss) big": {
        'marker': 'o', 
        'color': "#000000", 
        'label': 'DM-ML (old loss) 8x8x256 - 512',
        'linestyle': ':'
    },

    "dm-ML (new loss)": {
        'marker': 'o', 
        'color': "#18D3BA", 
        'label': 'DM-ML (new loss)',
        'linestyle': '-'
    },

    "dm-ML (old loss) w. prior": {
        'marker': 'o', 
        'color': "#C0D318", 
        'label': 'DM-ML (old loss) w. prior 8x8x128 - 256',
        'linestyle': '-'
    },

    "cm": {
        'marker': '^', 
        'color': "#C00B0B", 
        'label': 'CM 1x1x16',
        'linestyle': '-'
    },

    "einets": {
        'marker': 's', 
        'color': "#000000", 
        'label': 'EiNet',
        'linestyle': '--'
    }
}

def confidence_intervals(mean_data, variance_data, ci_multiplier=1.0):
    """
    Helper method to calculate confidence intervals from mean and variance
    
    Parameters:
    -----------
    mean_data : array-like
        Mean values
    variance_data : array-like  
        Variance values
    ci_multiplier : float, default=1.0
        Multiplier for confidence interval (e.g., 1.96 for 95% CI)
        
    Returns:
    --------
    tuple : (lower_bounds, upper_bounds)
        Lower and upper confidence interval bounds
    """
    std_data = np.sqrt(variance_data)
    ci_width = ci_multiplier * std_data
    lower_bounds = mean_data - ci_width
    upper_bounds = mean_data + ci_width
    return lower_bounds, upper_bounds

class ModelPlotter:
    def __init__(self, model_registry=None):
        self.registry = model_registry or MODEL_REGISTRY
    
    def get_model_style(self, model_id):
        """Get consistent styling for a model"""
        if model_id not in self.registry:
            raise ValueError(f"Model {model_id} not found in registry")
        return self.registry[model_id]
    
    def line_plot(self, ax, data_configs, **plot_kwargs):
        """
        Plot lines for multiple models
        data_configs: list of tuples (model_id, x_data, y_data, additional_kwargs)
        """
        default_kwargs = {'alpha': 0.8, 'linewidth': 2}
        default_kwargs.update(plot_kwargs)
        print(len(data_configs))

        for config in data_configs:
            if len(config) == 3:
                model_id, x_data, y_data = config
                extra_kwargs = {}
            else:
                model_id, x_data, y_data, extra_kwargs = config
            
            style = self.get_model_style(model_id)
            plot_kwargs_final = {**default_kwargs, **extra_kwargs}
            
            ax.plot(x_data, y_data, 
                   marker=style['marker'],
                   color=style['color'],
                   label=style['label'],
                   linestyle=style['linestyle'],
                   **plot_kwargs_final)
    
    def fill_between_plot(self, ax, data_configs, ci_multiplier=1.0, ci_type="std", **fill_kwargs):
        """
        Fill between areas for confidence intervals
        
        Parameters:
        -----------
        ax : matplotlib axis
            The axis to plot on
        data_configs : list of tuples
            Each tuple: (model_id, x_data, lower_data, upper_data, additional_kwargs)
        ci_multiplier : float, default=1.0
            Multiplier for confidence interval calculation
            - 1.0 = ±1 standard deviation (~68% for normal distribution)
            - 1.96 = ±1.96 standard deviations (~95% for normal distribution) 
            - 2.0 = ±2 standard deviations (~95.4% for normal distribution)
        ci_type : str, default="standard deviation"
            Description of what the confidence interval represents
        fill_kwargs : dict
            Additional arguments passed to fill_between
        """
        default_kwargs = {'alpha': 0.3}
        default_kwargs.update(fill_kwargs)
        
        # Store CI info for potential legend/title use
        self.last_ci_info = {
            'multiplier': ci_multiplier,
            'type': ci_type,
            'description': f"±{ci_multiplier} {ci_type}"
        }
        
        for config in data_configs:
            if len(config) == 4:
                model_id, x_data, lower_data, upper_data = config
                extra_kwargs = {}
            else:
                model_id, x_data, lower_data, upper_data, extra_kwargs = config
            
            style = self.get_model_style(model_id)
            fill_kwargs_final = {**default_kwargs, **extra_kwargs}
            
            ax.fill_between(x_data, lower_data, upper_data,
                           color=style['color'],
                           **fill_kwargs_final)
    
    def get_ci_description(self):
        """Get description of the last confidence interval used"""
        if hasattr(self, 'last_ci_info'):
            return self.last_ci_info['description']
        return "Confidence interval"
    
    def scatter_plot(self, ax, data_configs, **scatter_kwargs):
        """
        Scatter plot for multiple models
        data_configs: list of tuples (model_id, x_data, y_data, additional_kwargs)
        """
        default_kwargs = {'alpha': 0.7, 's': 50}
        default_kwargs.update(scatter_kwargs)
        
        for config in data_configs:
            if len(config) == 3:
                model_id, x_data, y_data = config
                extra_kwargs = {}
            else:
                model_id, x_data, y_data, extra_kwargs = config
            
            style = self.get_model_style(model_id)
            scatter_kwargs_final = {**default_kwargs, **extra_kwargs}
            
            ax.scatter(x_data, y_data,
                      marker=style['marker'],
                      color=style['color'],
                      label=style['label'],
                      **scatter_kwargs_final)
    
    def bar_plot(self, ax, data_configs, **bar_kwargs):
        """
        Bar plot for multiple models
        data_configs: list of tuples (model_id, x_data, y_data, additional_kwargs)
        """
        default_kwargs = {'alpha': 0.8}
        default_kwargs.update(bar_kwargs)
        
        for config in data_configs:
            if len(config) == 3:
                model_id, x_data, y_data = config
                extra_kwargs = {}
            else:
                model_id, x_data, y_data, extra_kwargs = config
            
            style = self.get_model_style(model_id)
            bar_kwargs_final = {**default_kwargs, **extra_kwargs}
            
            ax.bar(x_data, y_data,
                   color=style['color'],
                   label=style['label'],
                   **bar_kwargs_final)