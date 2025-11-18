from src.scripts.eval.plotting.ModelPlotter import ModelPlotter, confidence_intervals

class AutoPlotConfig:
    """Helper class to automatically generate plot configurations from dataframes."""
    
    def __init__(self, ci_multiplier=1):
        self.ci_multiplier = ci_multiplier
        self.plotter = ModelPlotter()
    
    def add_dataframe(self, name, df, x_col='n_par', sort_by=None):
        """
        Add a dataframe to the configuration.
        
        Args:
            name: Identifier for this dataset
            df: DataFrame containing the data
            x_col: Column to use for x-axis (default: 'n_par')
            sort_by: Column to sort by (default: None, uses x_col if needed)
        
        Returns:
            Dictionary containing line and fill configs for all available metrics
        """
        if df is None or df.empty:
            return None
        
        # Sort if needed
        if sort_by is not None:
            df = df.sort_values(sort_by)
        
        config = {
            'name': name,
            'x': df[x_col].values,
            'metrics': {}
        }
        
        # Auto-detect available metrics (columns with corresponding variance columns)
        for col in df.columns:
            if col.endswith('_var'):
                continue
            
            # Check if this metric has data
            if col in df.columns and not df[col].isna().all():
                metric_config = {
                    'y': df[col].values,
                    'has_variance': False
                }
                
                # Check for variance column
                var_col = f"{col}_var"
                if var_col in df.columns and not df[var_col].isna().all():
                    bounds = confidence_intervals(
                        df[col].values, 
                        df[var_col].values, 
                        self.ci_multiplier
                    )
                    metric_config['has_variance'] = True
                    metric_config['bounds'] = bounds
                
                config['metrics'][col] = metric_config
        
        return config
    
    def build_configs(self, dataset_configs, metric):
        """
        Build line and fill configs for a specific metric.
        
        Args:
            dataset_configs: List of configs from add_dataframe
            metric: Metric name (e.g., 'BPD', 'FID')
        
        Returns:
            Tuple of (line_configs, fill_configs)
        """
        line_configs = []
        fill_configs = []

        for config in dataset_configs:
            if config is None:
                continue
            
            if metric not in config['metrics']:
                continue
            
            metric_data = config['metrics'][metric]
            
            # Add line config
            line_configs.append((
                config['name'],
                config['x'],
                metric_data['y']
            ))
            
            # Add fill config if variance exists
            if metric_data['has_variance']:
                fill_configs.append((
                    config['name'],
                    config['x'],
                    *metric_data['bounds']
                ))
        
        return line_configs, fill_configs

