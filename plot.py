import pandas as pd
import matplotlib.pyplot as plt
from src.scripts.eval.plotting.AutoPlotConfig import AutoPlotConfig
import os

def load_csv_data(eval_dir, dataset):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir = f"{current_dir}/results/{dataset}/con"
    cm_df = pd.read_csv(f"{dir}/cm_con_cm_4_lat_1_1_16_old_nBins1_seed_0/test-performance-1.csv")
    cm_n = "CM"
    
    dmrs_df = pd.read_csv(f"{dir}/dm_con_lat_4_4_512_old_cb_size-4096_3l_overlap_b_10.0_RS_pcnn_seed_0/test-performance-1.csv")
    dmrs_n = "DMRS 4x4"

    dmrs_df2 = pd.read_csv(f"{dir}/dm_con_lat_4_4_512_old_cb_size-4096_3l_overlap_b_10.0_RS_pcnn_seed_0/test-performance-1.csv")
    dmrs_n2 = "DMRS 2x2"

    dmbs_df = pd.read_csv(f"{dir}/dm_con_lat_4_4_512_old_cb_size-4096_3l_overlap_b_10.0_BS_pcnn_seed_0/test-performance-1.csv")
    dmbs_n = "DMBS 4x4"

    einet_df = pd.read_csv(f"{dir}/dm_con_lat_4_4_512_old_cb_size-4096_3l_overlap_b_10.0_RS_pcnn_seed_0/test-performance-1.csv")
    einet_n = "EiNet"

    dataframes = [ 
        (cm_n, cm_df),
        (dmrs_n, dmrs_df),
        (dmrs_n2, dmrs_df2),
        (dmbs_n, dmbs_df),
        (einet_n, einet_df),
    ]
    
    return dataframes


DATA_FRAMES = {
    "MNIST": load_csv_data,
}

def plot():
    auto_config = AutoPlotConfig(ci_multiplier=1)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    eval_dir = f"{current_dir}/grid_search_evals3"
    data = "MNIST"

    dfs = DATA_FRAMES[data](eval_dir, data)
    datasets = [auto_config.add_dataframe(name, df) for name, df in dfs]

    available_metrics = []
    for dataset in datasets:
        if dataset is not None:
            available_metrics = list(dataset['metrics'].keys())
            break

    metrics_to_plot = ['BPD']
    metrics_to_plot = [m for m in metrics_to_plot if m in available_metrics]
    
    if not metrics_to_plot:
        print("No metrics to plot!")
        return
    
    fig, axes = plt.subplots(len(metrics_to_plot), 1, 
                             figsize=(10, 4*len(metrics_to_plot)), 
                             sharex=True)
    
    if len(metrics_to_plot) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics_to_plot):
        line_configs, fill_configs = auto_config.build_configs(datasets, metric)

        if not line_configs:
            continue
        auto_config.plotter.line_plot(ax, line_configs)
        if fill_configs:
            auto_config.plotter.fill_between_plot(
                ax, fill_configs,
                ci_multiplier=auto_config.ci_multiplier,
                ci_type="std"
            )
        
        ci_description = auto_config.plotter.get_ci_description() if fill_configs else "No CI"
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('symlog')
        #ax.set_yscale('symlog')
        ax.set_title(f'{data} - {metric} vs #Parameters')
    
    axes[-1].set_xlabel('#Parameters')
    
    plt.tight_layout()
    plt.savefig(f'{data}-{metric}_vs_#parameters.png')


if __name__ == "__main__":
    plot()