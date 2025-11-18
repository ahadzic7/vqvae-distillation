import pandas as pd
import matplotlib.pyplot as plt
from scripts.eval.plotting.AutoPlotConfig import AutoPlotConfig
import os

def data_frames_SVHN(eval_dir, dataset):
    dir = f"{eval_dir}/{dataset}/con"

    df1 = pd.read_csv(f"{dir}/dm_con_8_8_256_512_4l_overlap_vqvae_ML_old/performance-1.csv")
    n1 = "dm-ML (old loss) big"
    
    df2 = pd.read_csv(f"{dir}/cm_con_1_1_16_4_16384/performance-5.csv")
    n2 = "cm"
    
    # Example: df3 has FID but no FID_var - will still plot line, just no fill
    # df3 = pd.read_csv(f"/home/hadziarm/vqvae-tpm/einets_celeba_128_con.csv")
    # datasets.append(auto_config.add_dataframe('einets', df3))
    
    
    df4 = pd.read_csv(f"{dir}/dm_con_8_8_256_512_4l_overlap_pcnn_RS/test-performance-1.csv")
    n4 = "dm-RS (new loss)"

    df7 = pd.read_csv(f"{dir}/dm_con_lat_8_8_128_old_256_4l_overlap_pcnn_ML/test-performance-1.csv")
    n7 = "dm-ML (old loss) w. prior"

    df8 = pd.read_csv(f"{dir}/dm_con_lat_8_8_128_old_256_4l_overlap_pcnn_RS/test-performance-1.csv")
    n8 = "dm-RS (old loss)"

    df9 = pd.read_csv(f"{dir}/dm_con_lat_8_8_128_old_256_4l_overlap_vqvae_ML/test-performance-1.csv")
    n9 = "dm-ML (old loss)"

    return [ 
        
        # (n4, df4),
        (n8, df8),

        # (n1, df1),
        # (n7, df7),
        (n9, df9),

        


        (n2, df2),
    ]

def data_frames_celeba(eval_dir, dataset):
    dir = f"{eval_dir}/{dataset}"

    df1 = pd.read_csv(f"{dir}/con/dm_con_32_32_256_512_4l_overlap_pixel_cnn_og/performance-5.csv")
    n1 = "dm-RS (old loss)"

    df2 = pd.read_csv(f"{dir}/con/cm_con_1_1_32_4_16384/performance-5.csv")
    n2 = "cm"
    
    df3 = pd.read_csv(f"{dir}/con/einets_celeba_128_con.csv")
    n3 = "einets"

    df4 = pd.read_csv(f"{dir}/con/dm_con_32_32_256_512_4l_overlap_pixel_cnn/performance-5.csv")
    n4 = "dm-RS"

    return [(n1, df1), (n2, df2), (n3, df3), (n4, df4)]

def load_csv_data(eval_dir, dataset):
    dir = f"/home/hadziarm/vqvae-tpm/evaluations/{dataset}/con"
    df2 = pd.read_csv(f"{dir}/cm_con_1_1_16_4_16384/performance-5.csv")
    n2 = "cm"

    dir_path = os.path.join(eval_dir, dataset, "con")
    dataframes = [(n2, df2)]
    friendly_name = f"DM"
    file = "test-performance-1.csv"
    for root, folders, _ in os.walk(dir_path):
        for folder in folders:
            file_path = os.path.join(root, folder, file)
            df = pd.read_csv(file_path)
            dataframes.append((friendly_name, df))
    
    return dataframes

def load_csv_data(eval_dir, dataset):
    dir = f"/home/hadziarm/vqvae-tpm/evaluations/{dataset}/con"
    df2 = pd.read_csv(f"{dir}/cm_con_1_1_16_4_16384/performance-5.csv")
    n2 = "cm"

    dir = f"{eval_dir}/{dataset}/con"
    
    best_fid = pd.read_csv(f"{dir}/dm_con_lat_16_16_256_new_1024_4l_overlap_vqvae_ML/test-performance-1.csv")
    best_fid_n = "DM best fid"

    best_bpd = pd.read_csv(f"{dir}/dm_con_lat_4_4_256_new_64_4l_overlap_vqvae_ML/test-performance-1.csv")
    best_bpd_n = "DM best bpd"

    best_lsm = pd.read_csv(f"{dir}/dm_con_lat_16_16_32_new_4096_4l_overlap_vqvae_ML/test-performance-1.csv")
    best_lsm_n = "DM best lsm"

    best_ns = pd.read_csv(f"{dir}/dm_con_lat_16_16_32_new_512_4l_overlap_vqvae_ML/test-performance-1.csv")
    best_ns_n = "DM best ns"

    dataframes = [ 
        (n2, df2),
        (best_fid_n, best_fid),
        (best_bpd_n, best_bpd),
        (best_lsm_n, best_lsm),
        (best_ns_n, best_ns),
    ]
    
    return dataframes

# def load_csv_data_vae(eval_dir, dataset):
#     dir = f"/home/hadziarm/vqvae-tpm/grid_search_evals3/{dataset}/con"
#     df2 = pd.read_csv(f"{dir}/cm_con_1_1_16_4_16384/performance-5.csv")
#     n2 = "cm"

#     dir = f"{eval_dir}/{dataset}/con"
    
#     for s in range(5):
#         best_fid_seed = pd.read_csv(f"{dir}/dm_con_lat_8_8_256_old_4l_overlap_b_1.0_ML_vae_{s}/test-performance-1.csv")
#     best_fid_n = "DM best fid"

#     best_bpd = pd.read_csv(f"{dir}/dm_con_lat_4_4_256_new_64_4l_overlap_vqvae_ML/test-performance-1.csv")
#     best_bpd_n = "DM best bpd"

#     best_lsm = pd.read_csv(f"{dir}/dm_con_lat_16_16_32_new_4096_4l_overlap_vqvae_ML/test-performance-1.csv")
#     best_lsm_n = "DM best lsm"

#     best_ns = pd.read_csv(f"{dir}/dm_con_lat_16_16_32_new_512_4l_overlap_vqvae_ML/test-performance-1.csv")
#     best_ns_n = "DM best ns"

#     dataframes = [ 
#         (n2, df2),
#         (best_fid_n, best_fid),
#         (best_bpd_n, best_bpd),
#         (best_lsm_n, best_lsm),
#         (best_ns_n, best_ns),
#     ]
    
#     return dataframes

def load_csv(path, name, seed=None):
    df = pd.read_csv(path)
    if seed is not None:
        df['seed'] = seed
    return (name, df)

def process(fid_frames):
    fid_df = pd.concat(fid_frames, ignore_index=True)

    # Group by n_components and compute mean and variance
    grouped = fid_df.groupby("n_components")

    # Compute mean and variance for each metric
    fid_avg_df = pd.DataFrame({
        "n_components": grouped["n_components"].first(),
        "n_par": grouped["n_par"].mean(),
        "BPD": grouped["BPD"].mean(),
        "BPD_var": grouped["BPD"].var(ddof=0),
        "FID": grouped["FID"].mean(),
        "FID_var": grouped["FID"].var(ddof=0)
    }).reset_index(drop=True)
    return fid_avg_df

def load_csv_data_vae(eval_dir, dataset):
    base_dir = f"{eval_dir}/{dataset}/con"

    # Load CM baseline
    dir = f"/home/hadziarm/vqvae-tpm/evaluations/{dataset}/con"
    df2 = pd.read_csv(f"{dir}/cm_con_1_1_16_4_16384/test-performance-5.csv")
    n2 = "cm"
    dataframes = [(n2, df2)]

    # Load DM best fid across seeds
    fid_frames = [
        pd.read_csv(f"{base_dir}/dm_con_lat_8_8_256_old_4l_overlap_b_1.0_ML_vae_{s}/test-performance-1.csv").assign(seed=s)
        for s in range(5)
    ]
    dataframes.append(("DM best fid", process(fid_frames)))

    
    fid_frames = [
        pd.read_csv(f"{base_dir}/dm_con_lat_1_1_256_old_4l_overlap_b_50.0_ML_vae_{s}/test-performance-1.csv").assign(seed=s)
        for s in range(5)
    ]
    dataframes.append(("DM best bpd", process(fid_frames)))

    fid_frames = [
        pd.read_csv(f"{base_dir}/dm_con_lat_1_1_1024_old_4l_overlap_b_50.0_ML_vae_{s}/test-performance-1.csv").assign(seed=s)
        for s in range(5)
    ]
    dataframes.append(("DM best lsm", process(fid_frames)))


    fid_frames = [
        pd.read_csv(f"{base_dir}/dm_con_lat_8_8_256_old_4l_overlap_b_50.0_ML_vae_{s}/test-performance-1.csv").assign(seed=s)
        for s in range(5)
    ]
    dataframes.append(("DM best ns", process(fid_frames)))

    return dataframes

DATA_FRAMES = {
    # "SVHN": data_frames_SVHN,
    "celeba": data_frames_celeba,
    # "SVHN": load_csv_data,
    "SVHN": load_csv_data_vae,
}

def plot():
    auto_config = AutoPlotConfig(ci_multiplier=1)
    # eval_dir = "/home/hadziarm/vqvae-tpm/evaluations"
    eval_dir = "/home/hadziarm/vqvae-tpm/grid_search_evals3"
    data = "SVHN"

    dfs = DATA_FRAMES[data](eval_dir, data)
    datasets = [auto_config.add_dataframe(name, df) for name, df in dfs]

    available_metrics = []
    for dataset in datasets:
        if dataset is not None:
            available_metrics = list(dataset['metrics'].keys())
            break

    metrics_to_plot = ['BPD', 'FID']
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
        ax.set_title(f'{data} - {metric} vs #Parameters ({ci_description})')
    
    axes[-1].set_xlabel('#Parameters')
    
    plt.tight_layout()
    plt.savefig("test7.png")


if __name__ == "__main__":
    plot()