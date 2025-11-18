from pathlib import Path
import re
import pandas as pd
import sys

def patterns(model):
    if model == "vqvae":
        pattern = (
            r'^dm_con_lat_'
            r'(?P<HEIGHT>\d+)_'
            r'(?P<WIDTH>\d+)_'
            r'(?P<CH>\d+)_'
            r'(?P<VERSION>\w+)_'
            r'cb_size-(?P<K>\d+)_'
            r'(?P<ARCH>[\w\d_-]+)_'
            r'b_(?P<BETA>[0-9.]+)_'
            r'(?P<DT>[\w-]+)_'
            r'(?P<MODEL>[\w-]+)_'
            r'(?P<SEED>\d+)$'
        )
        # pattern = (
        #     r'^dm_con_lat_'
        #     r'(?P<HEIGHT>\d+)_'
        #     r'(?P<WIDTH>\d+)_'
        #     r'(?P<CH>\d+)_'
        #     r'(?P<VERSION>\w+)_'
        #     r'cb_size-(?P<K>\d+)_'
        #     r'(?P<ARCH>[\w\d_-]+)_'
        #     r'b_(?P<BETA>[0-9.]+)_'
        #     r'(?P<DT>[\w-]+)_'
        #     r'(?P<MODEL>[\w-]+)_'
        #     r'seed_(?P<SEED>\d+)$'
        # )
            # r'^dm_con_lat_'
            # r'(?P<HEIGHT>\d+)_'
            # r'(?P<WIDTH>\d+)_'
            # r'(?P<CH>\d+)_'
            # r'(?P<VERSION>[\w-]+)_'
            # r'(?P<LAYERS>[\w\d-]+)_'
            # r'(?P<ARCH>[\w\d_-]+)_'
            # r'b_(?P<BETA>[0-9.]+)_'
            # r'(?P<DT>[\w-]+)_'
            # r'(?P<MODEL>[\w-]+)_'
            # r'seed_(?P<SEED>\d+)$'
    elif model == "vae":
        pattern = (
            r'^dm_con_lat_'
            r'(?P<HEIGHT>\d+)_'
            r'(?P<WIDTH>\d+)_'
            r'(?P<CH>\d+)_'
            r'(?P<VERSION>\w+)_'
            r'(?P<LAYERS>[\w\d]+)_'
            r'(?P<ARCH>[\w_]+)_'
            r'(?P<B>\w+)_'
            r'(?P<BETA>[0-9.]+)_'
            r'(?P<DT>\w+)_'
            r'(?P<MODEL>\w+)_'
            r'(?P<SEED>\d+)$'
        )
    elif model == "ae":
        pattern = (
            r'^dm_con_lat_'
            r'(?P<HEIGHT>\d+)_'
            r'(?P<WIDTH>\d+)_'
            r'(?P<CH>\d+)_'
            r'(?P<VERSION>\w+)_'
            r'(?P<LAYERS>[\w\d]+)_'
            r'(?P<ARCH>[\w_]+)_'
            r'(?P<DT>\w+)_'
            r'(?P<MODEL>\w+)_'
            r'(?P<SEED>\d+)$'
        )
    elif model == "cm":
        pattern = (
            r'^cm_con_cm_4_lat_'
            r'(?P<HEIGHT>\d+)_'
            r'(?P<WIDTH>\d+)_'
            r'(?P<CH>\d+)_'
            r'(?P<VERSION>\w+)_'
            r'nBins(?P<NBINS>\d+)_'
            r'seed_(?P<SEED>\d+)$'
        )
    elif model == "einet":
        pattern = (
            r'^einet_con_(?P<PD_TYPE>pdv|pd)_'
            r'(?P<PD>\d+)_'
            r'(?P<K>\d+)_'
            r'(?P<I>\d+)_'
            r'seed_(?P<SEED>\d+)$'
        )

   
    return pattern

def _parse_settings_from_name(name: str) -> dict:
    print(name)
    settings = {}
    pattern = patterns("vqvae")
    print(pattern)
    m = re.match(pattern, name.strip())
    if m:
        for k, v in m.groupdict().items():
            if k == "HEIGHT":
                settings["DIM"] = f"{v}x{v}"
                continue
            elif k == "WIDTH":
                continue
            elif k == "MODEL" or k == "A" or k == "B":
                continue
            if k == "K" and v == "521":
                print(f"adjusting K from {v} to 512")
                v = 512

            settings[k] = v
        return settings

    # fallback: keep existing parsing for key=value tokens
    tokens = re.split(r'[._\-]+', name)
    for t in tokens:
        if '=' in t:
            k, v = t.split('=', 1)
            settings[k] = v

    return settings



def analyze(dir_path: str = "./grid_search/grid_search_evals11/MNIST/con", out_csv: str = "./grid_search/grid_search_summary11.csv"):
    base = Path(dir_path)
    if not base.exists():
        print(f"directory not found: {base}", file=sys.stderr)
        return

    rows = []
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue

        settings = _parse_settings_from_name(sub.name)


        csv_files = list(sub.glob("test-performance-1.csv"))
        if not csv_files:
            continue
        for csv in csv_files:
            try:
                df = pd.read_csv(csv)
            except Exception as e:
                print(f"skipping {csv}: {e}", file=sys.stderr)
                continue
            # take only the final row (e.g., last epoch / largest n_components)
            last = df.iloc[-1]
            # extract BPD and FID if present
            
            bpd = last.get("BPD", None)
            fid = last.get("FID", None)
            # coerce to float if possible
            try:
                bpd = float(bpd) if bpd is not None and not pd.isna(bpd) else None
            except Exception:
                pass
            try:
                fid = float(fid) if fid is not None and not pd.isna(fid) else None
            except Exception:
                pass
            row = {"BPD": bpd, "FID": fid}
            row.update(settings)
            rows.append(row)

    if not rows:
        print("no csv performance files found", file=sys.stderr)
        return

    summary = pd.DataFrame(rows)


    summary.to_csv(out_csv, index=False)
    # print(f"summary written to {out_csv}")
    # # print concise preview
    # print(summary.loc[:, ["BPD", "FID"]].to_string(index=False))

    # ---- NEW PART: average across seeds ----
    # Define which columns define a unique model configuration (everything except SEED)
    # print(summary.columns)
    grouping_cols = [c for c in summary.columns if c not in {"SEED", "BPD", "FID"}]

    # Aggregate metrics by mean over seeds
    avg_summary = (
        summary.groupby(grouping_cols, dropna=False, as_index=False)
        .agg({"BPD": "mean", "FID": "mean"})
    )
    

    # Save both per-seed results and averaged summary
    summary.to_csv(out_csv.replace(".csv", "_all.csv"), index=False)
    avg_summary.to_csv(out_csv, index=False)

    print(f"summary written to {out_csv} (averaged by seed)")
    print(f"all individual runs written to {out_csv.replace('.csv', '_all.csv')}")
    # concise preview
    print(avg_summary.loc[:, ["BPD", "FID"]].to_string(index=False))

if __name__ == '__main__':
    analyze()