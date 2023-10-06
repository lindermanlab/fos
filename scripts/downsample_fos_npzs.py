# downsample_fos_npzs.py
# this downsamples the fos_npz's by a factor
import os
import numpy as np
import pandas as pd
import argparse

from tqdm.auto import trange

NPZ_DATA_DIREC = "/Users/xaviergonzalez/Desktop/xavier_folders/stanford/linderman/serotonin/data/npz"

parser = argparse.ArgumentParser()
parser.add_argument('--downsample', type=int, default=4, help='factor by which we downsample')
parser.add_argument('--data_direc', type=str, default=NPZ_DATA_DIREC, help='direc where data is stored')

args = parser.parse_args()

if __name__ == '__main__':
    NPZ_DATA_DIREC = args.data_direc
    OUT_DIREC = args.data_direc + f"_{args.downsample}"
    if not os.path.exists(OUT_DIREC):
        os.makedirs(OUT_DIREC)
        print(f"Directory {OUT_DIREC} created.")
    else:
        print(f"Directory {OUT_DIREC} already exists.")

    for mouse in trange(1,191):
        fname = os.path.join(NPZ_DATA_DIREC, f"{mouse:03d}_data.npz")
        if not os.path.exists(fname):
            continue

        # load the npz
        data = np.load(fname)

        sum_of_logs = np.sum(data['log_vs'] * data['Ns'])
        # print(f"sum of the logs is {sum_of_logs}")

        # downsample
        coords = data['coords'] // args.downsample

        df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'z': coords[:, 2],
            'log_v': data['log_vs'] * data['Ns'],
            'Ns': data['Ns']
        })

        # Aggregate the data by voxel
        aggdata = pd.pivot_table(df, values=["log_v", "Ns"],
                                index=["x", "y", "z"],
                                aggfunc="sum")

        # Get the number of cells per 100um voxel
        Ns = np.array(aggdata["Ns"], dtype=np.int16)

        # Compute the mean intensity
        aggdata["log_v_bar"] = aggdata["log_v"] / aggdata["Ns"]
        log_vs = np.array(aggdata["log_v_bar"], dtype=np.float32)

        # Get the sparse array representation in smaller dtypes
        xs = np.array(aggdata.index.get_level_values(0), dtype=np.int16)
        ys = np.array(aggdata.index.get_level_values(1), dtype=np.int16)
        zs = np.array(aggdata.index.get_level_values(2), dtype=np.int16)
        coords = np.column_stack([xs, ys, zs])

        # print(f"the sum of the logs in the output is {np.sum(log_vs * Ns)}")

        # print(f"the shape of coords is {coords.shape}")
        # print(f"the shape of log_vs is {log_vs.shape}")
        # print(f"the shape of Ns is {Ns.shape}")

        # Save compressed
        np.savez(os.path.join(OUT_DIREC, f"{mouse:03d}_data_{args.downsample}.npz"),
                coords=coords, log_vs=log_vs, Ns=Ns)

        