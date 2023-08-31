# compress_fos_csvs.py
# this code takes in the 190 .cat files
# and returns 190 .npz files
# each .npz files has:
# * a list of coordinates
# * a list of average logs
# * a list of counts (for that voxel)


import os
import numpy as np
import pandas as pd

from tqdm.auto import trange

# RAW_DATA_DIREC = "/Volumes/Storage/serotonin/csv"
# NPZ_DATA_DIREC = "/Volumes/Storage/serotonin/npz"

RAW_DATA_DIREC = "/Users/xaviergonzalez/Desktop/xavier_folders/stanford/linderman/serotonin/data/cfos"
NPZ_DATA_DIREC = "/Users/xaviergonzalez/Desktop/xavier_folders/stanford/linderman/serotonin/data/npz"

for mouse in trange(1, 191):
    fname = os.path.join(RAW_DATA_DIREC, "{:03d}_cat_pts.csv".format(mouse))
    if not os.path.exists(fname):
        continue

    # Load the csv
    data = pd.read_csv(
        os.path.join(RAW_DATA_DIREC, "{:03d}_cat_pts.csv".format(mouse)),
        names=["x", "y", "z", "v", "allen"],
        header=None, sep=" ")

    # We want to work with the log of the Fos intensities.
    data["log_v"] = np.log(data["v"])
    # Add a dummy `count` column to aggregate cell counts
    data["count"] = 1

    # Aggregate the data by voxel
    aggdata = pd.pivot_table(data, values=["log_v", "count"],
                            index=["x", "y", "z"],
                            aggfunc="sum")

    # Get the number of cells per 25um voxel
    Ns = np.array(aggdata["count"], dtype=np.int16)

    # Compute the mean intensity
    aggdata["log_v_bar"] = aggdata["log_v"] / aggdata["count"]
    log_vs = np.array(aggdata["log_v_bar"], dtype=np.float32)

    # Plot the distribution of average log fos intensity in this mouse
    # plt.hist(aggdata["log_v_bar"], 100, alpha=0.5, density=True)
    # plt.xlabel(r"$\log v$")
    # plt.ylabel("empirical density")
    # plt.title("mouse {} average log fos intensity distribution".format(mouse))

    # Get the sparse array representation in smaller dtypes
    xs = np.array(aggdata.index.get_level_values(0), dtype=np.int16)
    ys = np.array(aggdata.index.get_level_values(1), dtype=np.int16)
    zs = np.array(aggdata.index.get_level_values(2), dtype=np.int16)
    coords = np.column_stack([xs, ys, zs])

    # Save compressed
    np.savez(os.path.join(NPZ_DATA_DIREC, "{:03d}_data.npz".format(mouse)),
             coords=coords, log_vs=log_vs, Ns=Ns)
