# process_fos.py
# script to take the downsampled voxels
# and returns the matrix of num_mice=168 x num_alive_voxels
# as well as info about the alive voxels
# the 100 um voxels are 89 x 95 x 80
# there are 168 mice
# why do we not have the same number of alive voxels as previously?

import os
import numpy as np
from tqdm import tqdm

NPZ_DATA_DIREC = "/Users/xaviergonzalez/Desktop/xavier_folders/stanford/linderman/serotonin/data/npz_4"
OUT_DIREC = "/Users/xaviergonzalez/Desktop/xavier_folders/stanford/linderman/serotonin/data"

mouse_files = sorted(os.listdir(NPZ_DATA_DIREC))

NUM_MICE = 168
DIMS = (89, 95, 80)

counts = np.zeros((NUM_MICE,) + DIMS, dtype=int)
avg_logs = np.zeros((NUM_MICE,) + DIMS)
idxs = np.zeros(NUM_MICE)

for (i,f) in tqdm(enumerate(mouse_files)):
    idx = int(f[:3])
    idxs[i] = idx
    fname = os.path.join(NPZ_DATA_DIREC, f)
    data = np.load(fname)
    coords = data['coords']
    counts[i, coords[:, 0], coords[:, 1], coords[:, 2]] = data['Ns']
    avg_logs[i, coords[:, 0], coords[:, 1], coords[:, 2]] = data['log_vs']

flat_counts = np.reshape(counts, (NUM_MICE, -1))
flat_avg_logs = np.reshape(avg_logs, (NUM_MICE, -1))

alive_voxels_counts = np.sum(flat_counts, axis=0) > 0 #alive voxels now in 'C' order
alive_voxels_logs = np.sum(flat_avg_logs, axis=0) > 0
assert np.array_equal(alive_voxels_counts, alive_voxels_logs)
alive_voxels = alive_voxels_counts
print(np.sum(alive_voxels_counts)) #annoying that this is not 382533
print(np.sum(alive_voxels_logs))

flat_counts_alive = flat_counts[:, alive_voxels]
flat_logs_alive = flat_avg_logs[:, alive_voxels]

np.save(os.path.join(OUT_DIREC, "dims.npy"), DIMS)
np.save(os.path.join(OUT_DIREC, "counts.npy"), flat_counts_alive)
np.save(os.path.join(OUT_DIREC, "avg_logs.npy"), flat_logs_alive)
np.save(os.path.join(OUT_DIREC, "alive_voxels.npy"), alive_voxels)
np.save(os.path.join(OUT_DIREC, "mouse_ids.npy"), idxs)