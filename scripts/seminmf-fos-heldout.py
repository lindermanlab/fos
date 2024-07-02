import click
import numpy as np
import os
import warnings
import pickle

import jax.numpy as jnp
import jax.random as jr
import wandb
from fastprogress import progress_bar

from fos import seminmf_full as seminmf

warnings.filterwarnings("ignore")

def load_data(data_dir, left_trunc):
    """Load the combined data
     
    Note: files are permanently available on [Google Drive](https://drive.google.com/drive/u/0/folders/1xD54Uq4cKJsACmy8KVZZZ5YGCjF6ZrkC). 
    The scratch storage may be deleted periodically by Sherlock.
    """
    data = np.load(os.path.join(data_dir, "downsampled_data_4.npz"))
    intensity_3d = data["intensity"][:-1]
    counts_3d = data["counts"][:-1]
    num_mice = counts_3d.shape[0]
    assert num_mice == 168
    assert intensity_3d.shape[0] == 168

    counts_3d = counts_3d[:, left_trunc:, :, :]
    intensity_3d = intensity_3d[:, left_trunc:, :, :]

    alive_voxels = jnp.sum(counts_3d, axis=0) > 0
    print(alive_voxels.sum(), "/", np.prod(intensity_3d.shape[1:]), "voxels are 'alive'")
    
    intensity = intensity_3d[:, alive_voxels]
    counts = counts_3d[:, alive_voxels]
    intensity[counts == 0] = 0.0

    # Load the drug ids ((M,) array of ints)
    drugs = np.load(os.path.join(data_dir, "drug_ids.npy"))

    return intensity, counts, alive_voxels, drugs


@click.command()
@click.option('--data_dir', default="/home/groups/swl1/swl1/serotonin/", help='path to data file (at 100um resolution).')
@click.option('--results_dir', default="/home/groups/swl1/swl1/fos/results/2024_07_01-12_00-heldout", help='path to folder where results are stored.')
@click.option('--left_trunc', default=8, help='number of voxels to drop on the left side of the brain')
@click.option('--num_factors', default=14, help='number of factors.')
@click.option('--sparsity_penalty', default=0.01, help='sparsity penalty.')
@click.option('--mean_func', default="softplus", help='mean (inverse link) function.')
@click.option('--elastic_net_frac', default=1.0, help='fraction of L1 vs L2 penalty.')
@click.option('--num_iters', default=500, help='max number of iterations of EM')
@click.option('--num_coord_ascent_iters', default=1, help='number of inner iterations of coordinate ascent')
@click.option('--test_frac', default=0.25, help='fraction of mice to withhold for testing')
@click.option('--seed', default=0, help='random seed')
@click.option('--wandb_project', default="serotonin-fos-seminmf-heldout", help='wandb project name')
def run_sweep(data_dir, 
              results_dir, 
              left_trunc, 
              num_factors, 
              sparsity_penalty, 
              mean_func, 
              elastic_net_frac, 
              num_iters, 
              num_coord_ascent_iters, 
              test_frac,
              seed,
              wandb_project):
    
    # Load the data
    intensity, counts, alive_voxels, drugs = load_data(data_dir, left_trunc)
    num_mice, num_alive_voxels = counts.shape

    # Hold out a fraction of mice for testing
    key = jr.PRNGKey(seed)
    test_inds = jr.choice(key, num_mice, replace=False, shape=(int(test_frac * num_mice),))
    mask = jnp.zeros(num_mice, dtype=bool)
    mask = mask.at[test_inds].set(True)

    result_file = os.path.join(results_dir, f"params_train.pkl")
    if os.path.exists(result_file):
        return
        
    train_counts = counts[~mask]
    train_intensity = intensity[~mask]
    
    # Some voxels may be empty now, unfortunately...
    train_alive_voxels = train_counts.sum(axis=0) > 0
    train_counts = train_counts[:, train_alive_voxels]
    train_intensity = train_intensity[:, train_alive_voxels]
    
    # Initialize wandb run
    run = wandb.init(
        project=wandb_project,
        job_type="train",
        config=dict(
            mask=mask,
            sparsity_penalty=sparsity_penalty,
            num_factors=num_factors,
            elastic_net_frac=elastic_net_frac,
            max_num_iters=num_iters,
            num_coord_ascent_iters=num_coord_ascent_iters,
            mean_func=mean_func,
            initialization="nnsvd",
            data_dir=data_dir,
            left_trunc=left_trunc)
        )
        
    # Initialize the seminmf model
    initial_params = seminmf.initialize_nnsvd(train_counts, 
                                              train_intensity, 
                                              num_factors, 
                                              mean_func, 
                                              drugs=None)
    
    print("fitting model")
    params, losses, heldout_loglikes = \
        seminmf.fit_poisson_seminmf(train_counts,
                                    train_intensity,
                                    initial_params,
                                    mask=None,
                                    mean_func=mean_func,
                                    sparsity_penalty=sparsity_penalty,
                                    elastic_net_frac=elastic_net_frac,
                                    num_iters=num_iters,
                                    num_coord_ascent_iters=num_coord_ascent_iters,
                                    tolerance=1e-5
                                    )
        
    wandb.run.summary["final_loss"] = losses[-1]
    wandb.run.summary["heldout_loglike"] = heldout_loglikes[-1]

    # Extract the full factors
    factors = jnp.zeros((num_factors, num_alive_voxels))
    factors = factors.at[:, train_alive_voxels].set(params.factors)

    # Save the results for this bootstrap
    print("saving results")
    with open(result_file, 'wb') as f:
        pickle.dump(dict(mask=mask,
                         factors=factors,
                         count_loadings=params.count_loadings, 
                         intensity_loadings=params.intensity_loadings),
                    f)

    artifact = wandb.Artifact(name="params_pkl", type="model")
    artifact.add_file(local_path=result_file)
    run.log_artifact(artifact)

    # Log results to wandb
    wandb.finish()


if __name__ == '__main__':
    run_sweep()
