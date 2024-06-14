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

def load_data(data_file, left_trunc):
    """Load the combined data
     
    Note: files are permanently available on [Google Drive](https://drive.google.com/drive/u/0/folders/1xD54Uq4cKJsACmy8KVZZZ5YGCjF6ZrkC). 
    The scratch storage may be deleted periodically by Sherlock.
    """
    data = np.load(data_file)
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
    return intensity, counts, alive_voxels


@click.command()
@click.option('--data_file', default="/scratch/groups/swl1/serotonin/npz_4/downsampled_data_4.npz", help='path to data file (at 100um resolution).')
@click.option('--results_dir', default="/home/groups/swl1/swl1/fos/results/2024_05_30-21_19-bootstrap", help='path to folder where results are stored.')
@click.option('--left_trunc', default=8, help='number of voxels to drop on the left side of the brain')
@click.option('--num_factors', default=14, help='number of factors.')
@click.option('--sparsity_penalty', default=0.01, help='sparsity penalty.')
@click.option('--mean_func', default="softplus", help='mean (inverse link) function.')
@click.option('--elastic_net_frac', default=1.0, help='fraction of L1 vs L2 penalty.')
@click.option('--num_iters', default=500, help='max number of iterations of EM')
@click.option('--num_coord_ascent_iters', default=1, help='number of inner iterations of coordinate ascent')
@click.option('--num_bootstrap', default=1000, help='number of bootstrap iterations')
@click.option('--wandb_project', default="serotonin-fos-seminmf-bootstrap", help='wandb project name')
def run_sweep(data_file, 
              results_dir, 
              left_trunc, 
              num_factors, 
              sparsity_penalty, 
              mean_func, 
              elastic_net_frac, 
              num_iters, 
              num_coord_ascent_iters, 
              num_bootstrap,
              wandb_project):
    
    # Load the data
    intensity, counts, alive_voxels = load_data(data_file, left_trunc)
    num_mice, num_alive_voxels = counts.shape

    # Run bootstrap 
    for i in progress_bar(range(num_bootstrap)):
        print(f"bootstrap sample {i}")
        result_file = os.path.join(results_dir, f"params_{i:04d}.pkl")
        if os.path.exists(result_file):
            continue
            
        # Create a bootstrapped dataset
        key = jr.PRNGKey(i)
        bootstrap_inds = jr.choice(key, num_mice, shape=(num_mice,), replace=True)
        bootstrap_counts = counts[bootstrap_inds]
        bootstrap_intensity = intensity[bootstrap_inds]
        
        # Some voxels may be empty now, unfortunately...
        bootstrap_alive_voxels = bootstrap_counts.sum(axis=0) > 0
        bootstrap_counts = bootstrap_counts[:, bootstrap_alive_voxels]
        bootstrap_intensity = bootstrap_intensity[:, bootstrap_alive_voxels]
        
        # Initialize wandb run
        run = wandb.init(
            project=wandb_project,
            job_type="train",
            config=dict(
                bootstrap_iter=i,
                bootstrap_inds=bootstrap_inds,
                sparsity_penalty=sparsity_penalty,
                num_factors=num_factors,
                elastic_net_frac=elastic_net_frac,
                max_num_iters=num_iters,
                num_coord_ascent_iters=num_coord_ascent_iters,
                mean_func=mean_func,
                initialization="nnsvd",
                data_file=data_file,
                drug_file="",
                left_trunc=left_trunc)
            )
        
        
        # Initialize the seminmf model
        initial_params = seminmf.initialize_nnsvd(bootstrap_counts, 
                                                bootstrap_intensity, 
                                                num_factors, 
                                                mean_func, 
                                                drugs=None)
        
        print("fitting model")
        params, losses, heldout_loglikes = \
            seminmf.fit_poisson_seminmf(bootstrap_counts,
                                        bootstrap_intensity,
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
        factors = factors.at[:, bootstrap_alive_voxels].set(params.factors)

        # Save the results for this bootstrap
        print("saving results")
        with open(result_file, 'wb') as f:
            pickle.dump(dict(bootstrap_inds=bootstrap_inds,
                            factors=factors,
                            count_loadings=params.count_loadings, 
                            intensity_loadings=params.intensity_loadings,
                            ), 
                        f)

        artifact = wandb.Artifact(name="params_pkl", type="model")
        artifact.add_file(local_path=result_file)
        run.log_artifact(artifact)

        # Log results to wandb
        wandb.finish()
    

if __name__ == '__main__':
    run_sweep()
