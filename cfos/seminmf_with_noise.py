import jax.numpy as jnp
import jax.random as jr

from functools import partial
from jax import jit, vmap
from tensorflow_probability.substrates import jax as tfp
from tqdm.auto import trange

from cfos.prox import prox_grad_descent, prox_grad_descent_python, project_simplex, soft_threshold

tfd = tfp.distributions

EPS = 1e-8

def compute_mean(loadings, weights):
    return jnp.einsum('mk, k...->m...', loadings, weights)


def compute_loss(data, counts, loadings, weights, emission_noise_var):
    scale = jnp.sum(counts > 0)
    mean = compute_mean(loadings, weights)
    dist = tfd.Normal(mean, jnp.sqrt(emission_noise_var / (counts + EPS)))
    lp = dist.log_prob(data)
    return -jnp.where(counts > 0, lp, 0.0).sum() / scale


###
# Helpers for tracking residual
#
def compute_residual(data, loadings, weights):
    return data - compute_mean(loadings, weights)


def update_residual(residual, loadings, weight):
    return residual + jnp.einsum('m, ...->m...', loadings, weight)


def downdate_residual(residual, loadings, weight):
    return residual - jnp.einsum('m, ...->m...', loadings, weight)
#
###

###
# Functions for coordinate ascent
#
@jit
def _update_one_weight(residual,
                       counts,
                       loadings_k,
                       init_weights_k,
                       emission_noise_var,
                       max_num_steps=100,
                       max_stepsize=1.0,
                       discount=0.9,
                       tol=1e-6,
                       verbosity=0):
    r""" Update one factor \theta_k holding the rest fixed using
    proximal gradient descent.
    """
    scale = residual.size

    @jit
    def objective(w):
        mu = jnp.einsum('m, ...->m...', loadings_k, w)
        dist = tfd.Normal(mu, jnp.sqrt(emission_noise_var / (counts + EPS)))
        lp = dist.log_prob(residual)
        return -jnp.where(counts > 0, lp, 0.0).sum() / scale

    # Prox operator is the projection step
    prox = lambda w, stepsize: project_simplex(w)

    return prox_grad_descent(objective,
                             prox,
                             init_weights_k,
                             max_num_steps=max_num_steps,
                             max_stepsize=max_stepsize,
                             discount=discount,
                             tol=tol,
                             verbosity=verbosity)


@jit
def _update_one_weight_exp(residual,
                           counts,
                           loadings_k,
                           emission_noise_var):
    r""" Update one factor \theta_k holding the rest fixed under an 
    *exponential prior distribution.* In this model, we can compute
    the coordinate-wise maximum in closed form. We will set the weights
    to the normalized maximum and rescale the loadings to keep the means
    the same. 
    """
    h = jnp.einsum('m...,m...,m->...', residual, counts / emission_noise_var, loadings_k)
    J = jnp.einsum('m...,m->...', counts / emission_noise_var, loadings_k**2)
    w = jnp.maximum(0.0, h / J)
    scale = w.sum()
    return w / scale, loadings_k * scale

@jit
def _update_all_weights_dpp(data,
                            counts,
                            loadings,
                            init_weights,
                            emission_noise_var,
                            dpp_prior_scale=1e8,
                            max_num_steps=100,
                            max_stepsize=0.01,
                            discount=0.9,
                            tol=1e-6,
                            verbosity=0):
    r""" Update one factor \theta_k holding the rest fixed using
    proximal gradient descent.
    """
    scale = data.size
    num_factors = init_weights.shape[0]

    @jit
    def objective(weights):
        # DPP prior
        sqrt_weights = jnp.sqrt(weights + EPS)
        R = jnp.einsum('k...,j...->kj', sqrt_weights, sqrt_weights) 
        R = 0.5 * (R + R.T) + 1e-4 * jnp.eye(num_factors)
        # lpri = jnp.linalg.slogdet(R )[1]
        L = jnp.linalg.cholesky(R)
        lpri = dpp_prior_scale * jnp.sum(jnp.log(jnp.diag(L)))

        # likelihood
        mu = jnp.einsum('mk, k...->m...', loadings, weights)
        dist = tfd.Normal(mu, jnp.sqrt(emission_noise_var / (counts + EPS)))
        ll = dist.log_prob(data)
        ll = jnp.where(counts > 0, ll, 0.0).sum()

        return -(lpri + ll) / scale

    # Prox operator is the projection step
    prox = lambda weights, stepsize: vmap(project_simplex)(weights)

    weights = prox_grad_descent(objective,
                             prox,
                             init_weights,
                             max_num_steps=max_num_steps,
                             max_stepsize=max_stepsize,
                             discount=discount,
                             tol=tol,
                             verbosity=verbosity)
    # assert jnp.all(jnp.isfinite(weights))
    return weights


@jit
def _update_one_loading(datapoint,
                        counts,
                        init_loading,
                        weights,
                        emission_noise_var,
                        loading_scale=0.1,
                        max_num_steps=100,
                        max_stepsize=0.1,
                        discount=0.9,
                        tol=1e-6,
                        verbosity=0):
    """
    Update the per-mouse factors
    """
    @jit
    def objective(bm):
        mu = jnp.einsum('k, k...->...', bm, weights)
        # return -tfd.Normal(mu, emission_noise_var).log_prob(datapoint).sum()
        dist = tfd.Normal(mu, jnp.sqrt(emission_noise_var / (counts + EPS)))
        lp = dist.log_prob(datapoint)
        return -jnp.where(counts > 0, lp, 0.0).sum() 

    # Prox operator is the soft-thresholding operator
    prox = lambda bm, stepsize: soft_threshold(bm, stepsize / loading_scale)

    return prox_grad_descent(objective,
                             prox,
                             init_loading,
                             max_num_steps=max_num_steps,
                             max_stepsize=max_stepsize,
                             discount=discount,
                             tol=tol,
                             verbosity=verbosity)

def _update_emission_noise_var(residual, 
                               counts, 
                               alpha=0.0001, 
                               beta=0.0001):
    """Update the emission noise variance via coordinate ascent.
    """
    alpha_post = alpha + 0.5 * jnp.sum(counts > 0, axis=0)
    beta_post = beta + 0.5 * jnp.sum(counts * residual**2, axis=0)
    # return beta_post / (alpha_post + 1)
    return beta_post / alpha_post
#
###

### 
# Initialization 
#
def impute_data(data, counts, rank, num_iters=10):
    """Impute missing data. This is a necsesary preprocessing step for
    nnsvd initialization.

    Args:
        data: _description_
        rank: _description_
        num_iters: _description_. Defaults to 10.
    """
    print("Imputing missing data based on a rank {} SVD".format(rank))
    shape = data.shape[1:]
    flat_data = data.reshape(data.shape[0], -1)
    flat_counts = data.reshape(counts.shape[0], -1)

    # Start by infilling mean
    mask = flat_counts > 0
    imputed_data = jnp.where(mask, flat_data, jnp.mean(data[counts > 0]))

    # Iteratively reconstruct with SVD and
    for itr in trange(num_iters):
        U, S, VT = jnp.linalg.svd(imputed_data, full_matrices=False)
        recon = (U[:, :rank] * S[:rank]) @ VT[:rank]
        imputed_data = jnp.where(mask, flat_data, recon)

    return imputed_data.reshape((-1,) + shape)


def initialize_prior(data,
                     num_factors,
                     loading_scale,
                     key):
    num_data = data.shape[0]
    shape = data.shape[1:]
    
    # Initialize the model
    this_key, key = jr.split(key)
    init_loadings = tfd.Laplace(0.0, loading_scale).sample(
        seed=this_key, sample_shape=(num_data, num_factors))

    this_key, key = jr.split(key)
    init_weights = tfd.Dirichlet(jnp.ones(jnp.prod(jnp.array(shape)))).sample(
        seed=this_key, sample_shape=(num_factors,)).reshape(
            (num_factors,) + shape)

    return init_loadings, init_weights


def initialize_nnsvd(data, counts, num_factors, imputation_rank=20):
    """Initialize the model with an SVD. Project the right singular vectors
    onto the non-negative orthant.
    """
    num_mice = data.shape[0]
    shape = data.shape[1:]
    
    # Impute missing data if any is missing
    if jnp.any(counts == 0):
        data = impute_data(data, counts, rank=imputation_rank)

    U, S, VT = jnp.linalg.svd(data.reshape(num_mice, -1), full_matrices=False)

    # flip signs on factors so that each has non-negative mean
    init_loadings = []
    init_weights = []
    for uk, sk, vk in zip(U.T[:num_factors], S[:num_factors], VT[:num_factors]):
        sign = jnp.sign(vk.mean())
        vk = jnp.clip(vk * sign, a_min=1e-6)
        scale = vk.sum()
        init_weights.append((vk / scale).reshape(shape))
        init_loadings.append(uk * sk * scale * sign )

    init_loadings = jnp.column_stack(init_loadings)
    init_weights = jnp.stack(init_weights)
    return init_loadings, init_weights
#
###

###
# Model fitting code
#

# def fit_batch(data, 
#               counts,
#               num_factors,
#               loading_scale=0.1,
#               initialization="nnsvd",
#               num_iters=100,
#               verbosity=0,
#               key=0):
#     """Fit the semi-NMF model with norm-constrained, low-resolution weights

#     Args:
#         data: (num_data,) + (shape) array where `shape` is
#               typically (height, width) or (height, width, depth)
#         num_factors: (int) number of factors in the model
#         key: (int) seed for initializing the model. Defaults to 0.
#     """
#     num_data = data.shape[0]
#     key = jr.PRNGKey(key) if isinstance(key, int) else key

#     # Initialize the loadings and weights
#     initialization = initialization.lower()
#     if initialization == "prior":
#         loadings, weights = \
#             initialize_prior(data,
#                              num_factors,
#                              loading_scale,
#                              key)
        
#     elif initialization == "nnsvd":
#         loadings, weights = \
#             initialize_nnsvd(data,
#                              counts,
#                              num_factors)
#     else:
#         raise Exception("invalid initialization method: {}".format(initialization))

#     # Initialize the emission noise variance
#     residual = compute_residual(data, loadings, weights)
#     emission_noise_var = _update_emission_noise_var(residual, counts)

#     # Run coordinate ascent
#     losses = [compute_loss(data, counts, loadings, weights, emission_noise_var)]
#     # print("initial loss: ", losses[0])
#     pbar = trange(num_iters)
#     for itr in pbar:

#         if verbosity > 0: print("Updating loadings")
#         for m in range(num_data):
#             loadings = loadings.at[m].set(
#                 _update_one_loading(data[m], counts[m], loadings[m], weights,
#                                     emission_noise_var,
#                                     loading_scale=loading_scale,
#                                     verbosity=verbosity))
#             # assert jnp.all(jnp.isfinite(loadings))

#         if verbosity > 0: print("Updating factors")
#         residual = compute_residual(data, loadings, weights)
#         for k in range(num_factors):
#             # "update" residual by adding contribution of current factor
#             residual = update_residual(residual, loadings[:, k], weights[k])
            
#             # Solve for best factor based on residual under *Dirichlet prior*
#             # weights = weights.at[k].set(
#             #     _update_one_weight(residual, counts, loadings[:, k], weights[k],
#             #                        emission_noise_var,
#             #                        verbosity=verbosity))
            
#             # Solve for best factor based on residual under *exponential prior*
#             w_k, beta_k = _update_one_weight_exp(residual, counts, loadings[:, k], emission_noise_var)
#             weights = weights.at[k].set(w_k)
#             loadings = loadings.at[:, k].set(beta_k)
            
#             # "downdate" residual by subtracting contribution of updated factor
#             residual = downdate_residual(residual, loadings[:, k], weights[k])

#         if verbosity > 0: print("Updating emission noise variance")
#         emission_noise_var = _update_emission_noise_var(residual, counts)

#         losses.append(compute_loss(data, counts, loadings, weights, emission_noise_var))
#         # print("loss: ", losses[-1])
#         pbar.set_description("loss: {:.4f}".format(losses[-1]))

#     return jnp.stack(losses), loadings, weights, emission_noise_var


def fit_batch(data, 
              counts,
              num_factors,
              loading_scale=0.1,
              initialization="nnsvd",
              num_iters=100,
              verbosity=0,
              key=0):
    """Fit the semi-NMF model with norm-constrained, low-resolution weights

    Args:
        data: (num_data,) + (shape) array where `shape` is
              typically (height, width) or (height, width, depth)
        num_factors: (int) number of factors in the model
        key: (int) seed for initializing the model. Defaults to 0.
    """
    num_data = data.shape[0]
    key = jr.PRNGKey(key) if isinstance(key, int) else key

    # Initialize the loadings and weights
    initialization = initialization.lower()
    if initialization == "prior":
        loadings, weights = \
            initialize_prior(data,
                             num_factors,
                             loading_scale,
                             key)
        
    elif initialization == "nnsvd":
        loadings, weights = \
            initialize_nnsvd(data,
                             counts,
                             num_factors)
    else:
        raise Exception("invalid initialization method: {}".format(initialization))

    # Initialize the emission noise variance
    residual = compute_residual(data, loadings, weights)
    emission_noise_var = _update_emission_noise_var(residual, counts)

    # Run coordinate ascent
    losses = [compute_loss(data, counts, loadings, weights, emission_noise_var)]
    # print("initial loss: ", losses[0])
    pbar = trange(num_iters)
    for itr in pbar:

        if verbosity > 0: print("Updating loadings")
        for m in range(num_data):
            loadings = loadings.at[m].set(
                _update_one_loading(data[m], counts[m], loadings[m], weights,
                                    emission_noise_var,
                                    loading_scale=loading_scale,
                                    verbosity=verbosity))
            # assert jnp.all(jnp.isfinite(loadings))

        if verbosity > 0: print("Updating factors")
        weights = _update_all_weights_dpp(data, 
                                          counts, 
                                          loadings, 
                                          weights,
                                          emission_noise_var,
                                          dpp_prior_scale=100.0)
        
        if verbosity > 0: print("Updating emission noise variance")
        emission_noise_var = _update_emission_noise_var(residual, counts)

        losses.append(compute_loss(data, counts, loadings, weights, emission_noise_var))
        # print("loss: ", losses[-1])
        pbar.set_description("loss: {:.4f}".format(losses[-1]))

    return jnp.stack(losses), loadings, weights, emission_noise_var
#

#
###