import jax.numpy as jnp
import jax.random as jr
import optax

from functools import partial
from jax import jit, lax, value_and_grad
from jax.scipy.special import logsumexp
from tensorflow_probability.substrates import jax as tfp
from tqdm.auto import trange

### 
# Globals
tfd = tfp.distributions
EPS = 1e-8


###
# Helpers
def compute_mean(loadings, weights):
    return jnp.einsum('mk, k...->m...', loadings, weights)


def compute_loss(data, counts, loadings, weights, emission_noise_var):
    scale = jnp.sum(counts > 0)
    mean = compute_mean(loadings, weights)
    dist = tfd.Normal(mean, jnp.sqrt(emission_noise_var / (counts + EPS)))
    lp = dist.log_prob(data)
    return -jnp.where(counts > 0, lp, 0.0).sum() / scale


def compute_residual(data, loadings, weights):
    return data - compute_mean(loadings, weights)


###
# Functions for coordinate ascent
#
def _initialize_weights_optimizer(init_weights, stepsize):
    optimizer = optax.adam(stepsize)
    optimizer_state = optimizer.init(jnp.log(init_weights))
    return optimizer, optimizer_state


def _update_weights(data,
                    counts,
                    loadings,
                    init_weights,
                    emission_noise_var,
                    optimizer, 
                    optimizer_state,
                    dpp_prior_scale=1e3,
                    num_steps=100):
    r""" Update the weights under a diversity-promoting DPP prior.
    """
    # assert init_weights.ndim == 3
    scale = data.size
    num_factors = init_weights.shape[0]

    def objective(log_weights):
        weights = jnp.exp(log_weights - logsumexp(log_weights, axis=(1, 2), keepdims=True))
        # TODO: Blur the weights before computing similarity?
        
        # Compute the DPP kernel derived from a multinomial likelihood
        # TODO: Consider doing this under a Gaussian likelihood
        sqrt_weights = jnp.sqrt(weights + EPS)
        R = jnp.einsum('k...,j...->kj', sqrt_weights, sqrt_weights) 
        R = 0.5 * (R + R.T) + 1e-4 * jnp.eye(num_factors)

        # Compute the log determinant
        # lpri = jnp.linalg.slogdet(R )[1]
        L = jnp.linalg.cholesky(R)
        lpri = dpp_prior_scale * jnp.sum(jnp.log(jnp.diag(L)))

        # likelihood
        mu = jnp.einsum('mk, k...->m...', loadings, weights)
        dist = tfd.Normal(mu, jnp.sqrt(emission_noise_var / (counts + EPS)))
        ll = dist.log_prob(data)
        ll = jnp.where(counts > 0, ll, 0.0).sum()
        return -(lpri + ll) / scale

    # One step of the algorithm
    def train_step(carry, args):
        log_weights, optimizer_state = carry
        loss, grads = value_and_grad(objective)(log_weights)
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        log_weights = optax.apply_updates(log_weights, updates)
        return (log_weights, optimizer_state), loss

    # Run the optimizer
    initial_carry =  (jnp.log(init_weights), optimizer_state)
    (log_weights, optimizer_state), losses = \
        lax.scan(train_step, initial_carry, None, length=num_steps)

    # Return the updated parameters
    weights = jnp.exp(log_weights - logsumexp(log_weights, axis=(1, 2), keepdims=True))
    return weights, optimizer_state
    

@jit
def _update_loadings(data,
                     counts,
                     weights,
                     emission_noise_var):
    """
    Update the per-mouse factors to their coordinate-wise maximum
    under a Gaussian prior.
    """
    num_factors = weights.shape[0]
    J = jnp.einsum('i...,m...,j...->mij', weights, counts / emission_noise_var, weights)
    J = 0.5 * (J + jnp.swapaxes(J, -1, -2)) + EPS * jnp.eye(num_factors)
    h = jnp.einsum('m...,m...,k...->mk', data, counts / emission_noise_var, weights)
    return jnp.linalg.solve(J, h)


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
                     counts, 
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

    # Initialize the emission noise variance
    residual = compute_residual(data, init_loadings, init_weights)
    emission_noise_var = _update_emission_noise_var(residual, counts)

    return init_loadings, init_weights, emission_noise_var


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

    # Initialize the emission noise variance
    residual = compute_residual(data, init_loadings, init_weights)
    emission_noise_var = _update_emission_noise_var(residual, counts)

    return init_loadings, init_weights, emission_noise_var
#
###

###
# Model fitting code
#
def fit_batch(data, 
              counts,
              num_factors,
              initialization="nnsvd",
              stepsize=1e-2,
              loading_scale=100.,
              dpp_prior_scale=1e3,
              num_iters=100,
              verbosity=0,
              key=None):
    """Fit the semi-NMF model with norm-constrained, low-resolution weights

    Args:
        data: (num_data,) + (shape) array where `shape` is
              typically (height, width) or (height, width, depth)
        num_factors: (int) number of factors in the model
        key: (int) seed for initializing the model. Defaults to 0.
    """
    # Initialize the parameters
    initialization = initialization.lower()
    if initialization == "prior":
        key = jr.PRNGKey(key) if isinstance(key, int) else key
        loadings, weights, emission_noise_var = \
            initialize_prior(data,
                             counts,
                             num_factors,
                             loading_scale,
                             key)
        
    elif initialization == "nnsvd":
        loadings, weights, emission_noise_var = \
            initialize_nnsvd(data,
                             counts,
                             num_factors)
    else:
        raise Exception("invalid initialization method: {}".format(initialization))

    # Initialize the weights optimizer
    optimizer, optimizer_state = _initialize_weights_optimizer(weights, stepsize)
    update_weights = jit(partial(_update_weights, 
                                 optimizer=optimizer,
                                 dpp_prior_scale=dpp_prior_scale))

    # Run coordinate ascent
    losses = [compute_loss(data, counts, loadings, weights, emission_noise_var)]
    pbar = trange(num_iters)
    for _ in pbar:
        if verbosity > 0: print("Updating loadings")
        loadings = _update_loadings(data, counts, weights, emission_noise_var)

        if verbosity > 0: print("Updating factors")
        weights, optimizer_state = \
            update_weights(data,
                            counts,
                            loadings,
                            weights,
                            emission_noise_var, 
                            optimizer_state=optimizer_state)
        
        if verbosity > 0: print("Updating emission noise variance")
        residual = compute_residual(data, loadings, weights)
        emission_noise_var = _update_emission_noise_var(residual, counts)

        losses.append(compute_loss(data, counts, loadings, weights, emission_noise_var))
        pbar.set_description("loss: {:.4f}".format(losses[-1]))

    return jnp.stack(losses), loadings, weights, emission_noise_var
#

#
###