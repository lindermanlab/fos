import dataclasses
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from functools import partial
from jax import jit, lax, value_and_grad, Array
from jax.image import resize
from jax.scipy.special import logsumexp
from tensorflow_probability.substrates import jax as tfp
from tqdm.auto import trange

### 
# Globals
tfd = tfp.distributions
EPS = 1e-6


# Helper function to make a dataclass a JAX PyTree
def register_pytree_node_dataclass(cls):
  _flatten = lambda obj: jax.tree_flatten(dataclasses.asdict(obj))
  _unflatten = lambda d, children: cls(**d.unflatten(children))
  jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)
  return cls


@register_pytree_node_dataclass
@dataclasses.dataclass(frozen=True)
class SemiNMFParams:
    """
    Container for the model parameters
    """
    loadings : Array
    factors : Array
    row_effect : Array
    col_effect : Array
    emission_noise_var : Array

    @property
    def ndim(self):
        """
        Return the number of dimensions of each data point / factor
        E.g., for image data this is 2, for volumetric Fos measurements, it's 3.
        """
        return self.factors.ndim - 1
    
    @property
    def num_factors(self):
        return self.factors.shape[0]


###
# Helpers
def _right_broadcast(arr, num_new_dims):
    """
    Add specified number of new axes to the right of arr.
    """
    return jnp.expand_dims(arr, tuple(range(arr.ndim, arr.ndim + num_new_dims)))


def compute_mean(params : SemiNMFParams):
    return _right_broadcast(params.row_effect, params.ndim) + \
        params.col_effect + \
        jnp.einsum('mk, k...->m...', params.loadings, params.factors)


def compute_loss(data, counts, params):
    scale = jnp.sum(counts > 0)
    mean = compute_mean(params)
    dist = tfd.Normal(mean, jnp.sqrt(params.emission_noise_var / (counts + EPS)))
    lp = dist.log_prob(data)
    return -jnp.where(counts > 0, lp, 0.0).sum() / scale


def compute_residual(data, params):
    return data - compute_mean(params)


###
# Functions for coordinate ascent
#
def _initialize_factors_optimizer(init_factors, stepsize):
    optimizer = optax.adam(stepsize)
    optimizer_state = optimizer.init(jnp.log(init_factors))
    return optimizer, optimizer_state


def _update_factors(data,
                    counts,
                    params,
                    optimizer, 
                    optimizer_state,
                    dpp_prior_scale=1e3,
                    dirichlet_prior_conc=0.1,
                    num_steps=100):
    r""" Update the factors under a diversity-promoting DPP prior.
    """
    # assert init_factors.ndim == 3
    scale = data.size

    def objective(log_factors):
        log_normalizer = logsumexp(log_factors, axis=tuple(range(1, params.ndim+1)), keepdims=True)
        log_factors -= log_normalizer
        factors = jnp.exp(log_factors)
        new_params = dataclasses.replace(params, factors=factors)

        # NEW: Penalize similarity to the constant factor
        # # Include the bias in the DPP kernel
        # # bias_factor = jnp.ones((1,) + factors.shape[1:])
        # # bias_factor /= bias_factor.sum()
        # # factors_and_bias = jnp.concatenate([bias_factor, factors])
        # # Compute the DPP kernel derived from a multinomial likelihood
        # # TODO: Blur the factors before computing similarity?
        # # Blurring ~= kernel from a spatially-correlated Gaussian likelihood?
        # sqrt_factors = jnp.sqrt(factors_and_bias + EPS)
        # R = jnp.einsum('k...,j...->kj', sqrt_factors, sqrt_factors) 
        # R = 0.5 * (R + R.T) + 1e-4 * jnp.eye(params.num_factors + 1)

        # NEW: Blur the factors computing similarity?
        # Blurring ~= kernel from a spatially-correlated Gaussian likelihood
        # blur_factors = resize(params.factors, (factors.shape[0],) + tuple(shp // 4 for shp in factors.shape[1:]), method="linear")
        
        # OLD
        # Compute the DPP kernel derived from a multinomial likelihood
        sqrt_factors = jnp.sqrt(factors + EPS)
        R = jnp.einsum('k...,j...->kj', sqrt_factors, sqrt_factors) 
        R = 0.5 * (R + R.T) + 1e-4 * jnp.eye(params.num_factors)

        # Compute the log determinant
        # lpri = jnp.linalg.slogdet(R )[1]
        L = jnp.linalg.cholesky(R)
        log_prior = dpp_prior_scale * jnp.sum(jnp.log(jnp.diag(L)))

        # Add the log prob of the factors under a Dirichlet prior
        # log_prior += (dirichlet_prior_conc - 1.0) * jnp.sum(log_factors)

        # Compute the log likelihood
        mu = compute_mean(new_params)
        dist = tfd.Normal(mu, jnp.sqrt(new_params.emission_noise_var / (counts + EPS)))
        ll = dist.log_prob(data)
        ll = jnp.where(counts > 0, ll, 0.0).sum()
        return -(log_prior + ll) / scale

    # One step of the algorithm
    def train_step(carry, args):
        log_factors, optimizer_state = carry
        loss, grads = value_and_grad(objective)(log_factors)
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        log_factors = optax.apply_updates(log_factors, updates)
        return (log_factors, optimizer_state), loss

    # Run the optimizer
    initial_carry =  (jnp.log(params.factors), optimizer_state)
    (log_factors, optimizer_state), losses = \
        lax.scan(train_step, initial_carry, None, length=num_steps)

    # Return the updated parameters
    log_normalizer = logsumexp(log_factors, axis=tuple(range(1, params.ndim+1)), keepdims=True)
    factors = jnp.exp(log_factors - log_normalizer)
    params = dataclasses.replace(params, factors=factors)
    return params, optimizer_state
    

@jit
def _update_loadings(data : Array,
                     counts : Array,
                     params : SemiNMFParams):
    """
    Update the per-mouse factors to their coordinate-wise maximum
    under a Gaussian prior.
    """
    # Original
    # J = jnp.einsum('i...,m...,j...->mij', params.factors, counts / params.emission_noise_var, params.factors)
    # J = 0.5 * (J + jnp.swapaxes(J, -1, -2)) + EPS * jnp.eye(params.num_factors)
    # h = jnp.einsum('m...,m...,k...->mk', data, counts / params.emission_noise_var, params.factors)
    # loadings = jnp.linalg.solve(J, h)
    # return dataclasses.replace(params, loadings=loadings)

    # New
    factors_and_bias = jnp.concatenate([
        jnp.ones((1,) + params.factors.shape[1:]),
        params.factors
    ])
    J = jnp.einsum('i...,m...,j...->mij', factors_and_bias, counts / params.emission_noise_var, factors_and_bias)
    J = 0.5 * (J + jnp.swapaxes(J, -1, -2)) + EPS * jnp.eye(params.num_factors + 1)
    h = jnp.einsum('m...,m...,k...->mk', data, counts / params.emission_noise_var, factors_and_bias)
    out = jnp.linalg.solve(J, h)
    row_effect, loadings = out[:, 0], out[:, 1:]
    return dataclasses.replace(params, row_effect=row_effect, loadings=loadings)
    

def _update_emission_noise_var(residual : Array, 
                               counts : Array, 
                               params : SemiNMFParams,
                               alpha=0.0001, 
                               beta=0.0001):
    """Update the emission noise variance via coordinate ascent.
    """
    alpha_post = alpha + 0.5 * jnp.sum(counts > 0, axis=0)
    beta_post = beta + 0.5 * jnp.sum(counts * residual**2, axis=0)
    emission_noise_var = beta_post / alpha_post     # (this is the mean, not the mode)
    return dataclasses.replace(params, emission_noise_var=emission_noise_var)


### 
# Initialization 
#
def impute_data(data : Array, 
                counts : Array, 
                rank : int, 
                num_iters : int = 10):
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
    flat_counts = counts.reshape(counts.shape[0], -1)

    # Start by infilling mean
    mask = flat_counts > 0
    imputed_data = jnp.where(mask, flat_data, jnp.mean(data[counts > 0]))

    # Iteratively reconstruct with SVD and
    for itr in trange(num_iters):
        U, S, VT = jnp.linalg.svd(imputed_data, full_matrices=False)
        recon = (U[:, :rank] * S[:rank]) @ VT[:rank]
        imputed_data = jnp.where(mask, flat_data, recon)

    return imputed_data.reshape((-1,) + shape)


def initialize_prior(data : Array,
                     counts : Array, 
                     num_factors : int,
                     loading_scale : float,
                     key : object):
    num_data = data.shape[0]
    shape = data.shape[1:]

    # Initialize the row- and column-effects
    row_effect = jnp.mean(data, axis=tuple(range(1, data.ndim)))
    col_effect = jnp.zeros(data.shape[1:])
    
    # Initialize the model
    this_key, key = jr.split(key)
    loadings = tfd.Laplace(0.0, loading_scale).sample(
        seed=this_key, sample_shape=(num_data, num_factors))

    this_key, key = jr.split(key)
    factors = tfd.Dirichlet(jnp.ones(jnp.prod(jnp.array(shape)))).sample(
        seed=this_key, sample_shape=(num_factors,)).reshape(
            (num_factors,) + shape)

    # Initialize the emission noise variance
    params = SemiNMFParams(loadings, factors, row_effect, col_effect, None)
    residual = compute_residual(data, params)
    params = _update_emission_noise_var(residual, counts, params)
    return params


def initialize_nnsvd(data, counts, num_factors, imputation_rank=20):
    """Initialize the model with an SVD. Project the right singular vectors
    onto the non-negative orthant.
    """
    num_mice = data.shape[0]
    shape = data.shape[1:]
    
    # Impute missing data if any is missing
    # Can't do the mean subtraction yet, but that should be captured by the 
    # first factor anyway.
    if jnp.any(counts == 0):
        data = impute_data(data, counts, rank=imputation_rank)

    # Initialize the row- and column-effects
    row_effect = jnp.mean(data, axis=tuple(range(1, data.ndim)))
    col_effect = jnp.zeros(data.shape[1:])

    # Now subtract the mean before doing the SVD
    data -= jnp.mean(data, axis=tuple(range(1, data.ndim)), keepdims=True)
    U, S, VT = jnp.linalg.svd(data.reshape(num_mice, -1), full_matrices=False)

    # flip signs on factors so that each has non-negative mean
    loadings = []
    factors = []
    for uk, sk, vk in zip(U.T[:num_factors], S[:num_factors], VT[:num_factors]):
        sign = jnp.sign(vk.mean())
        vk = jnp.clip(vk * sign, a_min=1e-6)
        scale = vk.sum()
        factors.append((vk / scale).reshape(shape))
        loadings.append(uk * sk * scale * sign )

    loadings = jnp.column_stack(loadings)
    factors = jnp.stack(factors)

    # Initialize the emission noise variance
    params = SemiNMFParams(loadings, factors, row_effect, col_effect, None)
    residual = compute_residual(data, params)
    params = _update_emission_noise_var(residual, counts, params)

    return params

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
    """Fit the semi-NMF model with norm-constrained, low-resolution factors

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
        params = initialize_prior(data,
                                  counts,
                                  num_factors,
                                  loading_scale,
                                  key)
        
    elif initialization == "nnsvd":
        params = initialize_nnsvd(data,
                                  counts,
                                  num_factors)
    else:
        raise Exception("invalid initialization method: {}".format(initialization))

    # HARD CODE THE TRUE EMISSION VARIANCE
    params = dataclasses.replace(params, emission_noise_var=0.1**2 * jnp.ones(data.shape[1:]))

    # Initialize the factors optimizer
    optimizer, optimizer_state = _initialize_factors_optimizer(params.factors, stepsize)
    update_factors = jit(partial(_update_factors, 
                                 optimizer=optimizer,
                                 dpp_prior_scale=dpp_prior_scale))

    # Run coordinate ascent
    losses = [compute_loss(data, counts, params)]
    pbar = trange(num_iters)
    for _ in pbar:
        if verbosity > 0: print("Updating loadings")
        params = _update_loadings(data, counts, params)

        if verbosity > 0: print("Updating factors")
        params, optimizer_state = \
            update_factors(data,
                           counts,
                           params,
                           optimizer_state=optimizer_state)
        
        # if verbosity > 0: print("Updating emission noise variance")
        # residual = compute_residual(data, params)
        # params = _update_emission_noise_var(residual, counts, params)

        losses.append(compute_loss(data, counts, params))
        pbar.set_description("loss: {:.4f}".format(losses[-1]))

    return jnp.stack(losses), params
