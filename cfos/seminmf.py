import jax.numpy as jnp
import jax.random as jr

from jax import jit
from jax.image import resize
from tensorflow_probability.substrates import jax as tfp
from tqdm.auto import trange

from cfos.prox import prox_grad_descent, prox_grad_descent_python, project_simplex, soft_threshold

tfd = tfp.distributions

RESIZE_METHOD = "linear"
EPS = 1e-8

def compute_mean(data, loadings, weights):
    shape = data.shape[1:]
    num_factors = weights.shape[0]
    return jnp.einsum('mk, k...->m...', loadings,
                      resize(weights, (num_factors,) + shape,
                             method=RESIZE_METHOD))


def compute_loss(data, counts, loadings, weights, emission_noise_scale):
    mean = compute_mean(data, loadings, weights)
    return -tfd.Normal(mean, jnp.sqrt(emission_noise_scale**2 / counts) + EPS).log_prob(data).mean()


def compute_residual(data, loadings, weights):
    return data - compute_mean(data, loadings, weights)


def update_residual(residual, loadings, weight):
    shape = residual.shape[1:]
    return residual + jnp.einsum('m, ...->m...', loadings,
                          resize(weight, shape, method=RESIZE_METHOD))


def downdate_residual(residual, loadings, weight):
    shape = residual.shape[1:]
    return residual - jnp.einsum('m, ...->m...', loadings,
                          resize(weight, shape, method=RESIZE_METHOD))


@jit
def _update_one_weight(residual,
                       counts,
                       loadings_k,
                       init_weights_k,
                       emission_noise_scale=0.5,
                       max_num_steps=100,
                       max_stepsize=1.0,
                       discount=0.9,
                       tol=1e-6,
                       verbosity=0):
    r""" Update one factor \theta_k holding the rest fixed using
    proximal gradient descent.
    """
    # scale = residual.shape[0]
    shp = residual.shape[1:]
    scale = residual.size

    @jit
    def objective(w):
        mu = jnp.einsum('m, ...->m...', loadings_k, resize(w, shp, method=RESIZE_METHOD))
        dist = tfd.Normal(mu, jnp.sqrt(emission_noise_scale**2 / counts) + EPS)
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
def _update_one_loading(datapoint,
                        counts,
                        init_loading,
                        weights,
                        emission_noise_scale=0.5,
                        loading_scale=0.1,
                        max_num_steps=100,
                        max_stepsize=0.1,
                        discount=0.5,
                        tol=1e-4,
                        verbosity=0):
    """
    Update the per-mouse factors
    """
    num_factors = weights.shape[0]

    @jit
    def objective(bm):
        mu = jnp.einsum('k, k...->...', bm,
                        resize(weights,
                               (num_factors,) + datapoint.shape,
                               method=RESIZE_METHOD))
        # return -tfd.Normal(mu, emission_noise_scale).log_prob(datapoint).sum()
        dist = tfd.Normal(mu, jnp.sqrt(emission_noise_scale**2 / counts) + EPS)
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


def initialize_prior(data,
                     num_factors,
                     downsample_factor,
                     loading_scale,
                     key):
    num_data = data.shape[0]
    shape = data.shape[1:]
    down_shape = tuple(d // downsample_factor for d in shape)

    # Initialize the model
    this_key, key = jr.split(key)
    init_loadings = tfd.Laplace(0.0, loading_scale).sample(
        seed=this_key, sample_shape=(num_data, num_factors))

    this_key, key = jr.split(key)
    init_weights = tfd.Dirichlet(jnp.ones(jnp.prod(jnp.array(down_shape)))).sample(
        seed=this_key, sample_shape=(num_factors,)).reshape(
            (num_factors,) + down_shape)

    return init_loadings, init_weights


def initialize_nnsvd(data, num_factors, downsample_factor):
    """Initialize the model with an SVD. Project the right singular vectors
    onto the non-negative orthant.
    """
    num_mice = data.shape[0]
    shape = data.shape[1:]
    down_shape = tuple(d // downsample_factor for d in shape)

    downsampled_data = resize(data, (num_mice,) + down_shape, method=RESIZE_METHOD)
    U, S, VT = jnp.linalg.svd(downsampled_data.reshape(num_mice, -1), full_matrices=False)

    # flip signs on factors so that each has non-negative mean
    init_loadings = []
    init_weights = []
    for uk, sk, vk in zip(U.T[:num_factors], S[:num_factors], VT[:num_factors]):
        sign = jnp.sign(vk.mean())
        vk = jnp.clip(vk * sign, a_min=1e-6)
        scale = vk.sum()
        init_weights.append((vk / scale).reshape(down_shape))
        init_loadings.append(uk * sk * scale * sign )

    init_loadings = jnp.column_stack(init_loadings)
    init_weights = jnp.stack(init_weights)
    return init_loadings, init_weights


def fit_batch(data, 
              counts,
              num_factors,
              emission_noise_scale=0.5,
              loading_scale=0.1,
              initialization="nnsvd",
              downsample_factor=4,
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
        loadings, weights = initialize_prior(data,
                                             num_factors,
                                             downsample_factor,
                                             loading_scale,
                                             key)
    elif initialization == "nnsvd":
        loadings, weights = initialize_nnsvd(data,
                                             num_factors,
                                             downsample_factor)
    else:
        raise Exception("invalid initialization method: {}".format(initialization))

    # Run coordinate ascent
    losses = [compute_loss(data, counts, loadings, weights, emission_noise_scale)]
    print("initial loss: ", losses[0])

    for itr in trange(num_iters):

        print("Updating loadings")
        for m in trange(num_data):
            loadings = loadings.at[m].set(
                _update_one_loading(data[m], counts[m], loadings[m], weights,
                                    emission_noise_scale=emission_noise_scale,
                                    loading_scale=loading_scale,
                                    verbosity=verbosity))

        print("Updating factors")
        residual = compute_residual(data, loadings, weights)
        for k in range(num_factors):
            # "update" residual by adding contribution of current factor
            residual = update_residual(residual, loadings[:, k], weights[k])
            # Solve for best factor based on residual
            weights = weights.at[k].set(
                _update_one_weight(residual, counts, loadings[:, k], weights[k],
                                   emission_noise_scale=emission_noise_scale,
                                   verbosity=verbosity))
            # "downdate" residual by subtracting contribution of updated factor
            residual = downdate_residual(residual, loadings[:, k], weights[k])

        losses.append(compute_loss(data, counts, loadings, weights, emission_noise_scale))
        print("loss: ", losses[-1])

    return jnp.stack(losses), loadings, weights
