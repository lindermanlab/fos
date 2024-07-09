import dataclasses
import jax.numpy as jnp
import jax.random as jr
import warnings

from functools import partial
from fastprogress import progress_bar
from jax import grad, hessian, vmap, lax, jit
from jax.nn import softplus
from jaxtyping import Array, Float

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
warnings.filterwarnings("ignore")

from fos.prox import soft_threshold
from fos.utils import register_pytree_node_dataclass, tree_add, tree_dot


@register_pytree_node_dataclass
@dataclasses.dataclass(frozen=True)
class SemiNMFParams:
    """
    Container for the model parameters
    """
    factors : Float[Array, "num_factors num_columns"]
    count_loadings : Float[Array, "num_rows num_factors"]
    count_row_effects : Float[Array, "num_rows"]
    count_col_effects : Float[Array, "num_columns"]
    intensity_loadings : Float[Array, "num_rows num_factors"]
    intensity_row_effects : Float[Array, "num_rows"]
    intensity_col_effects : Float[Array, "num_columns"]
    intensity_variance : Float[Array, "num_columns"]

    @property
    def num_factors(self):
        return self.factors.shape[0]


def soft_threshold(x, thresh):
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - thresh, 0.0)


def smooth_loss(params, counts, intensity, mask, mean_func):
    # -log p(counts | params)
    g = dict(softplus=softplus)[mean_func]
    count_means = g(params.count_row_effects[:, None] \
                    + params.count_col_effects \
                    + jnp.einsum('mk, kn->mn', params.count_loadings, params.factors))
    loss = jnp.where(mask, -tfd.Poisson(rate=count_means + 1e-8).log_prob(counts), 0.0).sum()

    # -log p(intensity | counts, params)
    intensity_means = params.intensity_row_effects[:, None] \
                    + params.intensity_col_effects \
                    + jnp.einsum('mk, kn->mn', params.intensity_loadings, params.factors)
    intensity_var = params.intensity_variance / (counts + 1e-8)
    loss += jnp.where(mask, -tfd.Normal(intensity_means, jnp.sqrt(intensity_var)).log_prob(intensity), 0.0).sum()
    return loss


grad_smooth_loss = grad(smooth_loss, argnums=0)


def penalty(params, sparsity_penalty, elastic_net_frac):
    loss = elastic_net_frac * sparsity_penalty * jnp.sum(abs(params.count_loadings))
    loss += 0.5 * (1 - elastic_net_frac) * sparsity_penalty * jnp.sum(params.count_loadings ** 2)
    loss += elastic_net_frac * sparsity_penalty * jnp.sum(abs(params.intensity_loadings))
    loss += 0.5 * (1 - elastic_net_frac) * sparsity_penalty * jnp.sum(params.intensity_loadings ** 2)
    return loss


@partial(jit, static_argnums=(4,))
def compute_loss(counts : Float[Array, "num_rows num_columns"],
                 intensity : Float[Array, "num_rows num_columns"],
                 mask : Float[Array, "num_rows num_columns"],
                 params : SemiNMFParams,
                 mean_func : str,
                 sparsity_penalty : float,
                 elastic_net_frac: float
                 ):
    loss = smooth_loss(params, counts, intensity, mask, mean_func)
    loss += penalty(params, sparsity_penalty, elastic_net_frac)
    return loss / counts.size


def heldout_loglike(counts : Float[Array, "num_rows num_columns"],
                    intensity : Float[Array, "num_rows num_columns"],
                    mask : Float[Array, "num_rows num_columns"],
                    params : SemiNMFParams,
                    mean_func : str):
    # -log p(counts | params)
    g = dict(softplus=softplus)[mean_func]
    count_means = g(params.count_row_effects[:, None] \
                    + params.count_col_effects \
                    + jnp.einsum('mk, kn->mn', params.count_loadings, params.factors))
    loss = jnp.where(~mask, tfd.Poisson(rate=count_means + 1e-8).log_prob(counts), 0.0).sum()

    # -log p(intensity | counts, params)
    intensity_means = params.intensity_row_effects[:, None] \
                    + params.intensity_col_effects \
                    + jnp.einsum('mk, kn->mn', params.intensity_loadings, params.factors)
    intensity_var = params.intensity_variance / (counts + 1e-8)
    loss += jnp.where(~mask, tfd.Normal(intensity_means, jnp.sqrt(intensity_var)).log_prob(intensity), 0.0).sum()
    return loss / counts.size



def backtracking_line_search(counts,
                             intensity,
                             mask,
                             params,
                             new_params,
                             mean_func,
                             sparsity_penalty,
                             elastic_net_frac,
                             alpha=0.5,
                             beta=0.5,
                             max_iters=20):
    # Precompute some constants
    dg = grad_smooth_loss(params, counts, intensity, mask, mean_func)
    descent_direction = tree_add(new_params, params, -1.0)
    dg_direc = tree_dot(dg, descent_direction)
    baseline = smooth_loss(params, counts, intensity, mask, mean_func)
    baseline += (1 - alpha) * penalty(params, sparsity_penalty, elastic_net_frac)

    def cond_fun(state):
        stepsize, itr = state
        new_params = tree_add(params, descent_direction, stepsize)
        new_loss = smooth_loss(new_params, counts, intensity, mask, mean_func)
        new_loss += penalty(new_params, sparsity_penalty, elastic_net_frac)
        bound = baseline + alpha * stepsize * dg_direc
        bound += alpha * penalty(new_params, sparsity_penalty, elastic_net_frac)
        return (new_loss > bound) & (itr < max_iters)

    def body_fun(state):
        stepsize, itr = state
        return beta * stepsize, itr + 1

    init_state = (1.0, 0)
    (stepsize, _) = lax.while_loop(cond_fun, body_fun, init_state)
    return tree_add(params, descent_direction, stepsize)


# def backtracking_line_search_scan(counts,
#                              intensity,
#                              mask,
#                              params,
#                              new_params,
#                              mean_func,
#                              sparsity_penalty,
#                              elastic_net_frac,
#                              alpha=0.5,
#                              beta=0.5,
#                              max_iters=20):
#     # Precompute some constants
#     dg = grad_smooth_loss(params, counts, intensity, mask, mean_func)
#     descent_direction = tree_add(new_params, params, -1.0)
#     dg_direc = tree_dot(dg, descent_direction)
#     baseline = smooth_loss(params, counts, intensity, mask, mean_func)
#     baseline += (1 - alpha) * penalty(params, sparsity_penalty, elastic_net_frac)

#     def _step(carry, stepsize):
#         prev_params, prev_criterion_met = carry

#         # Compute new params and check if loss is less than upper bound
#         new_params = tree_add(params, descent_direction, stepsize)
#         new_loss = smooth_loss(new_params, counts, intensity, mask, mean_func)
#         new_loss += penalty(new_params, sparsity_penalty, elastic_net_frac)
#         bound = baseline + alpha * stepsize * dg_direc
#         bound += alpha * penalty(new_params, sparsity_penalty, elastic_net_frac)
#         new_criterion_met = new_loss < bound

#         # If criterion not met on previous iteration, return these params
#         new_carry = lax.cond(
#             prev_criterion_met,
#             lambda: prev_params, prev_criterion_met,
#             lambda: new_params, new_criterion_met)

#         return new_carry, None

#     stepsizes = beta ** jnp.arange(max_iters)
#     (new_params, criterion_met), _ = lax.scan(_step, (params, False), stepsizes)
#     return new_params


@register_pytree_node_dataclass
@dataclasses.dataclass(frozen=True)
class QuadraticApprox:
    """
    Container for the model parameters
    """
    J_counts : Float[Array, "num_rows num_columns"]
    h_counts : Float[Array, "num_rows num_columns"]
    J_intensity : Float[Array, "num_rows num_columns"]
    h_intensity : Float[Array, "num_rows num_columns"]


def compute_quadratic_approx(counts, intensity, mask, params, mean_func):
    # Define key functions of the Poisson GLM
    A = jnp.exp                             # shorthand
    d2A = vmap(vmap(hessian(A)))            # want to broadcast scalar function to whole matrix

    if mean_func.lower() == "softplus":
        # Define numerically safe versions of log(softplus) and its gradients
        f = softplus
        log_softplus = lambda a: jnp.log(f(a))
        thresh = -10
        g = lambda a: jnp.where(a > thresh, log_softplus(a), a)
        dg = lambda a: jnp.where(a > thresh, vmap(vmap(grad(log_softplus)))(a), 1.0)
        d2g = lambda a: jnp.where(a > thresh, vmap(vmap(hessian(log_softplus)))(a), 0.0)
    else:
        raise Exception("invalid mean function: {}".format(mean_func))

    # Compute the quadratic approximation for the Poisson loss
    activations = params.count_row_effects[:, None] \
                + params.count_col_effects \
                + jnp.einsum('mk, kn->mn', params.count_loadings, params.factors)
    predictions = f(activations)
    J_counts = mask * (d2g(activations) * (predictions - counts) + (dg(activations))**2 * d2A(g(activations)))
    h_counts = mask * dg(activations) * (counts - predictions)

    # Compute the quadratic loss for the intensity
    predictions = params.intensity_row_effects[:, None] \
                + params.intensity_col_effects \
                + jnp.einsum('mk, kn->mn', params.intensity_loadings, params.factors)
    J_intensity = mask * counts / params.intensity_variance
    h_intensity = mask * counts / params.intensity_variance * (intensity - predictions)
    return QuadraticApprox(J_counts, h_counts, J_intensity, h_intensity)


def update_loadings(quad_approx,
                    params,
                    sparsity_penalty,
                    elastic_net_frac):
    """
    Update the loadings while holding the remaining parameters fixed.
    """
    def _update_one_loading(h_m, J_m, loading_m):
        """
        Coordinate descent to update the m-th loading
        """
        def _update_one_coord(h_m, args):
            """
            Update one coordinate of the m-th loading
            """
            loading_mk, factor_k = args

            # Compute the numerator (linear term) and denominator (quad term)
            # of the quadratic loss as a function of loading \beta_{mk}
            num = jnp.einsum('n,n->', factor_k, (h_m + J_m * loading_mk * factor_k))
            den = jnp.einsum('n,n,n->', J_m, factor_k, factor_k) + (1 - elastic_net_frac) * sparsity_penalty

            # Apply prox operator
            new_loading_mk = soft_threshold(num, elastic_net_frac * sparsity_penalty) / (den + 1e-8)

            # Update the weighted residual
            h_m += J_m * loading_mk * factor_k
            h_m -= J_m * new_loading_mk * factor_k
            return h_m, new_loading_mk

        # Scan over the (K,) dimension
        h_m, loading_m = lax.scan(_update_one_coord, h_m, (loading_m, params.factors))
        return h_m, loading_m

    # Update the count loadings
    h_counts, count_loadings = vmap(_update_one_loading)(quad_approx.h_counts, quad_approx.J_counts, params.count_loadings)

    # Update the intensity loadings
    h_intensity, intensity_loadings = vmap(_update_one_loading)(quad_approx.h_intensity, quad_approx.J_intensity, params.intensity_loadings)
    params = dataclasses.replace(params,
                                 count_loadings=count_loadings,
                                 intensity_loadings=intensity_loadings)

    quad_approx = dataclasses.replace(quad_approx,
                                      h_counts=h_counts,
                                      h_intensity=h_intensity)
    return quad_approx, params


def update_factors(quad_approx, params):
    """
    Update the factors while holding the remaining parameters fixed.
    """
    def _update_one_column(hc_n, Jc_n, hi_n, Ji_n, factor_n):
        """
        Coordinate descent to update the n-th column of all factors
        """
        def _update_one_coord(carry, args):
            r"""
            Update factor \theta_{nk}
            """
            hc_n, hi_n = carry
            factor_nk, count_loading_k, intensity_loading_k = args

            # Compute the numerator (linear term) and denominator (quad term)
            # of the quadratic loss as a function of factor \theta_{nk}
            num = jnp.einsum('m,m->', count_loading_k, (hc_n + Jc_n * factor_nk * count_loading_k))
            den = jnp.einsum('m,m,m->', Jc_n, count_loading_k, count_loading_k)

            # Add the contribution from the intensity
            num += jnp.einsum('m,m->', intensity_loading_k, (hi_n + Ji_n * factor_nk * intensity_loading_k))
            den += jnp.einsum('m,m,m->', Ji_n, intensity_loading_k, intensity_loading_k)

            # Apply prox operator to ensure loadings are non-negative
            new_factor_nk = jnp.maximum(num, 0.0) / (den + 1e-8)

            # Update weighted residuals
            hc_n += Jc_n * factor_nk * count_loading_k
            hc_n -= Jc_n * new_factor_nk * count_loading_k
            hi_n += Ji_n * factor_nk * intensity_loading_k
            hi_n -= Ji_n * new_factor_nk * intensity_loading_k
            return (hc_n, hi_n), new_factor_nk

        # Scan over the (K,) dimension
        (hc_n, hi_n), factor_n = \
            lax.scan(_update_one_coord,
                     (hc_n, hi_n),
                     (factor_n, params.count_loadings.T, params.intensity_loadings.T))

        return hc_n, hi_n, factor_n

    # Map over the (N,) dimension
    h_countsT, h_intensityT, factorsT = \
        vmap(_update_one_column)(quad_approx.h_counts.T,
                                 quad_approx.J_counts.T,
                                 quad_approx.h_intensity.T,
                                 quad_approx.J_intensity.T,
                                 params.factors.T)
    h_counts = h_countsT.T
    h_intensity = h_intensityT.T
    factors = factorsT.T

    # Make sure the factors are normalized
    scale = factors.sum(axis=1) + 1e-8
    factors /= scale[:, None]
    count_loadings = params.count_loadings * scale
    intensity_loadings = params.intensity_loadings * scale
    params = dataclasses.replace(params,
                                 factors=factors,
                                 count_loadings=count_loadings,
                                 intensity_loadings=intensity_loadings)
    quad_approx = dataclasses.replace(quad_approx, h_counts=h_counts, h_intensity=h_intensity)
    return quad_approx, params


def update_row_effect(quad_approx, params):
    """
    Update the row effect while holding the remaining parameters fixed.
    """
    def _update_one_row(h_m, J_m, row_effect_m):
        """
        Update the m-th row effect
        """
        # Compute the numerator (linear term) and denominator (quad term)
        # of the quadratic loss as a function of loading b_{m}
        num = jnp.einsum('n->', h_m + J_m * row_effect_m)
        den = jnp.einsum('n->', J_m)
        new_row_effect_m = num / den

        # Update residual
        h_m += J_m * row_effect_m
        h_m -= J_m * new_row_effect_m
        return h_m, new_row_effect_m

    # Update the row effects for the count data
    h_counts, count_row_effects = \
        vmap(_update_one_row)(quad_approx.h_counts, quad_approx.J_counts, params.count_row_effects)

    # Do the same for the intensity
    h_intensity, intensity_row_effects = \
        vmap(_update_one_row)(quad_approx.h_intensity, quad_approx.J_intensity, params.intensity_row_effects)

    params = dataclasses.replace(params,
                                 count_row_effects=count_row_effects,
                                 intensity_row_effects=intensity_row_effects)
    quad_approx = dataclasses.replace(quad_approx,
                                      h_counts=h_counts,
                                      h_intensity=h_intensity)
    return quad_approx, params


def update_column_effect(quad_approx, params):
    """
    Update the row effect while holding the remaining parameters fixed.
    """
    def _update_one_column(h_n, J_n, col_effect_n):
        """
        Update the n-th column effect
        """
        # Compute the numerator (linear term) and denominator (quad term)
        # of the quadratic loss as a function of loading c_{n}
        num = jnp.einsum('m->', h_n + J_n * col_effect_n)
        den = jnp.einsum('m->', J_n)
        new_col_effect_n = num / den

        # Update residual
        h_n += J_n * col_effect_n
        h_n -= J_n * new_col_effect_n
        return h_n, new_col_effect_n

    # Update the column effects for the counts
    h_intensityT, count_col_effects = \
        vmap(_update_one_column)(quad_approx.h_counts.T,
                                 quad_approx.J_counts.T,
                                 params.count_col_effects)
    h_counts = h_intensityT.T

    # Do the same for the intensity
    h_intensityT, intensity_col_effects = \
        vmap(_update_one_column)(quad_approx.h_intensity.T,
                                 quad_approx.J_intensity.T,
                                 params.intensity_col_effects)
    h_intensity = h_intensityT.T

    # Make sure column effects sum to zero
    mean = jnp.mean(count_col_effects)
    count_col_effects -= mean
    count_row_effects = params.count_row_effects + mean
    mean = jnp.mean(intensity_col_effects)
    intensity_col_effects -= mean
    intensity_row_effects = params.intensity_row_effects + mean

    params = dataclasses.replace(params,
                                 count_row_effects=count_row_effects,
                                 count_col_effects=count_col_effects,
                                 intensity_row_effects=intensity_row_effects,
                                 intensity_col_effects=intensity_col_effects)
    quad_approx = dataclasses.replace(quad_approx,
                                      h_counts=h_counts,
                                      h_intensity=h_intensity)
    return quad_approx, params


def update_emission_noise_var(counts, 
                              intensity,
                              mask,
                              params,
                              alpha=0.0001, 
                              beta=0.0001):
    """Update the emission noise variance via coordinate ascent.
    """
    # Compute the quadratic loss for the intensity
    predictions = params.intensity_row_effects[:, None] \
                + params.intensity_col_effects \
                + jnp.einsum('mk, kn->mn', params.intensity_loadings, params.factors)
    residual = intensity - predictions

    alpha_post = alpha + 0.5 * jnp.sum((mask * counts) > 0, axis=0)
    beta_post = beta + 0.5 * jnp.sum(mask * counts * residual**2, axis=0)
    intensity_variance = beta_post / alpha_post
    return dataclasses.replace(params, intensity_variance=intensity_variance)


def initialize_random(key, data, num_factors, mean_func):
    m, n = data.shape

    # Convert data to "targets" by inverting mean function
    if mean_func.lower() == "softplus":
        data = jnp.maximum(data, 1e-1)
        targets = data + jnp.log(1 - jnp.exp(-data))
    else:
        raise Exception("Invalid mean function: {}".format(mean_func))

    # Initialize the row and column effects
    row_effects = targets.mean(axis=1)
    col_effects = jnp.zeros(n)

    # initialie the factors randomly
    factors = jr.exponential(key, shape=(num_factors, n))
    factors /= factors.sum(axis=1, keepdims=True)
    loadings = jnp.zeros((m, num_factors))
    return SemiNMFParams(loadings, factors, row_effects, col_effects)


def initialize_nnsvd(counts, intensity, num_factors, mean_func, drugs=None):
    """Initialize the model with an SVD. Project the right singular vectors
    onto the non-negative orthant.
    """
    # Convert data to "targets" by inverting mean function
    if mean_func.lower() == "softplus":
        pseudocounts = jnp.maximum(counts, 1e-1)
        # y = log(1 + e^{x})  ->  x = log(e^y - 1) = y + log(1 - e^{-y})
        targets = pseudocounts + jnp.log(1 - jnp.exp(-pseudocounts))
    else:
        raise Exception("Invalid mean function: {}".format(mean_func))

    num_mice = counts.shape[0]
    shape = counts.shape[1:]

    # Initialize the row- and column-effects
    count_row_effect = jnp.mean(targets, axis=1)
    targets -= count_row_effect[:, None]

    if drugs is not None:
        # !!!!HACK!!!!! Leaking information about drugs into column effect
        count_col_effect = targets[drugs == 10].mean(axis=0)
    else:
        count_col_effect = targets.mean(axis=0)
    targets -= count_col_effect

    # Now run SVD on the residual
    U, S, VT = jnp.linalg.svd(targets.reshape(num_mice, -1), full_matrices=False)

    # flip signs on factors so that each has non-negative mean
    count_loadings = []
    factors = []
    for uk, sk, vk in zip(U.T[:num_factors], S[:num_factors], VT[:num_factors]):
        sign = jnp.sign(vk.mean())
        # sign = 1.0
        vk = jnp.clip(vk * sign, a_min=1e-8)
        scale = vk.sum()
        factors.append((vk / scale).reshape(shape))
        count_loadings.append(uk * sk * scale * sign )

    count_loadings = jnp.column_stack(count_loadings)
    factors = jnp.stack(factors)

    # Now compute intensity loadings using factors from the counts
    targets = jnp.where(counts > 0, intensity, 0.0)
    intensity_row_effect = jnp.sum(counts * targets, axis=1) / jnp.sum(counts, axis=1)
    targets -= intensity_row_effect[:, None]
    if drugs is not None:
        # !!!!HACK!!!!! Leaking information about drugs into column effect
        # intensity_col_effect = targets[drugs == 10].mean(axis=0)
        pass
    else:
        intensity_col_effect = jnp.sum(counts * targets, axis=0) / jnp.sum(counts, axis=0)
    targets -= intensity_col_effect

    # Solve for the intensity loading with a simple linear regression
    # y_m ~ \theta^T @ \beta_m -> \beta_m* = (\theta \theta^T)^{-1} \theta y_m
    # intensity_loadings = jnp.linalg.solve(factors @ factors.T, factors @ targets.T).T
    intensity_loadings = jnp.linalg.solve(
        jnp.einsum('mn, jn, kn->mjk', counts, factors, factors),
        jnp.einsum('mn, mn, kn->mk', counts, targets, factors))
    # intensity_loadings = jnp.einsum('mn,mn,kn->mk', counts, targets, factors) / jnp.einsum('mn, kn, kn->mk', counts, factors, factors)
    assert jnp.all(jnp.isfinite(intensity_loadings))

    # TODO: Compute the intensity variance
    intensity_variance = 1.0 * jnp.ones(shape)

    return SemiNMFParams(factors,
                         count_loadings,
                         count_row_effect,
                         count_col_effect,
                         intensity_loadings,
                         intensity_row_effect,
                         intensity_col_effect,
                         intensity_variance)


def initialize_prediction(counts, 
                          intensity, 
                          initial_params, 
                          mean_func):
    """Initialize the row factors for prediction tasks.
    """
    num_mice, num_voxels = counts.shape

    # Convert data to "targets" by inverting mean function
    if mean_func.lower() == "softplus":
        pseudocounts = jnp.maximum(counts, 1e-1)
        # y = log(1 + e^{x})  ->  x = log(e^y - 1) = y + log(1 - e^{-y})
        targets = pseudocounts + jnp.log(1 - jnp.exp(-pseudocounts))
    else:
        raise Exception("Invalid mean function: {}".format(mean_func))
    
    # Initialize the row- and column-effects
    targets -= initial_params.count_col_effects

    # Solve for count loadings using a simple regression
    factors = initial_params.factors
    padded_factors = jnp.row_stack((jnp.ones(num_voxels), factors))
    count_loadings = jnp.linalg.solve(
        jnp.einsum('jn, kn->jk', padded_factors, padded_factors),
        jnp.einsum('mn, kn->km', targets, padded_factors)).T
    assert jnp.all(jnp.isfinite(count_loadings))

    count_row_effects = count_loadings[:,0]
    count_loadings = count_loadings[:,1:]

    # Solve for the intensity loading with a weighted linear regression,
    # using the counts / variance
    # y_{m,n} ~ N(\beta_m^T \theta_k, \sigma_{m,n}^2)
    # L(\beta_m) = -1/2 \sum_n 1/\sigma_{m,n}^2 (y_{m,n} - \beta_m^T\theta_k)^2 
    #            = -1/2 \beta_m^T J_m \beta_m + \beta_m^\top h_m
    # J_m = \sum_n 1/\sigma_{m,n}^2 \theta_{k,n} \theta_{k,n}^T
    # h_m = \sum_n 1/\sigma_{m,n}^2 y_{m,n} \theta_{k,n}
    # -> \beta_m* = J_m^{-1} h_m
    targets = intensity - initial_params.intensity_col_effects
    weights = counts / initial_params.intensity_variance
    Js = jnp.einsum('mn, jn, kn->mjk', weights, padded_factors, padded_factors)
    hs = jnp.einsum('mn, mn, kn->mk', weights, jnp.where(counts > 0, targets, 0.0), padded_factors)
    intensity_loadings = jnp.linalg.solve(Js, hs[:,:,None])[:,:,0]
    assert jnp.all(jnp.isfinite(intensity_loadings))

    intensity_row_effects = intensity_loadings[:,0]
    intensity_loadings = intensity_loadings[:,1:]

    return dataclasses.replace(initial_params,
                               count_row_effects=count_row_effects,
                               count_loadings=count_loadings,
                               intensity_row_effects=intensity_row_effects,
                               intensity_loadings=intensity_loadings)


def fit_poisson_seminmf(counts,
                        intensity,
                        initial_params,
                        mask=None,
                        mean_func="softplus",
                        num_iters=10,
                        sparsity_penalty=1.0,
                        elastic_net_frac=0.0,
                        num_coord_ascent_iters=20,
                        tolerance=1e-1,
                        ):

    # Make mask if necessary
    mask = jnp.ones_like(counts, dtype=bool) if mask is None else mask
    assert mask.shape == counts.shape

    @jit
    def _step(params, _):
        """
        One sweep over parameter updates
        """
        # Update rows
        quad_approx = compute_quadratic_approx(counts, intensity, mask, params, mean_func)
        def _row_step(carry, _):
            quad_approx, params = carry
            quad_approx, params = update_loadings(quad_approx, params, sparsity_penalty, elastic_net_frac)
            quad_approx, params = update_row_effect(quad_approx, params)
            return (quad_approx, params), None
        (quad_approx, new_params), _ = lax.scan(_row_step, (quad_approx, params), None, length=num_coord_ascent_iters)
        params = backtracking_line_search(counts, intensity, mask, params, new_params, mean_func, sparsity_penalty, elastic_net_frac)

        # Update columns
        quad_approx = compute_quadratic_approx(counts, intensity, mask, params, mean_func)
        def _column_step(carry, _):
            quad_approx, params = carry
            quad_approx, params = update_factors(quad_approx, params)
            # quad_approx, params = update_column_effect(quad_approx, params)
            return (quad_approx, params), None
        (_, new_params), _ = lax.scan(_column_step, (quad_approx, params), None, length=num_coord_ascent_iters)
        params = backtracking_line_search(counts, intensity, mask, params, new_params, mean_func, sparsity_penalty, elastic_net_frac)

        # Update variance
        params = update_emission_noise_var(counts, intensity, mask, params)
        
        loss = compute_loss(counts, intensity, mask, params, mean_func, sparsity_penalty, elastic_net_frac)
        hll = heldout_loglike(counts, intensity, mask, params, mean_func)
        return params, loss, hll

    # Run coordinate ascent
    params = initial_params
    losses = [compute_loss(counts, intensity, mask, params, mean_func, sparsity_penalty, elastic_net_frac)]
    hlls = [heldout_loglike(counts, intensity, mask, params, mean_func)]
    pbar = progress_bar(range(num_iters))
    for itr in pbar:
        params, loss, hll = _step(params, itr)
        losses.append(loss)
        hlls.append(hll)
        assert jnp.isfinite(loss)
        pbar.comment = "loss: {:.4f}".format(losses[-1])

        if abs(losses[-1] - losses[-2]) < tolerance:
            break

    return params, jnp.stack(losses), jnp.stack(hlls)


def predict_poisson_seminmf(counts,
                            intensity,
                            params,
                            mean_func="softplus",
                            num_iters=10,
                            sparsity_penalty=1.0,
                            elastic_net_frac=0.0,
                            num_coord_ascent_iters=20,
                            tolerance=1e-1,
                            ):
    """
    Predict the loadings (row factors) given data and (column) factors.
    """
    # Initialize row parameters
    params = initialize_prediction(counts, intensity, params, mean_func)
    mask = jnp.ones_like(counts, dtype=bool) 

    @jit
    def _step(params, _):
        """
        One sweep over parameter updates
        """
        # Update rows
        quad_approx = compute_quadratic_approx(counts, intensity, mask, params, mean_func)
        def _row_step(carry, _):
            quad_approx, params = carry
            quad_approx, params = update_loadings(quad_approx, params, sparsity_penalty, elastic_net_frac)
            quad_approx, params = update_row_effect(quad_approx, params)
            return (quad_approx, params), None
        (quad_approx, new_params), _ = lax.scan(_row_step, (quad_approx, params), None, length=num_coord_ascent_iters)
        params = backtracking_line_search(counts, intensity, mask, params, new_params, mean_func, sparsity_penalty, elastic_net_frac)

        loss = compute_loss(counts, intensity, mask, params, mean_func, sparsity_penalty, elastic_net_frac)
        hll = heldout_loglike(counts, intensity, mask, params, mean_func)
        return params, loss, hll

    # Run coordinate ascent
    losses = [compute_loss(counts, intensity, mask, params, mean_func, sparsity_penalty, elastic_net_frac)]
    hlls = [heldout_loglike(counts, intensity, mask, params, mean_func)]
    pbar = progress_bar(range(num_iters))
    for itr in pbar:
        params, loss, hll = _step(params, itr)
        losses.append(loss)
        hlls.append(hll)
        assert jnp.isfinite(loss)
        pbar.comment = "loss: {:.4f}".format(losses[-1])

        if abs(losses[-1] - losses[-2]) < tolerance:
            break

    return params, jnp.stack(losses), jnp.stack(hlls)
