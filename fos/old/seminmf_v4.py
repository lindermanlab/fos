import dataclasses
import jax
import jax.numpy as jnp
import jax.random as jr
import warnings

from functools import partial
from jax import grad, hessian, vmap, lax, jit
from jax.nn import softplus
from jaxtyping import Array, Float, PyTree
from tensorflow_probability.substrates import jax as tfp
from tqdm.auto import trange

tfd = tfp.distributions
warnings.filterwarnings("ignore")


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
    loadings : Float[Array, "num_rows num_factors"]
    factors : Float[Array, "num_factors num_columns"]
    row_effects : Float[Array, "num_rows"]
    column_effects : Float[Array, "num_columns"]

    @property
    def num_factors(self):
        return self.factors.shape[0]


def soft_threshold(x, thresh):
    return x/jnp.abs(x) * jnp.maximum(jnp.abs(x) - thresh, 0.0)


def compute_activations(params : SemiNMFParams):
    return params.row_effects[:, None] + params.column_effects + \
        jnp.einsum('mk, kn->mn', params.loadings, params.factors)


def compute_loss(data, params : SemiNMFParams, mean_func):
    activations = compute_activations(params)
    return tfd.Poisson(rate=mean_func(activations)).log_prob(data).mean()


def compute_quadratic_approx(data, params, mean_func):
    # Define key functions of the Poisson GLM
    A = jnp.exp                             # shorthand
    d2A = vmap(vmap(hessian(A)))            # want to broadcast scalar function to whole matrix
    g = lambda a: jnp.log(mean_func(a))     # mapping to natural params
    dg = vmap(vmap(grad(g)))                # want to broadcast scalar function to whole matrix
    d2g = vmap(vmap(hessian(g)))            # want to broadcast scalar function to whole matrix

    # Compute the quadratic approximation
    activations = compute_activations(params)
    predictions = mean_func(activations)
    weights = d2g(activations) * (predictions - data) \
        + (dg(activations))**2 * d2A(g(activations))
    # working_data = activations + dg(activations) / weights * (data - predictions)
    # residual = working_data - activations
    residual = dg(activations) / weights * (data - predictions)
    assert jnp.all(weights > 0.0)
    
    return residual, weights
    

def update_loadings(residual, weights, params, sparsity_penalty):
    """
    Update the loadings while holding the remaining parameters fixed.
    """
    def _update_one_loading(residual_m, weights_m, loading_m):
        """
        Coordinate descent to update the m-th loading
        """
        def _update_one_coord(residual_m, args):
            """
            Update one coordinate of the m-th loading
            """
            loading_mk, factor_k = args
            
            # Compute the numerator (linear term) and denominator (quad term)
            # of the quadratic loss as a function of loading \beta_{mk}
            num = jnp.einsum('n,n,n->', weights_m, factor_k, (residual_m + loading_mk * factor_k))
            den = jnp.einsum('n,n,n->', weights_m, factor_k, factor_k)

            # Apply prox operator
            new_loading_mk = soft_threshold(num, sparsity_penalty) / den

            # Update residual (y - \sum_i w_i x_i) with the new w_i
            residual_m += loading_mk * factor_k
            residual_m -= new_loading_mk * factor_k

            return residual_m, new_loading_mk

        # Scan over the (K,) dimension
        residual_m, loading_m = lax.scan(_update_one_coord, residual_m, (loading_m, params.factors))
        return residual_m, loading_m

    # Map over the (M,) dimension
    residual, loadings = vmap(_update_one_loading)(residual, weights, params.loadings) 
    params = dataclasses.replace(params, loadings=loadings)
    return residual, params
    

def update_factors(residual, weights, params):
    """
    Update the factors while holding the remaining parameters fixed.
    """
    def _update_one_column(residual_n, weights_n, factor_n):
        """
        Coordinate descent to update the n-th column of all factors
        """
        def _update_one_coord(residual_n, args):
            r"""
            Update factor \theta_{nk}
            """
            factor_nk, loading_k = args
            
            # Compute the numerator (linear term) and denominator (quad term)
            # of the quadratic loss as a function of factor \theta_{nk}
            num = jnp.einsum('m,m,m->', weights_n, loading_k, (residual_n + factor_nk * loading_k))
            den = jnp.einsum('m,m,m->', weights_n, loading_k, loading_k)

            # Apply prox operator to ensure loadings are non-negative
            new_loading_nk = jnp.maximum(num, 0.0) / den

            # Update residual (y - \sum_i w_i x_i) with the new w_i
            residual_n += factor_nk * loading_k
            residual_n -= new_loading_nk * loading_k

            return residual_n, new_loading_nk

        # Scan over the (K,) dimension
        residual_n, factor_n = lax.scan(_update_one_coord, residual_n, (factor_n, params.loadings.T))
        return residual_n, factor_n

    # Map over the (N,) dimension
    residualT, factorsT = vmap(_update_one_column)(residual.T, weights.T, params.factors.T)
    residual = residualT.T
    factors = factorsT.T

    # Make sure the factors are normalized
    scale = factors.sum(axis=1)
    factors /= scale[:, None]
    loadings = params.loadings * scale
    params = dataclasses.replace(params, factors=factors, loadings=loadings)
    return residual, params
    

def update_row_effect(residual, weights, params):
    """
    Update the row effect while holding the remaining parameters fixed.
    """
    def _update_one_row(residual_m, weights_m, row_effect_m):
        """
        Update the m-th row effect
        """
        # Compute the numerator (linear term) and denominator (quad term)
        # of the quadratic loss as a function of loading b_{m}
        num = jnp.einsum('n,n->', weights_m, (residual_m + row_effect_m))
        den = jnp.einsum('n->', weights_m)
        new_row_effect_m = num / den

        # Update residual
        residual_m += row_effect_m
        residual_m -= new_row_effect_m
        return residual_m, new_row_effect_m

    # Map over the (M,) dimension
    residual, row_effects = vmap(_update_one_row)(residual, weights, params.row_effects) 
    params = dataclasses.replace(params, row_effects=row_effects)
    return residual, params


def update_column_effect(residual, weights, params):
    """
    Update the row effect while holding the remaining parameters fixed.
    """
    def _update_one_column(residual_n, weights_n, col_effect_n):
        """
        Update the n-th column effect
        """
        # Compute the numerator (linear term) and denominator (quad term)
        # of the quadratic loss as a function of loading c_{n}
        num = jnp.einsum('m,m->', weights_n, (residual_n + col_effect_n))
        den = jnp.einsum('m->', weights_n)
        new_col_effect_n = num / den

        # Update residual
        residual_n += col_effect_n
        residual_n -= new_col_effect_n
        return residual_n, new_col_effect_n

    # Map over the (N,) dimension
    residualT, column_effects = vmap(_update_one_column)(residual.T, weights.T, params.column_effects)
    residual = residualT.T

    # Make sure column effects sum to zero
    mean = jnp.mean(column_effects)
    column_effects -= mean
    row_effects = params.row_effects + mean
    params = dataclasses.replace(params, row_effects=row_effects, column_effects=column_effects)
    return residual, params


def initialize_greedy(data, num_factors):
    # Initialize the row and column effects
    row_effects = data.mean(axis=1)
    residual = data - row_effects[:, None]
    col_effect = residual.mean(axis=0)
    residual -= col_effect
    

    # Greedily initialize loadings and factors
    residual = data
    loadings = []
    factors = []
    
    for k in range(num_factors):
        # Find the voxel with the highest residual variance
        # weighted_mean = jnp.einsum('nij, nij->ij', residual, counts) / jnp.sum(counts, axis=0)
        # weighted_var = jnp.einsum('nij, nij->ij', (residual - weighted_mean)**2, counts) / jnp.sum(counts, axis=0)
        idx = jnp.argmax(jnp.var(residual, axis=0))

        # Use its residual as the loading
        loading = residual[:, idx]
        
        # Compute the least squares estimate of the factor
        factor = jnp.einsum('mn,m->n', residual, loading) / jnp.linalg.norm(loading, 2)**2
        
        # Clip and rescale
        factor = jnp.maximum(factor, 0.0)
        scale = factor.sum()
        assert scale > 0.0
        factor /= scale
        loading *= scale
        
        # Append
        loadings.append(loading)
        factors.append(factor)

        # Update residual
        residual -= jnp.einsum('m,n->mn', loading, factor)

    loadings = jnp.column_stack(loadings)
    factors = jnp.stack(factors)
    return SemiNMFParams(loadings, factors, row_effects, col_effect)
    

def fit_poisson_seminmf(data,
                        num_factors,
                        initial_params=None,
                        mean_func=softplus,
                        num_iters=10,
                        sparsity_penalty=1.0,
                        verbosity=0,
                        ):
    
    # Initialize 
    if initial_params is None:
        params = initialize_greedy(data, num_factors)
    else:
        params = initial_params
    
    # @jit
    def _step(params, _):
        """
        One sweep over parameter updates
        """
        residual, weights = compute_quadratic_approx(data, params, mean_func)
        residual, params = update_loadings(residual, weights, params, sparsity_penalty)
        residual, params = update_factors(residual, weights, params)
        # residual, params = update_column_effect(residual, weights, params)
        # residual, params = update_row_effect(residual, weights, params)
        loss = compute_loss(data, params, mean_func)
        return params, loss
        
    # Run coordinate ascent
    losses = [compute_loss(data, params, mean_func)]
    pbar = trange(num_iters)
    for itr in pbar:
        params, loss = _step(params, itr)
        losses.append(loss)
        pbar.set_description("loss: {:.4f}".format(losses[-1]))

    return params, jnp.stack(losses)
