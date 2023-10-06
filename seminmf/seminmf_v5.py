import dataclasses
import jax
import jax.numpy as jnp
import jax.random as jr
import warnings

from functools import partial
from jax import grad, hessian, vmap, lax, jit, tree_map
from jax.nn import softplus
from jax.tree_util import tree_reduce
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


def convex_combo(pytree1, pytree2, stepsize):
    f = lambda x, y: (1 - stepsize) * x + stepsize * y
    return tree_map(f, pytree1, pytree2)


def tree_add(pytree1, pytree2, scale=1.0):
    return tree_map(lambda x, y: x + scale * y, pytree1, pytree2)


def tree_dot(pytree1, pytree2):
    return tree_reduce(lambda carry, x: carry + x, 
                       tree_map(lambda x, y: jnp.sum(x * y), pytree1, pytree2),
                       0.0)


def soft_threshold(x, thresh):
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - thresh, 0.0)


def compute_activations(params : SemiNMFParams):
    return params.row_effects[:, None] + params.column_effects + \
        jnp.einsum('mk, kn->mn', params.loadings, params.factors)


def compute_loss(data : Float[Array, "num_rows num_columns"], 
                 params : SemiNMFParams, 
                 mean_func : str,
                 sparsity_penalty : float,
                 elastic_net_frac: float
                 ):
    g = dict(softplus=softplus)[mean_func]
    activations = compute_activations(params)
    loss = -tfd.Poisson(rate=g(activations) + 1e-8).log_prob(data).sum()
    loss += elastic_net_frac * sparsity_penalty * jnp.sum(abs(params.loadings))
    loss += 0.5 * (1 - elastic_net_frac) * sparsity_penalty * jnp.sum(params.loadings ** 2)
    return loss / data.size


def backtracking_line_search(data, 
                             params, 
                             new_params, 
                             mean_func, 
                             sparsity_penalty,
                             elastic_net_frac,
                             alpha=0.5, 
                             beta=0.9):
    mean_func = dict(softplus=softplus)[mean_func]
    
    def smooth_loss(params):
        activations = compute_activations(params)
        return -tfd.Poisson(rate=mean_func(activations) + 1e-8).log_prob(data).sum()
        
    def penalty(params):
        loss = elastic_net_frac * sparsity_penalty * jnp.sum(abs(params.loadings))
        loss += 0.5 * (1 - elastic_net_frac) * sparsity_penalty * jnp.sum(params.loadings ** 2)
        return loss
    
    dg = grad(smooth_loss)(params)
    descent_direction = tree_add(new_params, params, -1.0)

    def cond_fun(stepsize):
        new_params = tree_add(params, descent_direction, stepsize)
        new_loss = smooth_loss(new_params) + penalty(new_params)
        bound = smooth_loss(params) + penalty(params)
        bound += alpha * stepsize * tree_dot(dg, descent_direction)
        bound += alpha * (penalty(new_params) - penalty(params))
        return new_loss > bound
    
    def body_fun(stepsize):
        return beta * stepsize
    
    stepsize = lax.while_loop(cond_fun, body_fun, 1.0)
    return tree_add(params, descent_direction, stepsize)
    

def compute_quadratic_approx(data, params, mean_func):
    # Define key functions of the Poisson GLM
    A = jnp.exp                             # shorthand
    d2A = vmap(vmap(hessian(A)))            # want to broadcast scalar function to whole matrix

    if mean_func.lower() == "softplus":
        f = softplus

        # Define numerically safe versions of log(softplus) and its gradients
        log_softplus = lambda a: jnp.log(f(a))
        thresh = -10
        g = lambda a: jnp.where(a > thresh, log_softplus(a), a)
        dg = lambda a: jnp.where(a > thresh, vmap(vmap(grad(log_softplus)))(a), 1.0)
        d2g = lambda a: jnp.where(a > thresh, vmap(vmap(hessian(log_softplus)))(a), 0.0)
    else:
        raise Exception("invalid mean function: {}".format(mean_func))

    # Compute the quadratic approximation
    activations = compute_activations(params)
    predictions = f(activations)
    weights = d2g(activations) * (predictions - data) + (dg(activations))**2 * d2A(g(activations))
    # working_data = activations + dg(activations) / weights * (data - predictions)
    # residual = working_data - activations
    weighted_residual = dg(activations) * (data - predictions)
    return weighted_residual, weights
    

def update_loadings(weighted_residual, 
                    weights, 
                    params, 
                    sparsity_penalty, 
                    elastic_net_frac):
    """
    Update the loadings while holding the remaining parameters fixed.
    """
    def _update_one_loading(weighted_residual_m, weights_m, loading_m):
        """
        Coordinate descent to update the m-th loading
        """
        def _update_one_coord(weighted_residual_m, args):
            """
            Update one coordinate of the m-th loading
            """
            loading_mk, factor_k = args
            
            # Compute the numerator (linear term) and denominator (quad term)
            # of the quadratic loss as a function of loading \beta_{mk}
            num = jnp.einsum('n,n->', factor_k, (weighted_residual_m + weights_m * loading_mk * factor_k))
            den = jnp.einsum('n,n,n->', weights_m, factor_k, factor_k) + (1 - elastic_net_frac) * sparsity_penalty

            # Apply prox operator
            new_loading_mk = soft_threshold(num, elastic_net_frac * sparsity_penalty) / (den + 1e-8)
            
            # Update residual (y - \sum_i w_i x_i) with the new w_i
            weighted_residual_m += weights_m * loading_mk * factor_k
            weighted_residual_m -= weights_m * new_loading_mk * factor_k
            return weighted_residual_m, new_loading_mk

        # Scan over the (K,) dimension
        weighted_residual_m, loading_m = lax.scan(_update_one_coord, weighted_residual_m, (loading_m, params.factors))
        return weighted_residual_m, loading_m

    # Map over the (M,) dimension
    weighted_residual, loadings = vmap(_update_one_loading)(weighted_residual, weights, params.loadings) 
    params = dataclasses.replace(params, loadings=loadings)
    return weighted_residual, params
    

def update_factors(weighted_residual, weights, params):
    """
    Update the factors while holding the remaining parameters fixed.
    """
    def _update_one_column(weighted_residual_n, weights_n, factor_n):
        """
        Coordinate descent to update the n-th column of all factors
        """
        def _update_one_coord(weighted_residual_n, args):
            r"""
            Update factor \theta_{nk}
            """
            factor_nk, loading_k = args
            
            # Compute the numerator (linear term) and denominator (quad term)
            # of the quadratic loss as a function of factor \theta_{nk}
            num = jnp.einsum('m,m->', loading_k, (weighted_residual_n + weights_n * factor_nk * loading_k))
            den = jnp.einsum('m,m,m->', weights_n, loading_k, loading_k)

            # Apply prox operator to ensure loadings are non-negative
            new_factor_nk = jnp.maximum(num, 0.0) / (den + 1e-8)

            # Update residual (y - \sum_i w_i x_i) with the new w_i
            weighted_residual_n += weights_n * factor_nk * loading_k
            weighted_residual_n -= weights_n * new_factor_nk * loading_k

            return weighted_residual_n, new_factor_nk

        # Scan over the (K,) dimension
        weighted_residual_n, factor_n = lax.scan(_update_one_coord, weighted_residual_n, (factor_n, params.loadings.T))
        return weighted_residual_n, factor_n

    # Map over the (N,) dimension
    weighted_residualT, factorsT = vmap(_update_one_column)(weighted_residual.T, weights.T, params.factors.T)
    weighted_residual = weighted_residualT.T
    factors = factorsT.T

    # Make sure the factors are normalized
    scale = factors.sum(axis=1) + 1e-8
    factors /= scale[:, None]
    loadings = params.loadings * scale
    params = dataclasses.replace(params, factors=factors, loadings=loadings)
    return weighted_residual, params
    

def update_row_effect(weighted_residual, weights, params):
    """
    Update the row effect while holding the remaining parameters fixed.
    """
    def _update_one_row(weighted_residual_m, weights_m, row_effect_m):
        """
        Update the m-th row effect
        """
        # Compute the numerator (linear term) and denominator (quad term)
        # of the quadratic loss as a function of loading b_{m}
        num = jnp.einsum('n->', weighted_residual_m + weights_m * row_effect_m)
        den = jnp.einsum('n->', weights_m)
        new_row_effect_m = num / den

        # Update residual
        weighted_residual_m += weights_m * row_effect_m
        weighted_residual_m -= weights_m * new_row_effect_m
        return weighted_residual_m, new_row_effect_m

    # Map over the (M,) dimension
    weighted_residual, row_effects = vmap(_update_one_row)(weighted_residual, weights, params.row_effects) 
    params = dataclasses.replace(params, row_effects=row_effects)
    return weighted_residual, params


def update_column_effect(weighted_residual, weights, params):
    """
    Update the row effect while holding the remaining parameters fixed.
    """
    def _update_one_column(weighted_residual_n, weights_n, col_effect_n):
        """
        Update the n-th column effect
        """
        # Compute the numerator (linear term) and denominator (quad term)
        # of the quadratic loss as a function of loading c_{n}
        num = jnp.einsum('m->', weighted_residual_n + weights_n * col_effect_n)
        den = jnp.einsum('m->', weights_n)
        new_col_effect_n = num / den

        # Update residual
        weighted_residual_n += weights_n * col_effect_n
        weighted_residual_n -= weights_n * new_col_effect_n
        return weighted_residual_n, new_col_effect_n

    # Map over the (N,) dimension
    weighted_residualT, column_effects = vmap(_update_one_column)(weighted_residual.T, weights.T, params.column_effects)
    weighted_residual = weighted_residualT.T

    # Make sure column effects sum to zero
    mean = jnp.mean(column_effects)
    column_effects -= mean
    row_effects = params.row_effects + mean
    params = dataclasses.replace(params, row_effects=row_effects, column_effects=column_effects)
    return weighted_residual, params


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


def initialize_greedy(data, num_factors, mean_func):
    # Convert data to "targets" by inverting mean function
    if mean_func.lower() == "softplus":
        data = jnp.maximum(data, 1e-1)
        targets = data + jnp.log(1 - jnp.exp(-data))
    else:
        raise Exception("Invalid mean function: {}".format(mean_func))

    # Initialize the row and column effects
    row_effects = targets.mean(axis=1)
    residual = targets - row_effects[:, None]
    # col_effect = residual.mean(axis=0)
    col_effect = jnp.zeros(data.shape[1])
    residual -= col_effect
    

    # Greedily initialize loadings and factors
    loadings = []
    factors = []
    
    for k in range(num_factors):
        # Find the voxel with the highest residual variance
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
                        mean_func="softplus",
                        num_iters=10,
                        sparsity_penalty=1.0,
                        elastic_net_frac=0.0,
                        num_coord_ascent_iters=20,
                        tolerance=1e-1,
                        ):
    
    # Initialize 
    if initial_params is None:
        params = initialize_greedy(data, num_factors, mean_func)
    else:
        params = initial_params
    
    @jit
    def _step(params, _):
        """
        One sweep over parameter updates
        """
        # Update rows
        weighted_residual, weights = compute_quadratic_approx(data, params, mean_func)
        def _row_step(carry, _):
            weighted_residual, params = carry
            weighted_residual, params = update_loadings(weighted_residual, weights, params, sparsity_penalty, elastic_net_frac)
            weighted_residual, params = update_row_effect(weighted_residual, weights, params)
            return (weighted_residual, params), None
        (weighted_residual, new_params), _ = lax.scan(_row_step, (weighted_residual, params), None, length=num_coord_ascent_iters)
        # params = convex_combo(params, new_params, stepsize)
        params = backtracking_line_search(data, params, new_params, mean_func, sparsity_penalty, elastic_net_frac)

        # Update columns    
        weighted_residual, weights = compute_quadratic_approx(data, params, mean_func)
        def _column_step(carry, _):
            weighted_residual, params = carry
            weighted_residual, params = update_factors(weighted_residual, weights, params)
            # weighted_residual, params = update_column_effect(weighted_residual, weights, params)
            return (weighted_residual, params), None
        (_, new_params), _ = lax.scan(_column_step, (weighted_residual, params), None, length=num_coord_ascent_iters)
        # params = convex_combo(params, new_params, stepsize)
        params = backtracking_line_search(data, params, new_params, mean_func, sparsity_penalty, elastic_net_frac)

        loss = compute_loss(data, params, mean_func, sparsity_penalty, elastic_net_frac)
        return params, loss
        
    # Run coordinate ascent
    losses = [compute_loss(data, params, mean_func, sparsity_penalty, elastic_net_frac)]
    pbar = trange(num_iters)
    for itr in pbar:
        params, loss = _step(params, itr)
        losses.append(loss)
        assert jnp.isfinite(loss)
        pbar.set_description("loss: {:.4f}".format(losses[-1]))

        if abs(losses[-1] - losses[-2]) < tolerance:
            break

    return params, jnp.stack(losses)
