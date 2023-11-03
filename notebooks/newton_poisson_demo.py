# %% 
import jax.numpy as jnp
import jax.random as jr
from jax.nn import softplus
from jax import grad, hessian, vmap, lax
from functools import partial

import matplotlib.pyplot as plt

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


import warnings
warnings.filterwarnings("ignore")

# %%
def sample_poisson_matrix_factorization(key, M, N, K, mean_func=softplus):
    k1, k2, k3, k4 = jr.split(key, 4)
    loadings = tfd.Normal(0, 1).sample(seed=k1, sample_shape=(M, K,))
    factors = tfd.Normal(0, 1).sample(seed=k2, sample_shape=(N, K))
    responses = tfd.Poisson(rate=mean_func(loadings @ factors.T)).sample(seed=k3)
    return loadings, factors, responses

M = 200
N = 10000
K = 10
true_loadings, true_factors, responses = sample_poisson_matrix_factorization(jr.PRNGKey(0), M, N, K)

print(true_loadings.shape)
print(true_factors.shape)
print(responses.shape)

# %%
# Start with the special case of a Poisson GLM
# forget about intercept ?
# mean_func = softplus

# def poisson_glm_nll(weights, covariates, responses):
#     """
#     weights:    (K,) array
#     covariates: (N, K) array
#     responses:  (N,) array
#     """
#     return -tfd.Poisson(rate=mean_func(covariates @ weights)).log_prob(responses).mean()

# def quad_approx(current_weights, covariates, responses):
#     """
#     return a function that 
#     """
#     x0 = current_weights
#     f = partial(poisson_glm_nll, covariates=covariates, responses=responses)
#     df = grad(f)
#     d2f = hessian(f)
#     return lambda x: f(x0) + (x - x0) @ df(x0) + 0.5 * (x - x0).T @ d2f(x0) @ (x - x0)


#     def sample_poisson_glm(key, N, K):
#     k1, k2, k3 = jr.split(key, 3)
#     weights = tfd.Normal(0, 1).sample(seed=k2, sample_shape=(K,))
#     covariates = tfd.Normal(0, 1).sample(seed=k1, sample_shape=(N, K))
#     responses = tfd.Poisson(rate=mean_func(covariates @ weights)).sample(seed=k2)
#     return weights, covariates, responses

# N = 200
# K = 10
# true_weights, covariates, responses = sample_poisson_glm(jr.PRNGKey(0), N, K)


# poisson_glm_nll(true_weights, covariates, responses)

# curr_weights = jnp.zeros(K)
# q = quad_approx(curr_weights, covariates, responses)
# q(true_weights)

# q0 = lambda x0: q(curr_weights.at[0].set(x0))
# dq0 = grad(q0)
# d2q0 = hessian(q0)
# x0_star = -dq0(curr_weights[0]) / d2q0(curr_weights[0])

# x0s = jnp.linspace(-2, 2, 100)
# plt.plot(x0s, vmap(q0)(x0s))
# plt.axvline(x0_star, color='r')

# def newton(initial_weights, 
#            covariates, 
#            responses,
#            num_steps=10, 
#            num_rounds_coord_desc=3):
#     """
#     Solve for the weights of a Poisson GLM using Newton's method. Perform the 
#     Newton step with coordinate descent. 

#     Parameters
#     ----------
#     initial_weights: (K,) 
#     covariates:      (N, K)
#     responses:       (N,)

#     Returns
#     -------
#     weights:         (K,)
#     """

#     K = initial_weights.size
#     f = partial(poisson_glm_nll, covariates=covariates, responses=responses)
#     df = grad(f)
#     d2f = hessian(f)

#     def _newton_step(x0, _):

#         # Make a quadratic approximation to the likelihood
#         q = lambda x: f(x0) + (x - x0) @ df(x0) + 0.5 * (x - x0).T @ d2f(x0) @ (x - x0)
        
#         # Define inner and outer coordinate descent steps
#         def _inner_coord_desc_step(x, k):
#             qk = lambda xk: q(x.at[k].set(xk))
#             dqk = grad(qk)
#             d2qk = hessian(qk)
            
#             # TODO: put prox in here
#             x = x.at[k].set(-dqk(curr_weights[k]) / d2qk(curr_weights[k]))
#             return x, None

#         def _outer_coord_descent_step(x, _):
#             # Loop over coordinates 0, ..., K-1
#             x, _ = lax.scan(_inner_coord_desc_step, x, jnp.arange(K))
#             return x, None

#         x, _ = lax.scan(_outer_coord_descent_step, x0, None, length=num_rounds_coord_desc)
#         return x, f(x)

#     return lax.scan(_newton_step, initial_weights, None, length=num_steps)
    

# # %%timeit
# x, losses = newton(jnp.zeros(K), covariates, responses)

# plt.plot(losses)
# plt.axhline(poisson_glm_nll(true_weights, covariates, responses), color='r')
# plt.xlabel("iteration")
# plt.ylabel("loss")

# print("true: ", true_weights)
# print("est:  ", x)


# # %%timeit
# est_weights, losses = newton_v2(jnp.zeros(K), covariates, responses)

# plt.plot(losses)

# plt.plot(true_weights, 'ko')
# plt.plot(est_weights, 'r.')

# %%
def newton_glm(params,
               covariates,
               responses,
               mean_func=softplus,
               log_normalizer=jnp.exp,
               num_steps=10,
               num_rounds_coord_desc=3):
    """
    Solve for the weights of a GLM using Newton's method. Perform the
    Newton step with coordinate descent.

    Parameters
    ----------
    params: (K,)
    covariates:      (N, K)
    responses:       (N,)

    Returns
    -------
    weights:         (K,)
    """

    N, K = covariates.shape
    assert params.shape == (K,)
    assert responses.shape == (N,)

    # Define key functions of the GLM
    A = log_normalizer                      # shorthand
    d2A = vmap(hessian(A))
    g = lambda a: jnp.log(mean_func(a))     # mapping to natural params
    dg = vmap(grad(g))
    d2g = vmap(hessian(g))
    loss = lambda b: -jnp.mean(responses * g(covariates @ b) - A(g(covariates @ b)))

    def _newton_step(params, _):

        # Make a quadratic approximation to the likelihood by computing the
        # weights and the working responses
        activations = covariates @ params
        preds = mean_func(activations)
        weights = d2g(activations) * (preds - responses) \
            + (dg(activations))**2 * d2A(g(activations))
        working_responses = activations + dg(activations) / weights * (responses - preds)
        residual = working_responses - activations

        # Define inner and outer coordinate descent steps
        def _inner_coord_desc_step(residual, args):
            bk, xk = args
            # Compute the numerator (linear term) and denominator (quad term)
            # of the quadratic loss as a function of bk
            num = jnp.einsum('n,n,n->', weights, xk, (residual + bk * xk))
            den = jnp.einsum('n,n,n->', weights, xk, xk)

            # TODO: Apply prox operator
            new_bk = num / den

            # Update residual (y - \sum_i w_i x_i) with the new w_i
            residual += bk * xk
            residual -= new_bk * xk

            return residual, new_bk

        def _outer_coord_descent_step(carry, _):
            residual, params = carry
            residual, params = lax.scan(_inner_coord_desc_step, residual, (params, covariates.T))
            return (residual, params), None

        # Run the outer loop of coordinate descent
        (_, params), _ = lax.scan(_outer_coord_descent_step,
                                  (residual, params),
                                  None,
                                  length=num_rounds_coord_desc)
        return params, loss(params)
        # return params, poisson_glm_nll(params, covariates, responses)

    return lax.scan(_newton_step, params, None, length=num_steps)


# %%
# Update all the loadings for fixed factors

# newton takes:
# initial_weights: (K,) 
# covariates:      (N, K)
# responses:       (N,)
loadings, losses = vmap(newton_glm, in_axes=(0, None, 0))(jnp.zeros((M, K)), true_factors, responses) 

# %%
plt.plot(losses.T)
plt.xlabel("iteration")
plt.ylabel("negative log like")
plt.show()


# %%

# Update all the factors for fixed loadings
# initial_weights: (K,) 
# covariates:      (N, K)
# responses:       (N,)
factors, losses = vmap(newton_glm, in_axes=(0, None, 0))(jnp.zeros((N, K)), true_loadings, responses.T)

assert jnp.all(jnp.isfinite(losses))
jnp.min(losses), jnp.max(losses)
# %%
