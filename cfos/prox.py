import jax.numpy as jnp

from functools import partial
from jax import jit, grad
from scipy.optimize import bisect
from tqdm.auto import trange


def prox_grad_descent(objective,
                      prox,
                      init_params,
                      max_num_steps=100,
                      max_stepsize=1.0,
                      discount=0.9,
                      tol=1e-6,
                      verbosity=0):
    r""" Run proximal gradient descent on an objective using the given
    proximal operator.

    prox: params x stepsize -> new_params
    """
    # We need the gradient of the objective
    g = jit(objective)
    dg = jit(grad(objective))

    # Run projected gradient descent with backtracking line search
    params = init_params
    curr_obj = jnp.inf
    pbar = trange(max_num_steps) if verbosity > 0 else range(max_num_steps)
    for _ in pbar:
        # Backtracking line search
        stepsize = max_stepsize
        while True:
            # Compute generalized gradient
            G = (params - prox(params - stepsize * dg(params), stepsize)) / stepsize

            # Update params and evaluate objective
            new_params = params - stepsize * G
            new_obj = g(new_params)

            # Compute upper bound on objective via quadratic Taylor approx
            bound  = g(params) \
                    - stepsize * jnp.sum(dg(params) * G) \
                    + 0.5 * stepsize * jnp.sum(G**2)

            # If the new objective > bound, shrink stepsize. Otherwise break.
            if new_obj > bound:
                if verbosity > 0: print("shrinking!")
                stepsize = stepsize * discount
            else:
                break

        # Update params once we've found an acceptable step size
        params = new_params

        if verbosity > 0:
            print("final stepsize: ", stepsize)

        # Check
        if abs(new_obj - curr_obj) < tol:
            break
        else:
            curr_obj = new_obj

    return params


@jit
def _simplex_lagrangian(w, lmbda):
    return jnp.sum(jnp.clip(w - lmbda, a_min=0.0)) - 1.0


def project_simplex(w):
    """Project onto simplex following the approach in Ch 6.2.5 of
    https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
    """
    lmbda_max = jnp.max(w)
    lmbda_star = bisect(partial(_simplex_lagrangian, w),
                        lmbda_max - 1, lmbda_max)
    return jnp.clip(w - lmbda_star, a_min=0.0)


def soft_threshold(x, thresh):
    return x/jnp.abs(x) * jnp.maximum(jnp.abs(x) - thresh, 0)
