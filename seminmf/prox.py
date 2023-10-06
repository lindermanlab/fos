import jax.numpy as jnp

from functools import partial
from jax import jit, grad, lax
from scipy.optimize import bisect
from tqdm.auto import trange


def prox_grad_descent_python(objective,
                             prox,
                             init_params,
                             max_num_steps=100,
                             max_stepsize=1.0,
                             discount=0.9,
                             tol=1e-6,
                             verbosity=0):
    r""" Run proximal gradient descent on an objective using the given
    proximal operator. This is a non-jittable version that uses python
    for and while loops. See below for a jittable JAX version.

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


def prox_grad_descent(objective,
                      prox,
                      init_params,
                      max_num_steps=100,
                      max_num_backtrack_steps=100,
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

    # Run proximal gradient descent until convergence
    def _step_cond(state):
        _, old_obj, curr_obj, counter = state
        return (abs(curr_obj - old_obj) > tol) & (counter <= max_num_steps)
        
    def _step_body(state):
        params, _, curr_obj, counter = state

        # Define the condition and body of a while loop for backtracking line search
        def _backtrack_cond(backtrack_state):
            stepsize, backtrack_counter = backtrack_state

            # Compute generalized gradient
            G = (params - prox(params - stepsize * dg(params), stepsize)) / stepsize

            # Update params and evaluate objective
            new_params = params - stepsize * G
            new_obj = g(new_params)

            # Compute upper bound on objective via quadratic Taylor approx
            lower_bound = g(params) \
                - stepsize * jnp.sum(dg(params) * G) \
                    + 0.5 * stepsize * jnp.sum(G**2)

            # Continue to decrease stepsize while objective exceeds lower bound
            # return (new_obj > lower_bound) & (backtrack_counter < max_num_steps)
            return (new_obj > lower_bound) & (backtrack_counter < max_num_backtrack_steps)
        def _backtrack_body(backtrack_state):
            stepsize, backtrack_counter = backtrack_state
            return stepsize * discount, backtrack_counter + 1

        # Run backtracking line search to find stepsize
        stepsize, _ = lax.while_loop(_backtrack_cond, _backtrack_body, (max_stepsize, 0))
                
        # Perform update with this stepsize
        G = (params - prox(params - stepsize * dg(params), stepsize)) / stepsize
        new_params = params - stepsize * G
        new_obj = g(new_params)
        return new_params, curr_obj, new_obj, counter + 1

    params, _, _, _ = lax.while_loop(_step_cond, _step_body, (init_params, jnp.inf, 0.0, 0))
    return params


@jit
def _simplex_lagrangian(w, lmbda):
    return jnp.sum(jnp.clip(w - lmbda, a_min=0.0)) - 1.0


def bisect_jax(f, a, b, tol=1e-8, max_iter=50):
    """Find a root of f via bisection.

    Args:
        f: a function that is monotonically increasing or decreasing on the interval [a, b]
        a: lower bound on root
        b: upper bound on root
    """
    def _bisect_cond(state):
        (a, f_a), (b, f_b), counter = state
        m = (a + b) / 2
        return (abs(f(m)) > tol) & ((b - a) / 2 > tol) & (counter < max_iter)

    def _bisect_body(state):
        (a, f_a), (b, f_b), counter = state
        m = (a + b) / 2
        f_m = f(m)
        # if sign(f(m)) = sign(f(m)) then a <- m else b <- m // new interval
        return lax.cond(jnp.sign(f_m) == jnp.sign(f_a),
                        lambda: ((m, f_m), (b, f_b), counter+1), # predicate true
                        lambda: ((a, f_a), (m, f_m), counter+1), # predicate false
                        )

    # breakpoint()
    init_state = ((a, f(a)), (b, f(b)), 0)
    ((a, _), (_, _), _) = lax.while_loop(_bisect_cond, _bisect_body, init_state)
    return a


def project_simplex(w):
    """Project onto simplex following the approach in Ch 6.2.5 of
    https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
    """
    lmbda_max = jnp.max(w)
    lmbda_star = bisect_jax(partial(_simplex_lagrangian, w),
                            lmbda_max - 1, lmbda_max)
    return jnp.clip(w - lmbda_star, a_min=0.0)


def soft_threshold(x, thresh):
    return x/jnp.abs(x) * jnp.maximum(jnp.abs(x) - thresh, 0.0)
