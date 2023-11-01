import dataclasses
import jax
import jax.numpy as jnp
import warnings

from jax import tree_map
from jax.tree_util import tree_reduce
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
warnings.filterwarnings("ignore")


# Helper function to make a dataclass a JAX PyTree
def register_pytree_node_dataclass(cls):
  _flatten = lambda obj: jax.tree_flatten(dataclasses.asdict(obj))
  _unflatten = lambda d, children: cls(**d.unflatten(children))
  jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)
  return cls


def convex_combo(pytree1, pytree2, stepsize):
    f = lambda x, y: (1 - stepsize) * x + stepsize * y
    return tree_map(f, pytree1, pytree2)


def tree_add(pytree1, pytree2, scale=1.0):
    return tree_map(lambda x, y: x + scale * y, pytree1, pytree2)


def tree_dot(pytree1, pytree2):
    return tree_reduce(jnp.add,
                       tree_map(lambda x, y: jnp.sum(x * y), pytree1, pytree2),
                       0.0)
