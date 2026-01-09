from abc import abstractmethod

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class Potential(nnx.Module):
    def __init__(self, dim: int):
        self.dim = dim

    @abstractmethod
    def __call__(self, x: jax.Array) -> jax.Array:
        # This should return a scalar. It will be vmapped later.
        return


class GaussianPotential(Potential):
    def __init__(self, mu: jax.Array, sigma: float = 1.0):
        super().__init__(mu.shape[0])
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.sum((x - self.mu) ** 2 / self.sigma**2)
