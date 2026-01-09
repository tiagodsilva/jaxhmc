from abc import abstractmethod

# from functools import partial
import flax.nnx as nnx
import jax
import jax.numpy as jnp

# import jax.scipy as jsp


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
        self.sigma = 2 * sigma

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.sum((x - self.mu) ** 2 / self.sigma**2)


class GaussianMixturePotential(Potential):
    def __init__(self, means: jax.Array, sigma: float):
        super().__init__(means.shape[1])
        self.means = means
        self.sigma = sigma
        self.k = means.shape[0]

    def __call__(self, x: jax.Array) -> jax.Array:
        potential = (x - self.means) ** 2 / self.sigma**2
        potential = jnp.sum(potential, axis=1)
        potential = jax.nn.logsumexp(-potential / 2, axis=0)
        return -potential + jnp.log(self.k)
