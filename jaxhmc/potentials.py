from abc import abstractmethod

# from functools import partial
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

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


class BananaPotential(Potential):
    def __init__(self, dim: int = 2, b: float = 4):
        super().__init__(dim)
        self.dim = dim
        self.b = b

    def __call__(self, x: jax.Array) -> jax.Array:
        x_sq = x * x
        log_p = x_sq[0] + (x[1] - self.b * (x_sq[0] - 1)) ** 2
        return log_p


class RingsPotential(Potential):
    def __init__(self, dim: int = 2, radii: tuple[float, float] = (2.0, 4.0), sigma: float = 0.15):
        super().__init__(dim)
        self.dim = dim
        self.radii = jnp.array(radii)  # radii of the two rings
        self.sigma = sigma  # width of each ring

    def __call__(self, x: jax.Array) -> jax.Array:
        r = jnp.sqrt(jnp.sum(x**2))
        ring_potentials = (r - self.radii) ** 2 / (2 * self.sigma**2)
        return -jax.nn.logsumexp(-ring_potentials)


def plot_potential(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    res: float,
    potential: Potential,
    ax: plt.Axes,
    alpha: float = 1.0,
):
    pot_vmap = jax.vmap(jax.vmap(potential, 1, 0), 1, 0)

    x = jnp.linspace(xmin, xmax, endpoint=True, num=res)
    y = jnp.linspace(ymin, ymax, endpoint=True, num=res)

    xy = jnp.meshgrid(x, y)
    xy = jnp.stack(xy)

    log_p = -pot_vmap(xy)

    # As we simply want the shape, we normalize log_p
    log_p = log_p - jax.nn.logsumexp(log_p, axis=(0, 1))
    p = jnp.exp(log_p)

    ax.contourf(xy[0], xy[1], p, alpha=alpha)

    return p
