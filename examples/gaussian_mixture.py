import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxhmc.eval import run
from jaxhmc.potentials import GaussianMixturePotential

real_dim = 2
fictious_dim = 0

# We the run the HMC
means = jnp.vstack(
    [
        jnp.ones(real_dim) + 2,
        jnp.ones(real_dim),
        jnp.ones(real_dim) - 2,
        jnp.array([-1, 3]),
        jnp.array([3, -1]),
    ],
)

potential = GaussianMixturePotential(means=means, sigma=0.01)
run(potential, dim=real_dim + fictious_dim, chain_length=100_000, include_hmc=False)

plt.savefig("examples/gaussian_mixture_samples.png")

plt.show()
