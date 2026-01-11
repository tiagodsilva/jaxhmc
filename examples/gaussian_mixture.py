import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxhmc.eval import run
from jaxhmc.potentials import GaussianMixturePotential

dim = 2

# We the run the HMC
means = jnp.vstack(
    [
        jnp.ones(dim) + 2,
        jnp.ones(dim),
        jnp.ones(dim) - 2,
        jnp.array([-1, 3]),
        jnp.array([3, -1]),
    ],
)

potential = GaussianMixturePotential(means=means, sigma=0.25)
run(potential)

plt.savefig("examples/gaussian_mixture_samples.png")

plt.show()
