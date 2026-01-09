import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxhmc.mcmc import HMCConfig, hmc
from jaxhmc.potentials import GaussianMixturePotential

# We first sample a initial position
key = jax.random.key(42)
key, subkey = jax.random.split(key, 2)
initial_position = jax.random.uniform(subkey, (4, 2))

dim = initial_position.shape[1]

# We the run the HMC
means = jnp.vstack([jnp.ones(dim) + 2, jnp.ones(dim), jnp.ones(dim) - 2])

potential = GaussianMixturePotential(means=means, sigma=0.28)
config = HMCConfig(
    initial_step_size=0.2,
    max_path_len=2,
    iterations=100_000,
    initial_precm=jnp.eye(dim),
    key=key,
)


s = time.monotonic_ns()
momenta, samples = hmc(
    potential=potential,
    initial_position=initial_position,
    config=config,
)

e = time.monotonic_ns()
print("Time taken with JIT:", (e - s) / 1e9)

warmup = min(5000, int(config.iterations * 0.1))
samples = samples[warmup:]
samples = samples.reshape(-1, 2)
# We then plot the samples

plt.scatter(samples[warmup:, 0], samples[warmup:, 1], alpha=0.6)
plt.scatter(means[:, 0], means[:, 1], color="red", marker="x", label="Mean")
plt.legend()
plt.savefig("examples/gaussian_mixture_samples.png")
plt.show()
