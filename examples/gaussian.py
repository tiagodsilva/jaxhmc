import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxhmc.mcmc import HMCConfig, hmc
from jaxhmc.potentials import GaussianPotential

# We first sample a initial position
key = jax.random.key(42)
key, subkey = jax.random.split(key, 2)
initial_position = jax.random.uniform(subkey, (16, 2))

# We the run the HMC
mu = jnp.ones(initial_position.shape[1]) + 2
potential = GaussianPotential(mu=mu)
config = HMCConfig(
    initial_step_size=0.1,
    max_path_len=2,
    warmup_steps=100,
    iterations=20000,
    initial_precm=jnp.eye(potential.dim),
    key=key,
)


# We try with and without JIT.

# We start without JIT.
#
s = time.monotonic_ns()
momenta, samples = hmc(
    potential=potential,
    initial_position=initial_position,
    config=config,
)

momenta.block_until_ready()
samples.block_until_ready()
e = time.monotonic_ns()

print("Time taken without JIT:", (e - s) / 1e9)


# We then JIT the sampler.
hmc_jit = jax.jit(hmc)

s = time.monotonic_ns()
momenta, samples = hmc_jit(
    potential=potential,
    initial_position=initial_position,
    config=config,
)

momenta.block_until_ready()
samples.block_until_ready()
e = time.monotonic_ns()
print("Time taken with JIT:", (e - s) / 1e9)  # ~ 30% speed up

warmup = 2000
samples = samples[warmup:]
samples = samples.reshape(-1, 2)
# We then plot the samples
plt.figure(figsize=(6, 6))
plt.scatter(samples[warmup:, 0], samples[warmup:, 1])
plt.scatter([mu[0]], [mu[1]], color="red", marker="x", label="Mean")
plt.legend()
plt.savefig("examples/gaussian_samples.png")
plt.show()
