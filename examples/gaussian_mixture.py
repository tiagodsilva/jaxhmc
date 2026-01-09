import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxhmc.mcmc import HMCConfig, RandomWalkConfig, hmc, random_walk
from jaxhmc.potentials import GaussianMixturePotential

SAMPLES = 1_000_000

# We first sample a initial position
key = jax.random.key(42)
key, subkey = jax.random.split(key, 2)
initial_position = jax.random.uniform(subkey, (4, 2))

dim = initial_position.shape[1]

# We the run the HMC
means = jnp.vstack([jnp.ones(dim) + 2, jnp.ones(dim), jnp.ones(dim) - 2])

potential = GaussianMixturePotential(means=means, sigma=0.25)
config = HMCConfig(
    initial_step_size=0.2,
    max_path_len=2,
    iterations=SAMPLES,
    initial_precm=jnp.eye(dim),
    key=key,
)

momenta, samples_hmc = hmc(
    potential=potential,
    initial_position=initial_position,
    config=config,
)

warmup = min(5000, int(config.iterations * 0.1))
samples_hmc = samples_hmc[warmup:]
samples_hmc = samples_hmc.reshape(-1, 2)

config = RandomWalkConfig(
    initial_step_size=0.2,
    iterations=SAMPLES,
    key=key,
)

samples_rw = random_walk(
    potential=potential,
    initial_position=initial_position,
    config=config,
)

warmup = min(5000, int(config.iterations * 0.1))
samples_rw = samples_rw[warmup:]
samples_rw = samples_rw.reshape(-1, 2)

plt.figure(figsize=(12, 6))
# We then plot the samples
plt.subplot(1, 2, 1)
indices = jax.random.choice(key=key, a=samples_hmc.shape[0], shape=(100_000,), replace=True)
plt.scatter(samples_hmc[indices, 0], samples_hmc[indices, 1], alpha=0.6)
plt.scatter(means[:, 0], means[:, 1], color="red", marker="x", label="Mean")
plt.title("HMC Samples")
plt.legend()

plt.subplot(1, 2, 2)
indices = jax.random.choice(key=key, a=samples_rw.shape[0], shape=(100_000,), replace=True)
plt.scatter(samples_rw[indices, 0], samples_rw[indices, 1], alpha=0.6)
plt.scatter(means[:, 0], means[:, 1], color="red", marker="x", label="Mean")
plt.title("Random Walk Samples")

plt.savefig("examples/gaussian_mixture_samples.png")
plt.show()
