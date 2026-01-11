import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxhmc.mcmc import HMCConfig, RandomWalkConfig, hmc, random_walk
from jaxhmc.potentials import Potential, plot_potential


def run_hmc(
    potential: Potential,
    q: jax.Array,
    key: jax.Array,
    chain_length: int,
    burnin: int,
    dim: int,
):
    config = HMCConfig(
        initial_step_size=0.2,
        max_path_len=1,
        iterations=chain_length,
        initial_precm=jnp.eye(dim),
        key=key,
    )

    start_time = time.monotonic_ns()
    _, samples_hmc = hmc(
        potential=potential,
        initial_position=q,
        config=config,
    )

    samples_hmc.block_until_ready()
    end_time = time.monotonic_ns()
    print(f"HMC took {(end_time - start_time) / 1e6:.2f} ms")

    samples_hmc = samples_hmc[burnin:]
    samples_hmc = samples_hmc.reshape(-1, 2)

    return samples_hmc


def run_rw(
    potential: Potential,
    q: jax.Array,
    key: jax.Array,
    chain_length: int,
    burnin: int,
    dim: int,
):
    config = RandomWalkConfig(
        initial_step_size=0.2,
        iterations=chain_length,
        key=key,
    )

    start_time = time.monotonic_ns()
    samples_rw = random_walk(
        potential=potential,
        initial_position=q,
        config=config,
    )

    samples_rw.block_until_ready()
    end_time = time.monotonic_ns()
    print(f"Random Walk took {(end_time - start_time) / 1e6:.2f} ms")

    samples_rw = samples_rw[burnin:]
    samples_rw = samples_rw.reshape(-1, 2)

    return samples_rw


def plot_samples_overlay(
    potential: Potential,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    samples: int,
    n_samples_to_plot: int,
    key: jax.Array,
    ax: plt.Axes,
    title: str,
    alpha: float = 0.6,
    res: int = 1000,
):
    indices = jax.random.choice(key=key, a=samples.shape[0], shape=(n_samples_to_plot,), replace=True)

    plot_potential(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        res=res,
        potential=potential,
        ax=ax,
        alpha=alpha,
    )

    plt.scatter(samples[indices, 0], samples[indices, 1], alpha=alpha, c="orange")
    plt.title(title)


def run(
    potential: int,
    dim: int = 2,
    chain_length: int = 100_000,
    burnin_prop: float = 0.2,
    plot_prop: float = 0.05,
    alpha: float = 0.6,
    seed: int = 42,
):
    burnin = int(chain_length * burnin_prop)
    n_samples_to_plot = int((chain_length - burnin) * plot_prop)

    # Create config for HMC
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key, 2)
    initial_position = jax.random.uniform(subkey, (16, dim))
    dim = initial_position.shape[1]

    samples_hmc = run_hmc(potential, initial_position, key, chain_length, burnin, dim)
    samples_rw = run_rw(potential, initial_position, key, chain_length, burnin, dim)

    xmin = samples_rw[:, 0].min()
    xmax = samples_rw[:, 0].max()
    ymin = samples_rw[:, 1].min()
    ymax = samples_rw[:, 1].max()

    plt.figure(figsize=(18, 6))

    ax = plt.subplot(1, 3, 1)
    # Plot samples for HMC
    plot_samples_overlay(
        potential=potential,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        samples=samples_hmc,
        n_samples_to_plot=n_samples_to_plot,
        key=key,
        ax=ax,
        title="HMC Samples",
        alpha=alpha,
    )
    plt.legend()

    ax = plt.subplot(1, 3, 2)
    # Plot samples for Random Walk

    plot_samples_overlay(
        potential=potential,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        samples=samples_rw,
        n_samples_to_plot=n_samples_to_plot,
        key=key,
        ax=ax,
        title="Random Walk Samples",
        alpha=alpha,
    )

    ax = plt.subplot(1, 3, 3)
    # Plot the density of the target distribution
    plot_potential(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        res=1000,
        potential=potential,
        ax=ax,
    )

    plt.tight_layout()
