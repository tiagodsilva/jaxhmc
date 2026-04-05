from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import typer
from flax.typing import Axes

from jaxhmc.kitty_utils import show_kitty
from jaxhmc.mcmc import HMCConfig, hmc
from jaxhmc.potentials import BananaPotential

app = typer.Typer()


@partial(jax.vmap, in_axes=(0, None, None, None), out_axes=0)
def run_for_step_size(step_size, key, pot, batch_size):
    k1, k2 = jax.random.split(key)
    config = HMCConfig(
        initial_step_size=step_size,
        max_path_len=1,
        iterations=2000,
        initial_precm=jnp.eye(pot.dim),
        key=k1,
        should_tune=False,
        return_alphas=True,
    )

    _, _, alpha = hmc(
        pot, jax.random.uniform(k2, (batch_size, pot.dim)), config
    )

    return alpha.mean(axis=1)


@app.command()
def main(seed: int = 42, batch_size: int = 32):
    key = jax.random.key(seed)
    pot = BananaPotential()

    plt.figure(figsize=(18, 6))

    num = 10
    step_sizes = jnp.logspace(-3, 0, num=num, base=10)

    alphas = run_for_step_size(step_sizes, key, pot, batch_size)

    ax = plt.subplot(1, 2, 1)
    for i in range(num):
        ax.plot(alphas[i], label=f"{step_sizes[i]:.2e}")

    ax.set_ylabel("$\\alpha$")
    ax.set_xlabel("iterations")
    ax.legend(title="step sizes")

    ax = plt.subplot(1, 2, 2)
    ax.plot(step_sizes, alphas.mean(axis=1))
    ax.set_xscale("log")
    ax.set_xlabel("step size")
    ax.set_ylabel("$\\alpha$")

    plt.savefig("examples/step_size_analysis.png")

    show_kitty()


if __name__ == "__main__":
    app()
