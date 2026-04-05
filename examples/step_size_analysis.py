import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import typer

from jaxhmc.kitty_utils import show_kitty
from jaxhmc.mcmc import HMCConfig, hmc
from jaxhmc.potentials import BananaPotential

app = typer.Typer()


def run_for_step_size(
    step_size: float,
    key: jax.Array,
    pot: BananaPotential,
    batch_size: int,
    iterations: int,
):
    k1, k2 = jax.random.split(key)

    config = HMCConfig(
        initial_step_size=step_size,
        max_path_len=1,
        iterations=iterations,
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
def main(seed: int = 42, batch_size: int = 32, iterations: int = 2000):
    key = jax.random.key(seed)
    pot = BananaPotential()

    plt.figure(figsize=(18, 6))

    num = 10
    step_sizes = jnp.logspace(-3, 0, num=num, base=10)

    alphas = jax.vmap(run_for_step_size, in_axes=(0, None, None, None, None))(
        step_sizes, key, pot, batch_size, iterations
    )

    # Compute elapsed time per iteration
    iterations_for_elapsed = 100
    elapsed_per_step = {}
    for i in range(num):
        start = time.perf_counter()
        run_for_step_size(
            step_sizes[i], key, pot, batch_size, iterations_for_elapsed
        )
        elapsed = time.perf_counter() - start
        elapsed_per_step[float(step_sizes[i])] = (
            elapsed / iterations_for_elapsed
        )

    ax = plt.subplot(1, 3, 1)
    for i in range(num):
        ax.plot(alphas[i], label=f"{step_sizes[i]:.2e}")

    ax.set_ylabel("$\\alpha$")
    ax.set_xlabel("iterations")
    ax.legend(title="step sizes")

    ax = plt.subplot(1, 3, 2)
    ax.plot(step_sizes, alphas.mean(axis=1))
    ax.set_xscale("log")
    ax.set_xlabel("step size")
    ax.set_ylabel("$\\alpha$")

    ax = plt.subplot(1, 3, 3)
    ax.plot(step_sizes, [elapsed_per_step[float(s)] for s in step_sizes])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step size")
    ax.set_ylabel("elapsed time (s)")

    plt.savefig("examples/step_size_analysis.png")

    show_kitty()


if __name__ == "__main__":
    app()
