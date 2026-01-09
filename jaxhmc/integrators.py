from functools import partial

import flax.nnx as nnx
import jax
import jax.numpy as jnp

# Based on https://github.com/ColCarroll/minimc/blob/master/minimc/integrators.py


def _step(
    _,
    carry: tuple[jax.Array, jax.Array],
    precm: jax.Array,
    potential_grad: nnx.Module,
    step_size: float,
):
    p, q = carry  # (B, d)
    p = p - 0.5 * step_size * potential_grad(q)  # (B, d)
    q = q + step_size * jnp.einsum("ij, bi->bj", precm, p)  # (B, d)
    p = p - 0.5 * step_size * potential_grad(q)  # (B, d)
    return p, q


def leapfrog(
    p: jax.Array,
    q: jax.Array,
    precm: jax.Array,
    potential_grad: nnx.Module,
    step_size: float,
    steps: int,
):
    p, q = jax.lax.fori_loop(
        lower=0,
        upper=steps,
        body_fun=partial(
            _step,
            precm=precm,
            step_size=step_size,
            potential_grad=potential_grad,
        ),
        init_val=(p, q),
    )
    return p, q


# def leapfrog_2stage(p: jax.Array, q: jax.Array, step_size: float):
#     return
