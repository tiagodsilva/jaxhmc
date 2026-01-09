import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from jaxhmc.integrators import leapfrog


@pytest.fixture
def d():
    return 2


@pytest.fixture
def B():
    return 16


@pytest.fixture
def potential() -> nnx.Module:
    class Potential(nnx.Module):
        def __call__(self, p: jax.Array):
            # Gaussian distribution
            return jnp.sum(p**2)

    pfunc = Potential()
    return pfunc


@pytest.fixture
def precm(d: int) -> jax.Array:
    return jnp.eye(d)


@pytest.fixture
def p(B: int, d: int) -> jax.Array:
    return jnp.zeros((B, d))


@pytest.fixture
def q(B: int, d: int) -> jax.Array:
    return jnp.zeros((B, d))


def test_leapfrog_return_correct_shape(
    p: jax.Array,
    q: jax.Array,
    potential: nnx.Module,
    precm: jax.Array,
):
    potential_grad = jax.vmap(jax.grad(potential), in_axes=0)
    p_new, q_new = leapfrog(
        p,
        q,
        precm=precm,
        potential_grad=potential_grad,
        steps=20,
        step_size=1e-1,
    )
    assert isinstance(p_new, jax.Array)
    assert isinstance(q_new, jax.Array)
    assert p.shape == p_new.shape
    assert q.shape == q_new.shape


def test_leapfrog_p_does_not_change_when_potential_is_constant(
    p: jax.Array,
    q: jax.Array,
    precm: jax.Array,
):
    def potential(x):
        return jnp.ones_like(x).sum()

    def potential_grad(x):
        return jnp.zeros_like(x)

    steps = 20
    step_size = 1e-1

    p_new, q_new = leapfrog(
        p,
        q,
        precm=precm,
        potential_grad=potential_grad,
        steps=steps,
        step_size=step_size,
    )
    assert jnp.allclose(p, p_new)
    assert jnp.allclose(q_new, q + steps * p)
