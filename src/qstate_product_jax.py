import functools
from typing import Tuple

import jax
from jax import numpy as jnp
import numpy as np


@functools.lru_cache(maxsize=None)
def _basis_bits(n: int) -> jnp.ndarray:
    """Return (2^n, n) bits for computational basis states.

    NOTE: This function is used inside jax.jit-ed code paths.
    Do NOT construct JAX arrays here (it can leak tracers via lru_cache).
    We precompute via NumPy and convert to JAX arrays at the call site.

    bits[k, i] is 0/1 for qubit i (i=0 is MSB, matches other code in this repo).
    """
    basis = np.arange(2**n, dtype=np.int32)[:, None]
    shifts = np.arange(n - 1, -1, -1, dtype=np.int32)[None, :]
    bits = (basis >> shifts) & 1
    return bits.astype(np.int32)


@functools.lru_cache(maxsize=None)
def _basis_signs_z(n: int) -> jnp.ndarray:
    """Return (2^n, n) array with entries in {+1,-1} for Z expectations.

    Precomputed via NumPy for the same reason as _basis_bits.
    """
    bits = _basis_bits(n).astype(np.int32)
    return (1.0 - 2.0 * bits.astype(np.float32)).astype(np.float32)


def ry_product_state_from_angles(theta: jnp.ndarray) -> jnp.ndarray:
    """Build product Ry state vectors from angles.

    Args:
        theta: (N,n) angles in radians.

    Returns:
        psi: (N,2^n) complex64. For each qubit i: |psi_i> = cos(theta/2)|0> + sin(theta/2)|1>

    Notes:
        This matches the numpy implementation in src/mnist01_codec.py, but is JAX-friendly.
    """
    theta = jnp.asarray(theta, dtype=jnp.float32)
    if theta.ndim != 2:
        raise ValueError(f"theta must be (N,n); got {theta.shape}")

    n = int(theta.shape[1])
    bits = jnp.asarray(_basis_bits(n), dtype=jnp.int32)  # (2^n,n)

    c = jnp.cos(theta / 2.0)  # (N,n)
    s = jnp.sin(theta / 2.0)  # (N,n)

    # amplitude[b] = prod_i (c_i if bit=0 else s_i)
    bits_f = bits[None, :, :].astype(jnp.float32)  # (1,2^n,n)
    c_term = c[:, None, :] ** (1.0 - bits_f)
    s_term = s[:, None, :] ** bits_f
    amp = jnp.prod(c_term * s_term, axis=2)  # (N,2^n)

    return amp.astype(jnp.complex64)


def z_expectations_from_state(state: jnp.ndarray, n: int) -> jnp.ndarray:
    """Compute <Z_i> for each qubit from pure state vectors.

    Args:
        state: (N,2^n) complex
    Returns:
        zexp: (N,n) float32 in [-1,1]
    """
    state = jnp.asarray(state)
    if state.ndim != 2 or state.shape[1] != 2**n:
        raise ValueError(f"state must be (N,{2**n}) complex; got {state.shape}")

    p = jnp.abs(state) ** 2.0  # (N,2^n)
    signs = jnp.asarray(_basis_signs_z(n), dtype=jnp.float32)  # (2^n,n)
    zexp = p @ signs
    return jnp.clip(zexp, -1.0, 1.0).astype(jnp.float32)


def project_to_product_ry(state: jnp.ndarray, n: int, eps: float = 1e-7) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Project arbitrary states to product Ry manifold using only <Z_i>.

    This matches the lossy decoding assumption used elsewhere in this repo:
    theta_i = arccos(<Z_i>), then |psi> = \otimes_i (cos(theta_i/2)|0> + sin(theta_i/2)|1>).

    Returns (projected_states, theta).
    """
    zexp = z_expectations_from_state(state, n=n)
    zexp = jnp.clip(zexp, -1.0 + eps, 1.0 - eps)
    theta = jnp.arccos(zexp).astype(jnp.float32)
    proj = ry_product_state_from_angles(theta)
    return proj, theta


def sample_ry_product_states(
    key: jax.Array,
    n: int,
    batch: int,
    theta_min: float = 0.0,
    theta_max: float = float(jnp.pi),
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample product Ry states with per-qubit angles uniform in [theta_min, theta_max].

    Returns (states, theta).
    """
    key = jax.random.PRNGKey(int(key[0])) if getattr(key, "shape", None) == (2,) else key
    theta = jax.random.uniform(
        key,
        shape=(batch, n),
        minval=jnp.asarray(theta_min, dtype=jnp.float32),
        maxval=jnp.asarray(theta_max, dtype=jnp.float32),
        dtype=jnp.float32,
    )
    states = ry_product_state_from_angles(theta)
    return states, theta
