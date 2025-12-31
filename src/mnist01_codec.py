from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .mnist01_data import PCA8Model, StandardScaler


@dataclass(frozen=True)
class LatentCodec:
    pca: PCA8Model
    scaler: StandardScaler
    clip_u: float = 0.999999

    def image_to_latent(self, x14: np.ndarray) -> np.ndarray:
        """x14: (N,14,14) float in [0,1]. Returns (N,8) latent."""
        x = np.asarray(x14, dtype=np.float32)
        z = self.pca.transform(x.reshape(x.shape[0], -1))
        return z

    def latent_to_image(self, z: np.ndarray) -> np.ndarray:
        """z: (N,8) -> (N,14,14) float in [0,1] (clipped)."""
        x = self.pca.inverse_transform(np.asarray(z, dtype=np.float32))
        x = x.reshape(z.shape[0], 14, 14)
        return np.clip(x, 0.0, 1.0)

    def latent_to_angles(self, z: np.ndarray) -> np.ndarray:
        """Map latent to Ry angles in [0,pi] using tanh on standardized latent."""
        z = np.asarray(z, dtype=np.float32)
        zn = self.scaler.transform(z)
        u = np.tanh(zn)  # (-1,1)
        theta = (np.pi / 2.0) * (u + 1.0)
        return theta.astype(np.float32)

    def angles_to_latent(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float32)
        u = (2.0 * theta / np.pi) - 1.0
        u = np.clip(u, -self.clip_u, self.clip_u)
        zn = np.arctanh(u)
        return self.scaler.inverse_transform(zn.astype(np.float32))


def ry_product_state(angles: np.ndarray) -> np.ndarray:
    """Build product state vector for angles (n,) or (N,n). Returns (N,2^n) complex64."""
    angles = np.asarray(angles, dtype=np.float32)
    if angles.ndim == 1:
        angles = angles[None, :]
    n = angles.shape[1]

    c = np.cos(angles / 2.0).astype(np.float32)
    s = np.sin(angles / 2.0).astype(np.float32)

    out = np.empty((angles.shape[0], 2**n), dtype=np.complex64)
    for k in range(angles.shape[0]):
        v = np.array([1.0], dtype=np.float32)
        for i in range(n):
            v = np.kron(v, np.array([c[k, i], s[k, i]], dtype=np.float32))
        out[k] = v.astype(np.complex64)
    return out


def z_expectations_from_state(state: np.ndarray, n: int) -> np.ndarray:
    """Compute <Z_i> for each qubit i from a pure state vector.

    state: (2^n,) or (N,2^n) complex
    returns: (n,) or (N,n) float32 in [-1,1]
    """
    psi = np.asarray(state)
    if psi.ndim == 1:
        psi = psi[None, :]

    if psi.shape[1] != 2**n:
        raise ValueError(f"state dim mismatch: got {psi.shape[1]}, expected {2**n}")

    p = (np.abs(psi) ** 2).astype(np.float64)  # (N,2^n)

    # Precompute signs for each basis state and qubit
    basis = np.arange(2**n, dtype=np.int64)
    signs = np.empty((2**n, n), dtype=np.float64)
    for i in range(n):
        bit = (basis >> (n - 1 - i)) & 1
        signs[:, i] = 1.0 - 2.0 * bit  # 0->+1, 1->-1

    zexp = p @ signs  # (N,n)
    zexp = np.clip(zexp, -1.0, 1.0)
    return zexp.astype(np.float32) if state.ndim == 2 else zexp[0].astype(np.float32)


def angles_from_z_expectations(zexp: np.ndarray) -> np.ndarray:
    """Given <Z_i>, recover angles via arccos(<Z_i>) (approx exact for product Ry states)."""
    zexp = np.asarray(zexp, dtype=np.float32)
    zexp = np.clip(zexp, -1.0, 1.0)
    return np.arccos(zexp).astype(np.float32)


def decode_state_to_latent(codec: LatentCodec, state: np.ndarray, n: int = 8) -> np.ndarray:
    zexp = z_expectations_from_state(state, n=n)
    theta = angles_from_z_expectations(zexp)
    return codec.angles_to_latent(theta)


def encode_latent_to_state(codec: LatentCodec, z: np.ndarray) -> np.ndarray:
    theta = codec.latent_to_angles(z)
    return ry_product_state(theta)
