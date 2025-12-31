import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class PCA8Model:
    mean_: np.ndarray            # (D,)
    components_: np.ndarray      # (8, D)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        xc = x - self.mean_[None, :]
        return xc @ self.components_.T

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float32)
        return z @ self.components_ + self.mean_[None, :]


@dataclass(frozen=True)
class StandardScaler:
    mean_: np.ndarray  # (d,)
    std_: np.ndarray   # (d,)

    def transform(self, z: np.ndarray) -> np.ndarray:
        return (z - self.mean_[None, :]) / self.std_[None, :]

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return z * self.std_[None, :] + self.mean_[None, :]


def downsample_28_to_14(x28: np.ndarray) -> np.ndarray:
    """2x2 average pooling from (N, 28, 28) -> (N, 14, 14)."""
    x28 = np.asarray(x28)
    if x28.ndim != 3 or x28.shape[1:] != (28, 28):
        raise ValueError(f"expected (N,28,28), got {x28.shape}")
    x = x28.reshape(x28.shape[0], 14, 2, 14, 2).mean(axis=(2, 4))
    return x.astype(np.float32)


def _try_load_mnist_torchvision(data_dir: str):
    try:
        import torch
        from torchvision.datasets import MNIST
        from torchvision import transforms
    except Exception as e:
        raise ImportError("torchvision not available") from e

    os.makedirs(data_dir, exist_ok=True)
    tfm = transforms.Compose([transforms.ToTensor()])

    tr = MNIST(root=data_dir, train=True, download=True, transform=tfm)
    te = MNIST(root=data_dir, train=False, download=True, transform=tfm)

    def _to_numpy(ds):
        xs = []
        ys = []
        for i in range(len(ds)):
            x, y = ds[i]
            xs.append((x.squeeze(0).numpy() * 255.0).astype(np.uint8))
            ys.append(int(y))
        return np.stack(xs), np.array(ys, dtype=np.int64)

    x_tr, y_tr = _to_numpy(tr)
    x_te, y_te = _to_numpy(te)
    return x_tr, y_tr, x_te, y_te


def _try_load_mnist_openml():
    try:
        from sklearn.datasets import fetch_openml
    except Exception as e:
        raise ImportError("scikit-learn not available") from e

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    x = mnist.data.reshape(-1, 28, 28).astype(np.uint8)
    y = mnist.target.astype(np.int64)

    # Standard split: first 60000 train, last 10000 test
    x_tr, y_tr = x[:60000], y[:60000]
    x_te, y_te = x[60000:], y[60000:]
    return x_tr, y_tr, x_te, y_te


def load_mnist_28x28(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST in (N,28,28) uint8. Tries torchvision first, then OpenML."""
    try:
        return _try_load_mnist_torchvision(data_dir)
    except Exception:
        return _try_load_mnist_openml()


def filter_binary_digits(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    digits=(0, 1),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d0, d1 = digits
    mtr = (y_tr == d0) | (y_tr == d1)
    mte = (y_te == d0) | (y_te == d1)

    x_tr2 = x_tr[mtr]
    y_tr2 = (y_tr[mtr] == d1).astype(np.int64)  # map {d0,d1}->{0,1}
    x_te2 = x_te[mte]
    y_te2 = (y_te[mte] == d1).astype(np.int64)
    return x_tr2, y_tr2, x_te2, y_te2


def fit_pca8(x: np.ndarray, k: int = 8) -> PCA8Model:
    """Fit PCA via SVD on centered data. x shape (N, D)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected (N,D), got {x.shape}")

    mean = x.mean(axis=0)
    xc = x - mean[None, :]

    # economical SVD
    # xc = U S Vt  => principal axes are rows of Vt
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    comps = vt[:k].astype(np.float32)
    return PCA8Model(mean_=mean.astype(np.float32), components_=comps)


def fit_standard_scaler(z: np.ndarray, eps: float = 1e-6) -> StandardScaler:
    z = np.asarray(z, dtype=np.float32)
    mean = z.mean(axis=0)
    std = z.std(axis=0)
    std = np.maximum(std, eps)
    return StandardScaler(mean_=mean.astype(np.float32), std_=std.astype(np.float32))
