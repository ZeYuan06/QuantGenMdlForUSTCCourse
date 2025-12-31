#!/usr/bin/env python

import argparse
import json
import os
import sys
from typing import Optional, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from src.mnist01_data import PCA8Model, StandardScaler
from src.mnist01_codec import LatentCodec, decode_state_to_latent


def _basis_signs_z(n: int) -> np.ndarray:
    """Return (2^n, n) array with entries in {+1,-1} for Z expectations."""
    basis = np.arange(2**n, dtype=np.int64)
    signs = np.empty((2**n, n), dtype=np.float64)
    for i in range(n):
        bit = (basis >> (n - 1 - i)) & 1
        signs[:, i] = 1.0 - 2.0 * bit  # 0->+1, 1->-1
    return signs


def _basis_signs_zz(n: int) -> np.ndarray:
    """Return (2^n, n*(n-1)/2) signs for ZZ expectations, ordered (0,1),(0,2)..."""
    z = _basis_signs_z(n)
    cols = []
    for i in range(n):
        for j in range(i + 1, n):
            cols.append((z[:, i] * z[:, j])[:, None])
    return np.concatenate(cols, axis=1) if cols else np.empty((2**n, 0), dtype=np.float64)


def _qstate_features_z_zz(q: np.ndarray, n: int) -> np.ndarray:
    """Map pure states (N,2^n) to real features (N, n + nC2) using Z and ZZ moments."""
    q = np.asarray(q)
    if q.ndim != 2 or q.shape[1] != 2**n:
        raise ValueError(f"qstates must be (N,{2**n}) complex; got {q.shape}")

    # probabilities over computational basis
    p = (np.abs(q) ** 2).astype(np.float64)

    signs_z = _basis_signs_z(n)
    z = p @ signs_z  # (N,n)

    signs_zz = _basis_signs_zz(n)
    zz = p @ signs_zz  # (N,nC2)

    feat = np.concatenate([z, zz], axis=1)
    return feat.astype(np.float32)


def _single_qubit_purity_mean(q: np.ndarray, n: int, maxN: int = 512) -> float:
    """Average single-qubit reduced-state purity over samples and qubits.

    For product states this is close to 1; for strongly entangled states it approaches 1/2.
    """
    q = np.asarray(q)
    if q.ndim != 2 or q.shape[1] != 2**n:
        raise ValueError(f"qstates must be (N,{2**n}) complex; got {q.shape}")

    q = q[: min(maxN, q.shape[0])]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)

    N = q.shape[0]
    psi_t = q.reshape((N,) + (2,) * n)
    pur = np.zeros((N, n), dtype=np.float64)

    for i in range(n):
        axes = (0, 1 + i) + tuple(j for j in range(1, n + 1) if j != 1 + i)
        t = np.transpose(psi_t, axes).reshape(N, 2, -1)
        rho = t @ np.conjugate(t).transpose(0, 2, 1)  # (N,2,2)
        pur[:, i] = np.real(np.einsum("nij,njk->nik", rho, rho).trace(axis1=1, axis2=2))

    return float(pur.mean())


def _try_load_qstates(gen_dir: str, n_qubits: int = 8) -> Optional[np.ndarray]:
    qst_path = os.path.join(gen_dir, f"qstates_gen_n{n_qubits}.npy")
    if not os.path.exists(qst_path):
        return None
    return np.load(qst_path).astype(np.complex64)


def _load_codec(data_dir: str) -> LatentCodec:
    mean = np.load(os.path.join(data_dir, "pca_mean_196.npy"))
    comps = np.load(os.path.join(data_dir, "pca_components_8x196.npy"))
    z_mean = np.load(os.path.join(data_dir, "latent_mean_8.npy"))
    z_std = np.load(os.path.join(data_dir, "latent_std_8.npy"))
    return LatentCodec(PCA8Model(mean, comps), StandardScaler(z_mean, z_std))


def _kid_polynomial_unbiased(x: np.ndarray, y: np.ndarray, degree: int = 3) -> float:
    """Unbiased KID using polynomial kernel k(a,b)=(a·b/d + 1)^degree."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = x.shape
    m, _ = y.shape

    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    def k(a, b):
        return (a @ b.T / d + 1.0) ** degree

    k_xx = k(x, x)
    k_yy = k(y, y)
    k_xy = k(x, y)

    # remove diagonals for unbiased estimate
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)

    mmd = k_xx.sum() / (n * (n - 1)) + k_yy.sum() / (m * (m - 1)) - 2.0 * k_xy.mean()
    return float(mmd)


def _mmd_rbf(x: np.ndarray, y: np.ndarray, sigma: Optional[float] = None) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # median heuristic on a subsample
    if sigma is None:
        rng = np.random.default_rng(0)
        xs = x[rng.choice(x.shape[0], size=min(512, x.shape[0]), replace=False)]
        ys = y[rng.choice(y.shape[0], size=min(512, y.shape[0]), replace=False)]
        d2 = np.sum((xs[:, None, :] - ys[None, :, :]) ** 2, axis=2)
        med = np.median(d2)
        sigma = np.sqrt(max(med, 1e-12))

    def k(a, b):
        d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        return np.exp(-d2 / (2.0 * sigma**2))

    k_xx = k(x, x)
    k_yy = k(y, y)
    k_xy = k(x, y)

    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)

    n = x.shape[0]
    m = y.shape[0]
    mmd = k_xx.sum() / (n * (n - 1)) + k_yy.sum() / (m * (m - 1)) - 2.0 * k_xy.mean()
    return float(mmd)


class MNIST01CNN(torch.nn.Module):
    def __init__(self, feat_dim: int = 64):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.fc1 = torch.nn.Linear(32 * 3 * 3, feat_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(feat_dim, 2)

    def forward(self, x):
        h = self.conv(x)
        h = h.reshape(h.shape[0], -1)
        feat = self.relu(self.fc1(h))
        logits = self.fc2(feat)
        return logits

    @torch.no_grad()
    def features(self, x):
        h = self.conv(x)
        h = h.reshape(h.shape[0], -1)
        feat = self.relu(self.fc1(h))
        return feat


def _load_classifier(ckpt_path: str, device: torch.device) -> MNIST01CNN:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = MNIST01CNN().to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def _infer_images(
    codec: LatentCodec,
    gen_dir: str,
    n_qubits: int = 8,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return images (N,14,14) float32 in [0,1], and optional latent (N,8)."""
    img_path = os.path.join(gen_dir, "images_gen_14x14.npy")
    lat_path = os.path.join(gen_dir, "latent_gen_8.npy")
    qst_path = os.path.join(gen_dir, "qstates_gen_n8.npy")

    if os.path.exists(img_path):
        imgs = np.load(img_path).astype(np.float32)
        lat = np.load(lat_path).astype(np.float32) if os.path.exists(lat_path) else None
        return imgs, lat

    if os.path.exists(lat_path):
        lat = np.load(lat_path).astype(np.float32)
        imgs = codec.latent_to_image(lat)
        return imgs.astype(np.float32), lat

    if os.path.exists(qst_path):
        q = np.load(qst_path).astype(np.complex64)
        lat = decode_state_to_latent(codec, q, n=n_qubits).astype(np.float32)
        imgs = codec.latent_to_image(lat)
        return imgs.astype(np.float32), lat

    raise FileNotFoundError(
        f"No generated outputs found in {gen_dir}. Expected one of: {img_path} / {lat_path} / {qst_path}"
    )


@torch.no_grad()
def _classifier_stats(model: MNIST01CNN, imgs: np.ndarray, device: torch.device):
    x = torch.from_numpy(imgs.astype(np.float32))[:, None, :, :].to(device)
    logits = model(x)
    prob = torch.softmax(logits, dim=1)
    conf, pred = torch.max(prob, dim=1)
    feat = model.features(x)

    pred = pred.cpu().numpy().astype(np.int64)
    conf = conf.cpu().numpy().astype(np.float32)
    feat = feat.cpu().numpy().astype(np.float32)

    frac1 = float((pred == 1).mean())
    return {
        "pred_frac_1": frac1,
        "conf_mean": float(conf.mean()),
        "conf_std": float(conf.std()),
        "features": feat,
        "pred": pred,
        "conf": conf,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mnist01")
    ap.add_argument("--gen", required=True, help="generated output directory")
    ap.add_argument("--classifier", default="data/mnist01/classifier/mnist01_cnn.pt")
    ap.add_argument("--out", default=None, help="output json path")
    ap.add_argument("--n_eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0, help="seed for selecting eval subsets")
    args = ap.parse_args()

    codec = _load_codec(args.data)

    x_te = np.load(os.path.join(args.data, "x_test_14x14.npy")).astype(np.float32)
    y_te = np.load(os.path.join(args.data, "y_test.npy")).astype(np.int64)

    # real test subset (randomized to avoid accidental ordering bias)
    rng = np.random.default_rng(args.seed)
    n_real = min(args.n_eval, x_te.shape[0])
    real_idx = rng.choice(x_te.shape[0], size=n_real, replace=False)
    x_real = x_te[real_idx]
    y_real = y_te[real_idx]

    # Load qstates if present; quantum-only metrics will prefer these.
    q_gen_all = _try_load_qstates(args.gen, n_qubits=8)
    q_real_path = os.path.join(args.data, "qstates_test_n8.npy")
    q_real_all = np.load(q_real_path).astype(np.complex64) if os.path.exists(q_real_path) else None

    # Image/latent metrics (may be lossy if only qstates exist).
    x_gen, z_gen = _infer_images(codec, args.gen, n_qubits=8)
    if x_gen.shape[0] > args.n_eval:
        gen_idx = rng.choice(x_gen.shape[0], size=args.n_eval, replace=False)
        x_gen = x_gen[gen_idx]
        if z_gen is not None:
            z_gen = z_gen[gen_idx]
    else:
        gen_idx = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_classifier(args.classifier, device)

    real_stats = _classifier_stats(model, x_real, device)
    gen_stats = _classifier_stats(model, x_gen, device)

    # classifier accuracy on real test subset (sanity)
    real_acc = float((real_stats["pred"] == y_real).mean())

    real_label_frac_1 = float((y_real == 1).mean())
    real_pred_frac_1 = float((real_stats["pred"] == 1).mean())

    # KID in feature space
    kid = _kid_polynomial_unbiased(real_stats["features"], gen_stats["features"], degree=3)

    # Optional: latent MMD if we have generated latent
    latent_mmd = None
    if z_gen is not None and os.path.exists(os.path.join(args.data, "latent_test_8.npy")):
        z_te_all = np.load(os.path.join(args.data, "latent_test_8.npy")).astype(np.float32)
        z_real = z_te_all[real_idx]
        latent_mmd = _mmd_rbf(z_real, z_gen)

    # Quantum-state metrics (primary when qstates are available).
    qstate_metrics = {
        "available": False,
        "natural_distance": None,
        "feature_mmd_rbf_z_zz": None,
        "single_qubit_purity_mean": None,
        "n_qubits": 8,
    }

    if q_gen_all is not None and q_real_all is not None:
        q_real = q_real_all[real_idx]
        q_gen = q_gen_all
        if gen_idx is not None and q_gen.shape[0] > args.n_eval:
            q_gen = q_gen[gen_idx]
        elif q_gen.shape[0] > args.n_eval:
            # In case we didn't subset images (e.g., exactly n_eval), still subset qstates.
            q_gen = q_gen[: args.n_eval]

        qstate_metrics["available"] = True

        # naturalDistance is defined on sets of state vectors; use JAX if available
        try:
            import jax
            from jax import numpy as jnp

            from src.distance_jax import naturalDistance

            qstate_metrics["natural_distance"] = float(
                naturalDistance(jnp.array(q_real, dtype=jnp.complex64), jnp.array(q_gen, dtype=jnp.complex64))
            )
        except Exception:
            qstate_metrics["natural_distance"] = None

        # Observable-feature MMD using Z and ZZ moments (quantum-only, no decoding).
        try:
            f_real = _qstate_features_z_zz(q_real, n=8)
            f_gen = _qstate_features_z_zz(q_gen, n=8)
            qstate_metrics["feature_mmd_rbf_z_zz"] = _mmd_rbf(f_real, f_gen)
        except Exception:
            qstate_metrics["feature_mmd_rbf_z_zz"] = None

        # Entanglement proxy: average single-qubit purity
        try:
            qstate_metrics["single_qubit_purity_mean"] = _single_qubit_purity_mean(q_gen, n=8, maxN=512)
        except Exception:
            qstate_metrics["single_qubit_purity_mean"] = None

    report = {
        "real_test_classifier_acc": real_acc,
        "real_label_frac_1": real_label_frac_1,
        "real_pred_frac_1": real_pred_frac_1,
        "generated_pred_frac_1": gen_stats["pred_frac_1"],
        "generated_conf_mean": gen_stats["conf_mean"],
        "generated_conf_std": gen_stats["conf_std"],
        "kid_feat": kid,
        "latent_mmd_rbf": latent_mmd,
        # Backward compatible top-level value (primary metric is now under qstate_metrics)
        "qstate_natural_distance": qstate_metrics["natural_distance"],
        "qstate_metrics": qstate_metrics,
        "n_real": int(x_real.shape[0]),
        "n_gen": int(x_gen.shape[0]),
        "gen_dir": args.gen,
        "seed": int(args.seed),
        "note": (
            "Primary metrics are quantum-state based (see qstate_metrics) when qstates are available. "
            "Classifier/KID/latent metrics operate on images (or decoded images if only qstates are provided). "
            "The qstates->image path is a lossy decoding that assumes product Ry states; for general entangled states, "
            "classifier-based metrics can look poor even when quantum-state metrics improve."
        ),
    }

    out_path = args.out or os.path.join(args.gen, "eval_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print("saved:", out_path)


if __name__ == "__main__":
    main()
