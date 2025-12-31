#!/usr/bin/env python

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.mnist01_data import (
    downsample_28_to_14,
    filter_binary_digits,
    fit_pca8,
    fit_standard_scaler,
    load_mnist_28x28,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mnist01", help="output directory")
    ap.add_argument("--digits", default="0,1", help="two digits, e.g. 0,1")
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    d0, d1 = [int(x.strip()) for x in args.digits.split(",")]

    x_tr, y_tr, x_te, y_te = load_mnist_28x28(os.path.join(args.out, "_raw"))
    x_tr, y_tr, x_te, y_te = filter_binary_digits(x_tr, y_tr, x_te, y_te, digits=(d0, d1))

    # subsample for speed / consistent experiment budget
    rng = np.random.default_rng(args.seed)
    tr_idx = rng.choice(x_tr.shape[0], size=min(args.n_train, x_tr.shape[0]), replace=False)
    te_idx = rng.choice(x_te.shape[0], size=min(args.n_test, x_te.shape[0]), replace=False)

    x_tr = x_tr[tr_idx]
    y_tr = y_tr[tr_idx]
    x_te = x_te[te_idx]
    y_te = y_te[te_idx]

    # normalize to [0,1]
    x_tr14 = downsample_28_to_14(x_tr.astype(np.float32) / 255.0)
    x_te14 = downsample_28_to_14(x_te.astype(np.float32) / 255.0)

    x_tr_flat = x_tr14.reshape(x_tr14.shape[0], -1)
    pca = fit_pca8(x_tr_flat, k=8)

    z_tr = pca.transform(x_tr_flat)
    z_te = pca.transform(x_te14.reshape(x_te14.shape[0], -1))

    scaler = fit_standard_scaler(z_tr)

    # Save images + labels
    np.save(os.path.join(args.out, "x_train_14x14.npy"), x_tr14.astype(np.float32))
    np.save(os.path.join(args.out, "y_train.npy"), y_tr.astype(np.int64))
    np.save(os.path.join(args.out, "x_test_14x14.npy"), x_te14.astype(np.float32))
    np.save(os.path.join(args.out, "y_test.npy"), y_te.astype(np.int64))

    # Save PCA + scaler
    np.save(os.path.join(args.out, "pca_mean_196.npy"), pca.mean_.astype(np.float32))
    np.save(os.path.join(args.out, "pca_components_8x196.npy"), pca.components_.astype(np.float32))
    np.save(os.path.join(args.out, "latent_train_8.npy"), z_tr.astype(np.float32))
    np.save(os.path.join(args.out, "latent_test_8.npy"), z_te.astype(np.float32))
    np.save(os.path.join(args.out, "latent_mean_8.npy"), scaler.mean_.astype(np.float32))
    np.save(os.path.join(args.out, "latent_std_8.npy"), scaler.std_.astype(np.float32))

    print(f"saved to {args.out}")
    print("train:", x_tr14.shape, z_tr.shape, "test:", x_te14.shape, z_te.shape)


if __name__ == "__main__":
    main()
