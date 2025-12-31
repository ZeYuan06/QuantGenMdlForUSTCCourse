#!/usr/bin/env python

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.mnist01_data import PCA8Model, StandardScaler
from src.mnist01_codec import LatentCodec, encode_latent_to_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mnist01", help="directory created by mnist01_prepare.py")
    ap.add_argument("--out", default=None, help="output dir (default: same as --data)")
    args = ap.parse_args()

    data_dir = args.data
    out_dir = args.out or data_dir
    os.makedirs(out_dir, exist_ok=True)

    mean = np.load(os.path.join(data_dir, "pca_mean_196.npy"))
    comps = np.load(os.path.join(data_dir, "pca_components_8x196.npy"))
    z_mean = np.load(os.path.join(data_dir, "latent_mean_8.npy"))
    z_std = np.load(os.path.join(data_dir, "latent_std_8.npy"))

    pca = PCA8Model(mean_=mean, components_=comps)
    scaler = StandardScaler(mean_=z_mean, std_=z_std)
    codec = LatentCodec(pca=pca, scaler=scaler)

    z_tr = np.load(os.path.join(data_dir, "latent_train_8.npy"))
    z_te = np.load(os.path.join(data_dir, "latent_test_8.npy"))

    s_tr = encode_latent_to_state(codec, z_tr)
    s_te = encode_latent_to_state(codec, z_te)

    np.save(os.path.join(out_dir, "qstates_train_n8.npy"), s_tr.astype(np.complex64))
    np.save(os.path.join(out_dir, "qstates_test_n8.npy"), s_te.astype(np.complex64))

    print("saved:")
    print(" ", os.path.join(out_dir, "qstates_train_n8.npy"), s_tr.shape)
    print(" ", os.path.join(out_dir, "qstates_test_n8.npy"), s_te.shape)


if __name__ == "__main__":
    main()
