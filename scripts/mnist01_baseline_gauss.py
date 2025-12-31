#!/usr/bin/env python

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.mnist01_data import PCA8Model, StandardScaler
from src.mnist01_codec import LatentCodec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mnist01")
    ap.add_argument("--out", default="data/mnist01/gen/baseline_gauss")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    mean = np.load(os.path.join(args.data, "pca_mean_196.npy"))
    comps = np.load(os.path.join(args.data, "pca_components_8x196.npy"))
    z_mean = np.load(os.path.join(args.data, "latent_mean_8.npy"))
    z_std = np.load(os.path.join(args.data, "latent_std_8.npy"))

    codec = LatentCodec(PCA8Model(mean, comps), StandardScaler(z_mean, z_std))

    rng = np.random.default_rng(args.seed)
    z = rng.normal(size=(args.n, 8)).astype(np.float32)
    # interpret z as standardized latent, inverse-scaling will be handled by codec.angles_to_latent only;
    # here we want latent in PCA space, so produce around training mean/std.
    z = z * z_std[None, :] + z_mean[None, :]

    imgs = codec.latent_to_image(z)
    np.save(os.path.join(args.out, "latent_gen_8.npy"), z.astype(np.float32))
    np.save(os.path.join(args.out, "images_gen_14x14.npy"), imgs.astype(np.float32))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
