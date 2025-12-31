#!/usr/bin/env python

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
from jax import numpy as jnp

from src.QGAN import HaarSampleGeneration, QGAN
from src.qstate_product_jax import project_to_product_ry, sample_ry_product_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help=".npz from mnist01_train_qgan.py")
    ap.add_argument("--out", default="data/mnist01/gen/qgan")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--na", type=int, default=0)
    ap.add_argument("--Lg", type=int, default=40)
    ap.add_argument("--Lc", type=int, default=12)
    ap.add_argument("--n_gen", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--noise",
        choices=["product_ry", "haar"],
        default="product_ry",
        help=(
            "Input noise distribution. For MNIST01 qstates (product Ry encoding), product_ry is recommended."
        ),
    )
    ap.add_argument(
        "--project_product_output",
        action="store_true",
        help=(
            "Project generated states to product Ry manifold before saving (via <Z_i> -> theta -> product state). "
            "This makes output compatible with MNIST01 encoding and improves purity toward 1."
        ),
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"qstates_gen_n{args.n}.npy")

    if (not args.force) and os.path.exists(out_path):
        print(f"[skip] exists: {out_path}")
        return

    ckpt = np.load(args.ckpt, allow_pickle=True)
    params_g = jnp.array(ckpt["params_g"], dtype=jnp.float32)

    model = QGAN(n=args.n, na=args.na, Lg=args.Lg, Lc=args.Lc)
    if args.noise == "haar":
        inputs = HaarSampleGeneration(args.n_gen, args.n, seed=args.seed)
        inputs = jnp.array(inputs, dtype=jnp.complex64)
    else:
        key_inputs = jax.random.PRNGKey(args.seed + 999)
        inputs, _ = sample_ry_product_states(key_inputs, n=args.n, batch=args.n_gen)

    key_meas = jax.random.PRNGKey(args.seed + 12345)
    states = model.dataGenerate(inputs, params_g, key_meas)
    if args.project_product_output:
        states, _ = project_to_product_ry(states, n=args.n)

    np.save(out_path, np.array(states).astype(np.complex64))
    print("saved:", out_path, states.shape)


if __name__ == "__main__":
    main()
