#!/usr/bin/env python

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from jax import numpy as jnp
from jax import random

from src.QDT_jax import HaarSampleGeneration, QDT
from src.qstate_product_jax import project_to_product_ry, sample_ry_product_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help=".npz from mnist01_train_qdt.py")
    ap.add_argument("--out", default="data/mnist01/gen/qdt")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--na", type=int, default=0)
    ap.add_argument("--L", type=int, default=80)
    ap.add_argument("--n_gen", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--noise",
        choices=["product_ry", "haar"],
        default="product_ry",
        help="noise distribution at t=T (matches training).",
    )
    ap.add_argument(
        "--project_product_output",
        action="store_true",
        help="Project generated states to product Ry manifold before saving.",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"qstates_gen_n{args.n}.npy")

    if (not args.force) and os.path.exists(out_path):
        print(f"[skip] exists: {out_path}")
        return

    ckpt = np.load(args.ckpt, allow_pickle=True)
    params = jnp.array(ckpt["params"], dtype=jnp.float32)

    model = QDT(n=args.n, na=args.na, L=args.L)

    if args.noise == "haar":
        inputs = HaarSampleGeneration(args.n_gen, args.n, seed=args.seed)
        inputs = jnp.array(inputs, dtype=jnp.complex64)
    else:
        key_inputs = random.PRNGKey(args.seed + 999)
        inputs, _ = sample_ry_product_states(key_inputs, n=args.n, batch=args.n_gen)

    zero_shape = 2 ** (args.n + args.na) - 2**args.n
    zero_tensor = jnp.zeros((args.n_gen, zero_shape), dtype=jnp.complex64)
    input_full = jnp.concatenate([inputs, zero_tensor], axis=1)

    key = random.PRNGKey(args.seed)
    states = model.backwardOutput(input_full, params, key)

    if args.project_product_output:
        states, _ = project_to_product_ry(states, n=args.n)

    np.save(out_path, np.array(states).astype(np.complex64))
    print("saved:", out_path, states.shape)


if __name__ == "__main__":
    main()
