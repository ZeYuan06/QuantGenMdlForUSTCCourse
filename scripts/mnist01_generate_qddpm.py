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

from src.QDDPM_jax import HaarSampleGeneration, QDDPM
from src.qstate_product_jax import project_to_product_ry, sample_ry_product_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help=".npz from mnist01_train_qddpm.py")
    ap.add_argument("--out", default="data/mnist01/gen/qddpm")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--na", type=int, default=2)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--n_gen", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--noise",
        choices=["product_ry", "haar"],
        default="product_ry",
        help="noise distribution used for inputs_T at t=T (matches training).",
    )
    ap.add_argument(
        "--project_product_output",
        action="store_true",
        help="Project x0 (generated states) to product Ry manifold before saving.",
    )
    ap.add_argument("--save_chain", action="store_true", help="also save full chain states_gen_Tp1.npy")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    out_path = os.path.join(args.out, f"qstates_gen_n{args.n}.npy")
    chain_path = os.path.join(args.out, f"qstates_chain_n{args.n}T{args.T}.npy")

    if (not args.force) and os.path.exists(out_path) and ((not args.save_chain) or os.path.exists(chain_path)):
        print(f"[skip] exists: {out_path}")
        return

    ckpt = np.load(args.ckpt, allow_pickle=True)
    params_tot = ckpt["params_tot"].astype(np.float32)

    if params_tot.shape[0] != args.T:
        raise ValueError(f"params_tot T mismatch: got {params_tot.shape}, expected T={args.T}")

    model = QDDPM(n=args.n, na=args.na, T=args.T, L=args.L)

    if args.noise == "haar":
        # HaarSampleGeneration in src/QDDPM_jax.py expects dim=2**n
        inputs_T = HaarSampleGeneration(args.n_gen, 2**args.n, seed=args.seed)
        inputs_T = jnp.array(inputs_T, dtype=jnp.complex64)
    else:
        key_inputs = random.PRNGKey(args.seed + 999)
        inputs_T, _ = sample_ry_product_states(key_inputs, n=args.n, batch=args.n_gen)

    states = model.backDataGeneration(inputs_T, jnp.array(params_tot), args.n_gen, seed=args.seed)
    states_np = np.array(states).astype(np.complex64)  # (T+1,N,2^n)

    x0 = states_np[0]
    if args.project_product_output:
        x0_j, _ = project_to_product_ry(jnp.array(x0), n=args.n)
        x0 = np.array(x0_j).astype(np.complex64)
        states_np[0] = x0
    np.save(out_path, x0)
    print("saved:", out_path, x0.shape)

    if args.save_chain:
        np.save(chain_path, states_np)
        print("saved:", chain_path, states_np.shape)


if __name__ == "__main__":
    main()
