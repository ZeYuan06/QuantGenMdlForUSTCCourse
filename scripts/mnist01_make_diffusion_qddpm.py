#!/usr/bin/env python

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from jax import numpy as jnp

from src.QDDPM_jax import setDiffusionDataMultiQubit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default="data/mnist01",
        help="Dataset directory (contains qstates_train_n{n}.npy) or a direct path to qstates_train_n{n}.npy",
    )
    ap.add_argument("--out", default="data/mnist01/diff")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--N", type=int, default=5000, help="number of states to diffuse (subsample from train)")
    ap.add_argument("--h_min", type=float, default=0.1)
    ap.add_argument("--h_max", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    out_path = os.path.join(args.out, f"mnist01Diff_n{args.n}T{args.T}_N{args.N}_h{args.h_min:g}-{args.h_max:g}.npy")
    if (not args.force) and os.path.exists(out_path):
        print(f"[skip] exists: {out_path}")
        return

    if os.path.isdir(args.data):
        qstates_path = os.path.join(args.data, f"qstates_train_n{args.n}.npy")
    else:
        qstates_path = args.data

    x0 = np.load(qstates_path).astype(np.complex64)
    if x0.shape[1] != 2**args.n:
        raise ValueError(f"qstates dim mismatch: got {x0.shape}, expected (N,{2**args.n})")

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(x0.shape[0], size=min(args.N, x0.shape[0]), replace=False)
    x0 = x0[idx]

    diff_hs = jnp.linspace(args.h_min, args.h_max, args.T)

    x0_j = jnp.array(x0)

    Xout = np.zeros((args.T + 1, x0.shape[0], 2**args.n), dtype=np.complex64)
    Xout[0] = x0

    for t in range(1, args.T + 1):
        xt = setDiffusionDataMultiQubit(x0_j, diff_hs[:t], args.n)
        Xout[t] = np.array(xt).astype(np.complex64)
        if t % 5 == 0 or t == args.T:
            print(f"[diff] t={t}/{args.T} done")

    np.save(out_path, Xout)
    print("saved:", out_path, Xout.shape)


if __name__ == "__main__":
    main()
