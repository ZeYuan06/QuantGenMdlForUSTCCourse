#!/usr/bin/env python

import argparse
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
from jax import numpy as jnp
from jax import random
import optax

from src.QDT_jax import HaarSampleGeneration, QDT
from src.distance_jax import naturalDistance
from src.qstate_product_jax import project_to_product_ry, sample_ry_product_states


def _save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default="data/mnist01",
        help="Dataset directory (contains qstates_train_n{n}.npy) or a direct path to qstates_train_n{n}.npy",
    )
    ap.add_argument("--out", default="data/mnist01/models/qdt")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--na", type=int, default=0)
    ap.add_argument("--L", type=int, default=80)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--noise",
        choices=["product_ry", "haar"],
        default="product_ry",
        help=(
            "noise distribution at t=T. For MNIST01 qstates (encoded as product Ry states), "
            "product_ry is the natural choice; haar tends to start highly entangled."
        ),
    )
    ap.add_argument(
        "--project_product_loss",
        action="store_true",
        help=(
            "Project model outputs to the product Ry manifold inside the loss (via <Z_i> -> theta -> product state). "
            "This matches how MNIST01 qstates are encoded and helps avoid learning highly-entangled solutions."
        ),
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    ckpt_path = os.path.join(
        args.out,
        f"qdt_mnist01_n{args.n}na{args.na}L{args.L}_b{args.batch}_e{args.epochs}.npz",
    )
    meta_path = ckpt_path.replace(".npz", ".json")

    if (not args.force) and os.path.exists(ckpt_path):
        print(f"[skip] checkpoint exists: {ckpt_path}")
        return

    if os.path.isdir(args.data):
        qstates_path = os.path.join(args.data, f"qstates_train_n{args.n}.npy")
    else:
        qstates_path = args.data

    data_true = np.load(qstates_path).astype(np.complex64)
    if data_true.shape[1] != 2**args.n:
        raise ValueError(f"qstates dim mismatch: got {data_true.shape}, expected (N,{2**args.n})")

    # Fixed generator inputs at t=T
    if args.noise == "haar":
        inputs = HaarSampleGeneration(args.batch, args.n, seed=args.seed)
        inputs = jnp.array(inputs, dtype=jnp.complex64)
    else:
        key_inputs = random.PRNGKey(args.seed + 999)
        inputs, _ = sample_ry_product_states(key_inputs, n=args.n, batch=args.batch)

    model = QDT(n=args.n, na=args.na, L=args.L)

    # For compatibility with the notebook pattern; only states_diff[0] is sampled.
    model.set_diffusionSet(jnp.stack([jnp.array(data_true), jnp.array(data_true)]))

    zero_shape = 2 ** (args.n + args.na) - 2**args.n
    zero_tensor = jnp.zeros((args.batch, zero_shape), dtype=jnp.complex64)
    input_full = jnp.concatenate([inputs, zero_tensor], axis=1)

    key = random.PRNGKey(args.seed)
    params = random.normal(key, shape=(2 * (args.n + args.na) * args.L,), dtype=jnp.float32)

    optimizer = optax.adam(learning_rate=args.lr)
    opt_state = optimizer.init(params)

    rng = np.random.default_rng(args.seed)

    def loss_func(params_local, true_batch, key_local):
        out = model.backwardOutput(input_full, params_local, key_local)
        if args.project_product_loss:
            out, _ = project_to_product_ry(out, n=args.n)
        return naturalDistance(out, true_batch)

    loss_vg = jax.jit(jax.value_and_grad(loss_func))

    @jax.jit
    def update(params_local, opt_state_local, true_batch, key_local):
        loss_value, grads = loss_vg(params_local, true_batch, key_local)
        updates, new_opt_state = optimizer.update(grads, opt_state_local, params_local)
        new_params = optax.apply_updates(params_local, updates)
        return new_params, new_opt_state, loss_value

    loss_hist = []
    t0 = time.time()

    data_true_j = jnp.array(data_true)

    for step in range(args.epochs):
        idx = rng.choice(data_true.shape[0], size=args.batch, replace=False)
        true_batch = data_true_j[idx]

        key, subkey = random.split(key)
        params, opt_state, loss_value = update(params, opt_state, true_batch, subkey)

        if step % 200 == 0:
            print(f"[qdt] step {step:06d} loss={float(loss_value):.6f} elapsed={time.time()-t0:.1f}s")

        loss_hist.append(float(loss_value))

    np.savez(
        ckpt_path,
        params=np.array(params),
        loss=np.array(loss_hist, dtype=np.float32),
    )

    _save_json(
        meta_path,
        {
            "n": args.n,
            "na": args.na,
            "L": args.L,
            "batch": args.batch,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "ckpt": ckpt_path,
            "qstates": qstates_path,
        },
    )

    print("saved:")
    print(" ", ckpt_path)
    print(" ", meta_path)


if __name__ == "__main__":
    main()
