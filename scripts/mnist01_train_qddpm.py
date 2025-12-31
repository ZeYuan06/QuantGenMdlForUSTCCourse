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

from src.QDDPM_jax import HaarSampleGeneration, QDDPM
from src.distance_jax import naturalDistance
from src.qstate_product_jax import project_to_product_ry, sample_ry_product_states


def _save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _default_diff_path(data_dir: str, n: int, T: int, N: int, h_min: float, h_max: float) -> str:
    return os.path.join(data_dir, "diff", f"mnist01Diff_n{n}T{T}_N{N}_h{h_min:g}-{h_max:g}.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mnist01", help="MNIST01 data dir")
    ap.add_argument("--out", default="data/mnist01/models/qddpm")
    ap.add_argument("--diff", default=None, help="path to diffusion npy (T+1,N,2^n)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--na", type=int, default=2)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--N_train", type=int, default=256)
    ap.add_argument("--N_diff", type=int, default=5000, help="expected diffusion dataset size for default path")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--noise",
        choices=["product_ry", "haar"],
        default="product_ry",
        help=(
            "noise distribution used for inputs_T at t=T. For MNIST01 product Ry qstates, "
            "product_ry avoids starting from highly entangled Haar states."
        ),
    )
    ap.add_argument(
        "--project_product_loss",
        action="store_true",
        help=(
            "Project backwardOutput_t results to product Ry manifold inside the loss (via <Z_i> -> theta -> product state). "
            "Useful when target data are product-encoded MNIST01 qstates."
        ),
    )
    ap.add_argument("--h_min", type=float, default=0.1)
    ap.add_argument("--h_max", type=float, default=2.0)
    ap.add_argument("--resume", action="store_true", help="resume from existing per-t params")
    ap.add_argument("--force", action="store_true", help="overwrite existing per-t params")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    diff_path = args.diff or _default_diff_path(args.data, args.n, args.T, args.N_diff, args.h_min, args.h_max)
    if not os.path.exists(diff_path):
        raise FileNotFoundError(
            f"diffusion file not found: {diff_path}. Run scripts/mnist01_make_diffusion_qddpm.py first."
        )

    states_diff = np.load(diff_path).astype(np.complex64)
    if states_diff.shape[0] != args.T + 1 or states_diff.shape[2] != 2**args.n:
        raise ValueError(
            f"diff shape mismatch: got {states_diff.shape}, expected (T+1,N,{2**args.n}) with T={args.T}"
        )

    # Fixed inputs at t=T
    if args.noise == "haar":
        # HaarSampleGeneration in src/QDDPM_jax.py expects dim=2**n
        inputs_T = HaarSampleGeneration(args.N_train, 2**args.n, seed=args.seed)
        inputs_T = jnp.array(inputs_T, dtype=jnp.complex64)
    else:
        key_inputs = random.PRNGKey(args.seed + 999)
        inputs_T, _ = sample_ry_product_states(key_inputs, n=args.n, batch=args.N_train)

    model = QDDPM(n=args.n, na=args.na, T=args.T, L=args.L)
    model.set_diffusionSet(jnp.array(states_diff))

    param_dim = 2 * (args.n + args.na) * args.L

    ckpt_prefix = os.path.join(args.out, f"qddpm_mnist01_n{args.n}na{args.na}T{args.T}L{args.L}")
    final_ckpt = ckpt_prefix + ".npz"
    meta_path = ckpt_prefix + ".json"

    if (not args.force) and (not args.resume) and os.path.exists(final_ckpt):
        print(f"[skip] final checkpoint exists: {final_ckpt}")
        return

    # per-step storage for resume
    def step_param_path(t: int) -> str:
        return ckpt_prefix + f"_t{t}.npy"

    def step_loss_path(t: int) -> str:
        return ckpt_prefix + f"_t{t}_loss.npy"

    key = random.PRNGKey(args.seed)
    rng = np.random.default_rng(args.seed)

    # Train from T-1 down to 0
    params_tot = np.zeros((args.T, param_dim), dtype=np.float32)
    loss_tot = np.zeros((args.T, args.epochs), dtype=np.float32)

    for t in range(args.T - 1, -1, -1):
        p_path = step_param_path(t)
        l_path = step_loss_path(t)

        if args.resume and os.path.exists(p_path) and os.path.exists(l_path) and (not args.force):
            print(f"[resume] t={t} from {p_path}")
            params_tot[t] = np.load(p_path).astype(np.float32)
            lt = np.load(l_path).astype(np.float32)
            loss_tot[t, : min(args.epochs, lt.shape[0])] = lt[: args.epochs]
            continue

        # Load already-trained steps > t if resuming partially
        for tt in range(t + 1, args.T):
            pt = step_param_path(tt)
            if os.path.exists(pt):
                params_tot[tt] = np.load(pt).astype(np.float32)

        params_tot_j = jnp.array(params_tot)

        # Prepare inputs for this step using fixed params for steps > t
        input_tplus1 = model.prepareInput_t(inputs_T, params_tot_j, t, args.N_train, seed=args.seed + 1000 + t)

        # init params_t
        key, subkey = random.split(key)
        params_t = random.normal(subkey, shape=(param_dim,), dtype=jnp.float32)

        optimizer = optax.adam(learning_rate=args.lr)
        opt_state = optimizer.init(params_t)

        states_diff_t = jnp.array(states_diff[t])

        def loss_func(params_local, true_batch, key_local):
            out = model.backwardOutput_t(input_tplus1, params_local, key_local)
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

        t0 = time.time()
        for step in range(args.epochs):
            idx = rng.choice(states_diff.shape[1], size=args.N_train, replace=False)
            true_batch = states_diff_t[idx]
            key, k1 = random.split(key)
            params_t, opt_state, loss_value = update(params_t, opt_state, true_batch, k1)

            if step % 200 == 0:
                print(
                    f"[qddpm] t={t:02d} step {step:05d} loss={float(loss_value):.6f} elapsed={time.time()-t0:.1f}s"
                )

            loss_tot[t, step] = float(loss_value)

        params_tot[t] = np.array(params_t).astype(np.float32)

        np.save(p_path, params_tot[t])
        np.save(l_path, loss_tot[t])
        print(f"[saved] t={t} params+loss")

    np.savez(final_ckpt, params_tot=params_tot, loss_tot=loss_tot, diff_path=diff_path)

    _save_json(
        meta_path,
        {
            "n": args.n,
            "na": args.na,
            "T": args.T,
            "L": args.L,
            "N_train": args.N_train,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "diff": diff_path,
            "ckpt": final_ckpt,
        },
    )

    print("saved:")
    print(" ", final_ckpt)
    print(" ", meta_path)


if __name__ == "__main__":
    main()
