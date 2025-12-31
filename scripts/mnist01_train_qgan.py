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

from src.QGAN import HaarSampleGeneration, QGAN
from src.qstate_product_jax import project_to_product_ry, sample_ry_product_states


def _save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def training_generation(model, inputs, params_g, params_c, epochs, lr):
    loss_hist = []
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params_g)

    def loss_func_gen(params_g, inputs, params_c, key_local):
        states_gen = model.dataGenerate(inputs, params_g, key_local)
        if getattr(model, "_project_product_loss", False):
            states_gen, _ = project_to_product_ry(states_gen, n=model.n)
        zs = model.classCircuit_vmap(states_gen, params_c)
        prob_real = (1.0 + jnp.real(zs)) / 2.0
        return -jnp.mean(prob_real)

    loss_vg = jax.jit(jax.value_and_grad(loss_func_gen))

    def update(params_g, inputs, params_c, opt_state, key_local):
        loss_value, grads = loss_vg(params_g, inputs, params_c, key_local)
        updates, new_opt_state = optimizer.update(grads, opt_state, params_g)
        new_params_g = optax.apply_updates(params_g, updates)
        return new_params_g, new_opt_state, loss_value

    t0 = time.time()
    key = random.PRNGKey(getattr(model, "_seed", 42) + 2000)
    for step in range(epochs):
        key, subkey = random.split(key)
        params_g, opt_state, loss_value = update(params_g, inputs, params_c, opt_state, subkey)
        if step % 100 == 0:
            print(f"[gen] step {step:05d} loss={float(loss_value):.6f} elapsed={time.time()-t0:.1f}s")
        loss_hist.append(loss_value)

    return params_g, params_c, jnp.stack(loss_hist)


def training_classification(model, inputs, data_true, params_g, params_c, epochs, lr, seed):
    loss_hist = []
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params_c)

    def loss_func_class(params_c, inputs, data_true, params_g, key_local):
        states_gen = model.dataGenerate(inputs, params_g, key_local)
        if getattr(model, "_project_product_loss", False):
            states_gen, _ = project_to_product_ry(states_gen, n=model.n)
        zs_gen = model.classCircuit_vmap(states_gen, params_c)
        zs_true = model.classCircuit_vmap(data_true, params_c)
        prob_real_fake = (1.0 + jnp.real(zs_gen)) / 2.0
        prob_real_real = (1.0 + jnp.real(zs_true)) / 2.0
        return jnp.mean(prob_real_fake - prob_real_real)

    loss_vg = jax.jit(jax.value_and_grad(loss_func_class))

    def update(params_c, inputs, data_true, params_g, opt_state, key_local):
        loss_value, grads = loss_vg(params_c, inputs, data_true, params_g, key_local)
        updates, new_opt_state = optimizer.update(grads, opt_state, params_c)
        new_params_c = optax.apply_updates(params_c, updates)
        return new_params_c, new_opt_state, loss_value

    rng = np.random.default_rng(seed)
    t0 = time.time()
    key = random.PRNGKey(seed + 1000)
    for step in range(epochs):
        idx = rng.choice(data_true.shape[0], size=inputs.shape[0], replace=False)
        key, subkey = random.split(key)
        params_c, opt_state, loss_value = update(params_c, inputs, data_true[idx], params_g, opt_state, subkey)
        if step % 100 == 0:
            print(f"[cla] step {step:05d} loss={float(loss_value):.6f} elapsed={time.time()-t0:.1f}s")
        loss_hist.append(loss_value)

    return params_g, params_c, jnp.stack(loss_hist)


def benchmark_probs(model, inputs, data_true, params_g, params_c, seed):
    key = random.PRNGKey(seed + 3000)
    states_gen = model.dataGenerate(inputs, params_g, key)
    if getattr(model, "_project_product_loss", False):
        states_gen, _ = project_to_product_ry(states_gen, n=model.n)
    zs_gen = model.classCircuit_vmap(states_gen, params_c)
    rng = np.random.default_rng(seed)
    idx = rng.choice(data_true.shape[0], size=inputs.shape[0], replace=False)
    zs_true = model.classCircuit_vmap(data_true[idx], params_c)

    prob_rr = jnp.mean((1.0 + jnp.real(zs_true)) / 2.0)
    prob_rf = jnp.mean((1.0 + jnp.real(zs_gen)) / 2.0)
    return prob_rr, prob_rf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default="data/mnist01",
        help="Dataset directory (contains qstates_train_n{n}.npy) or a direct path to qstates_train_n{n}.npy",
    )
    ap.add_argument("--out", default="data/mnist01/models/qgan")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--na", type=int, default=0)
    ap.add_argument("--Lg", type=int, default=40)
    ap.add_argument("--Lc", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs_c", type=int, default=1000)
    ap.add_argument("--epochs_g", type=int, default=2000)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--lr_c", type=float, default=5e-4)
    ap.add_argument("--lr_g", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--noise",
        choices=["product_ry", "haar"],
        default="product_ry",
        help=(
            "Input noise distribution. For MNIST01 qstates (product Ry encoding), product_ry is recommended; "
            "haar tends to start highly entangled."
        ),
    )
    ap.add_argument(
        "--project_product_loss",
        action="store_true",
        help=(
            "Project generated states to product Ry manifold inside the loss (via <Z_i> -> theta -> product state). "
            "This prevents the QGAN from drifting to highly-entangled solutions (purity~0.5)."
        ),
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    ckpt_path = os.path.join(
        args.out,
        f"qgan_mnist01_n{args.n}na{args.na}Lg{args.Lg}Lc{args.Lc}_cy{args.cycles}.npz",
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
    data_true = jnp.array(data_true)

    # Fixed generator inputs
    if args.noise == "haar":
        inputs = HaarSampleGeneration(args.batch, args.n, seed=args.seed)
        inputs = jnp.array(inputs, dtype=jnp.complex64)
    else:
        key_inputs = random.PRNGKey(args.seed + 999)
        inputs, _ = sample_ry_product_states(key_inputs, n=args.n, batch=args.batch)

    model = QGAN(n=args.n, na=args.na, Lg=args.Lg, Lc=args.Lc)
    # small flags for training helpers (kept on model to avoid threading args everywhere)
    model._project_product_loss = bool(args.project_product_loss)
    model._seed = int(args.seed)

    loss_class = []
    loss_gen = []

    key = random.PRNGKey(args.seed)

    params_g = random.normal(key, shape=(2 * (args.n + args.na) * args.Lg,))
    params_c = random.normal(key, shape=(2 * args.n * args.Lc,))

    for cy in range(args.cycles):
        print(f"=== cycle {cy+1}/{args.cycles} ===")

        params_g, params_c, loss_c = training_classification(
            model, inputs, data_true, params_g, params_c, args.epochs_c, args.lr_c, seed=args.seed + 10 + cy
        )
        loss_class.append(np.array(loss_c))

        prob_rr, prob_rf = benchmark_probs(
            model,
            inputs,
            data_true,
            params_g,
            params_c,
            seed=args.seed + 100 + cy,
        )
        print(f"[disc] Prob(real->real)={float(prob_rr):.4f} Prob(fake->real)={float(prob_rf):.4f}")

        params_g, params_c, loss_g = training_generation(
            model, inputs, params_g, params_c, args.epochs_g, args.lr_g
        )
        loss_gen.append(np.array(loss_g))

        prob_rr, prob_rf = benchmark_probs(
            model,
            inputs,
            data_true,
            params_g,
            params_c,
            seed=args.seed + 200 + cy,
        )
        print(f"[gen ] Prob(fake->real)={float(prob_rf):.4f} disc_cost={float(prob_rf-prob_rr):.4f}")

    np.savez(
        ckpt_path,
        params_g=np.array(params_g),
        params_c=np.array(params_c),
        loss_gen=np.array(loss_gen, dtype=object),
        loss_class=np.array(loss_class, dtype=object),
    )

    _save_json(
        meta_path,
        {
            "n": args.n,
            "na": args.na,
            "Lg": args.Lg,
            "Lc": args.Lc,
            "batch": args.batch,
            "epochs_c": args.epochs_c,
            "epochs_g": args.epochs_g,
            "cycles": args.cycles,
            "lr_c": args.lr_c,
            "lr_g": args.lr_g,
            "seed": args.seed,
            "ckpt": ckpt_path,
        },
    )

    print("saved:")
    print(" ", ckpt_path)
    print(" ", meta_path)


if __name__ == "__main__":
    main()
