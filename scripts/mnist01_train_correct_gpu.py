#!/usr/bin/env python

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@dataclass(frozen=True)
class Budget:
    n_train: int
    n_test: int
    clf_epochs: int
    clf_batch: int
    qdt_L: int
    qdt_batch: int
    qdt_epochs: int
    qdt_lr: float
    qddpm_na: int
    qddpm_L: int
    qddpm_T: int
    qddpm_N_diff: int
    qddpm_N_train: int
    qddpm_epochs: int
    qddpm_lr: float
    qgan_na: int
    qgan_Lg: int
    qgan_Lc: int
    qgan_batch: int
    qgan_epochs_c: int
    qgan_epochs_g: int
    qgan_cycles: int
    qgan_lr_c: float
    qgan_lr_g: float
    n_gen: int
    n_eval: int


BUDGETS: Dict[str, Budget] = {
    # Fast end-to-end sanity check (minutes). Not for quality.
    "smoke": Budget(
        n_train=256,
        n_test=64,
        clf_epochs=1,
        clf_batch=64,
        qdt_L=4,
        qdt_batch=32,
        qdt_epochs=50,
        qdt_lr=5e-4,
        qddpm_na=2,
        qddpm_L=1,
        qddpm_T=2,
        qddpm_N_diff=64,
        qddpm_N_train=32,
        qddpm_epochs=50,
        qddpm_lr=5e-4,
        qgan_na=0,
        qgan_Lg=8,
        qgan_Lc=4,
        qgan_batch=64,
        qgan_epochs_c=200,
        qgan_epochs_g=200,
        qgan_cycles=1,
        qgan_lr_c=5e-4,
        qgan_lr_g=5e-4,
        n_gen=64,
        n_eval=64,
    ),
    # Reasonable default that actually trains something, but won't run for days.
    "medium": Budget(
        n_train=8000,
        n_test=2000,
        clf_epochs=20,
        clf_batch=128,
        qdt_L=80,
        qdt_batch=128,
        qdt_epochs=5000,
        qdt_lr=5e-4,
        qddpm_na=2,
        qddpm_L=6,
        qddpm_T=20,
        qddpm_N_diff=5000,
        qddpm_N_train=256,
        qddpm_epochs=400,
        qddpm_lr=5e-4,
        qgan_na=0,
        qgan_Lg=40,
        qgan_Lc=12,
        qgan_batch=128,
        qgan_epochs_c=1000,
        qgan_epochs_g=2000,
        qgan_cycles=3,
        qgan_lr_c=5e-4,
        qgan_lr_g=5e-4,
        n_gen=2000,
        n_eval=2000,
    ),
    # Heavy run. Use only if you know the time budget.
    "full": Budget(
        n_train=8000,
        n_test=2000,
        clf_epochs=20,
        clf_batch=128,
        qdt_L=80,
        qdt_batch=128,
        qdt_epochs=20000,
        qdt_lr=5e-4,
        qddpm_na=2,
        qddpm_L=6,
        qddpm_T=20,
        qddpm_N_diff=5000,
        qddpm_N_train=256,
        qddpm_epochs=1200,
        qddpm_lr=5e-4,
        qgan_na=0,
        qgan_Lg=60,
        qgan_Lc=16,
        qgan_batch=128,
        qgan_epochs_c=2000,
        qgan_epochs_g=5000,
        qgan_cycles=5,
        qgan_lr_c=5e-4,
        qgan_lr_g=5e-4,
        n_gen=2000,
        n_eval=2000,
    ),
}


def _run(cmd: List[str], env: Dict[str, str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def _script(path: str) -> str:
    return os.path.join(REPO_ROOT, "scripts", path)


def _qdt_ckpt_path(out_dir: str, n: int, na: int, L: int, batch: int, epochs: int) -> str:
    return os.path.join(out_dir, f"qdt_mnist01_n{n}na{na}L{L}_b{batch}_e{epochs}.npz")


def _qddpm_ckpt_prefix(out_dir: str, n: int, na: int, T: int, L: int) -> str:
    return os.path.join(out_dir, f"qddpm_mnist01_n{n}na{na}T{T}L{L}")


def _qgan_ckpt_path(out_dir: str, n: int, na: int, Lg: int, Lc: int, cycles: int) -> str:
    # Keep consistent with scripts/mnist01_train_qgan.py naming.
    return os.path.join(out_dir, f"qgan_mnist01_n{n}na{na}Lg{Lg}Lc{Lc}_cy{cycles}.npz")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "One-click *correct* MNIST01 training pipeline (QDT + QDDPM) that runs on JAX GPU when available.\n"
            "Run this inside your existing environment (e.g. `conda activate qml_gpu`)."
        )
    )
    ap.add_argument("--data", default="data/mnist01", help="dataset dir (will be created if missing)")
    ap.add_argument("--exp", default=None, help="experiment root (default: same as --data)")
    ap.add_argument("--digits", default="0,1")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", choices=sorted(BUDGETS.keys()), default="medium")
    ap.add_argument("--cuda_visible_devices", default=None, help='optional, e.g. "0" or "0,1"')

    ap.add_argument("--skip_prepare", action="store_true")
    ap.add_argument("--skip_encode", action="store_true")
    ap.add_argument("--skip_classifier", action="store_true")
    ap.add_argument("--skip_qdt", action="store_true")
    ap.add_argument("--skip_qddpm", action="store_true")
    ap.add_argument("--skip_qgan", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")

    ap.add_argument("--force", action="store_true", help="pass --force to underlying steps where supported")
    ap.add_argument("--resume_qddpm", action="store_true", help="pass --resume to QDDPM training")

    ap.add_argument(
        "--project_product",
        action="store_true",
        help=(
            "Project outputs to product Ry manifold (train loss + generated outputs). "
            "Recommended for MNIST01 where qstates are encoded as product Ry states."
        ),
    )

    args = ap.parse_args()

    data_dir = os.path.abspath(args.data)
    exp_dir = os.path.abspath(args.exp or args.data)

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    budget = BUDGETS[args.budget]

    env = os.environ.copy()
    # Avoid grabbing all GPU memory on shared machines.
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # Reduce noisy TF/XLA logs (TensorCircuit may trigger them).
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # Print basic runtime info.
    _run(
        [
            sys.executable,
            "-c",
            "import jax; print('jax_backend', jax.default_backend()); print('jax_devices', jax.devices())",
        ],
        env,
    )

    # Stage 1: dataset prep
    if not args.skip_prepare:
        _run(
            [
                sys.executable,
                _script("mnist01_prepare.py"),
                "--out",
                data_dir,
                "--digits",
                args.digits,
                "--n_train",
                str(budget.n_train),
                "--n_test",
                str(budget.n_test),
                "--seed",
                str(args.seed),
            ],
            env,
        )

    # Stage 2: encode latent -> product states
    if not args.skip_encode:
        _run(
            [
                sys.executable,
                _script("mnist01_encode_states.py"),
                "--data",
                data_dir,
                "--out",
                data_dir,
            ],
            env,
        )

    # Stage 3: train classifier (torch, for evaluation)
    classifier_dir = os.path.join(exp_dir, "classifier")
    classifier_ckpt = os.path.join(classifier_dir, "mnist01_cnn.pt")
    if not args.skip_classifier:
        _run(
            [
                sys.executable,
                _script("mnist01_train_classifier.py"),
                "--data",
                data_dir,
                "--out",
                classifier_dir,
                "--epochs",
                str(budget.clf_epochs),
                "--batch",
                str(budget.clf_batch),
                "--seed",
                str(args.seed),
            ],
            env,
        )

    # Stage 4: QDT train + generate
    qdt_model_dir = os.path.join(exp_dir, "models", "qdt")
    qdt_gen_dir = os.path.join(exp_dir, "gen", "qdt")
    qdt_ckpt = _qdt_ckpt_path(qdt_model_dir, args.n, 0, budget.qdt_L, budget.qdt_batch, budget.qdt_epochs)

    if not args.skip_qdt:
        qdt_cmd = [
            sys.executable,
            _script("mnist01_train_qdt.py"),
            "--data",
            data_dir,
            "--out",
            qdt_model_dir,
            "--n",
            str(args.n),
            "--na",
            "0",
            "--L",
            str(budget.qdt_L),
            "--batch",
            str(budget.qdt_batch),
            "--epochs",
            str(budget.qdt_epochs),
            "--lr",
            str(budget.qdt_lr),
            "--seed",
            str(args.seed),
            "--noise",
            "product_ry",
        ]
        if args.project_product:
            qdt_cmd.append("--project_product_loss")
        if args.force:
            qdt_cmd.append("--force")
        _run(qdt_cmd, env)

        gen_cmd = [
            sys.executable,
            _script("mnist01_generate_qdt.py"),
            "--ckpt",
            qdt_ckpt,
            "--out",
            qdt_gen_dir,
            "--n",
            str(args.n),
            "--na",
            "0",
            "--L",
            str(budget.qdt_L),
            "--n_gen",
            str(budget.n_gen),
            "--seed",
            str(args.seed + 100),
            "--noise",
            "product_ry",
        ]
        if args.project_product:
            gen_cmd.append("--project_product_output")
        if args.force:
            gen_cmd.append("--force")
        _run(gen_cmd, env)

    # Stage 4.5: QGAN train + generate
    qgan_model_dir = os.path.join(exp_dir, "models", "qgan")
    qgan_gen_dir = os.path.join(exp_dir, "gen", "qgan")
    qgan_ckpt = _qgan_ckpt_path(
        qgan_model_dir,
        args.n,
        budget.qgan_na,
        budget.qgan_Lg,
        budget.qgan_Lc,
        budget.qgan_cycles,
    )

    if not args.skip_qgan:
        qgan_cmd = [
            sys.executable,
            _script("mnist01_train_qgan.py"),
            "--data",
            data_dir,
            "--out",
            qgan_model_dir,
            "--n",
            str(args.n),
            "--na",
            str(budget.qgan_na),
            "--Lg",
            str(budget.qgan_Lg),
            "--Lc",
            str(budget.qgan_Lc),
            "--batch",
            str(budget.qgan_batch),
            "--epochs_c",
            str(budget.qgan_epochs_c),
            "--epochs_g",
            str(budget.qgan_epochs_g),
            "--cycles",
            str(budget.qgan_cycles),
            "--lr_c",
            str(budget.qgan_lr_c),
            "--lr_g",
            str(budget.qgan_lr_g),
            "--seed",
            str(args.seed),
        ]
        if args.project_product:
            qgan_cmd += ["--noise", "product_ry", "--project_product_loss"]
        if args.force:
            qgan_cmd.append("--force")
        _run(qgan_cmd, env)

        qgan_gen_cmd = [
            sys.executable,
            _script("mnist01_generate_qgan.py"),
            "--ckpt",
            qgan_ckpt,
            "--out",
            qgan_gen_dir,
            "--n",
            str(args.n),
            "--na",
            str(budget.qgan_na),
            "--Lg",
            str(budget.qgan_Lg),
            "--Lc",
            str(budget.qgan_Lc),
            "--n_gen",
            str(budget.n_gen),
            "--seed",
            str(args.seed + 150),
        ]
        if args.project_product:
            qgan_gen_cmd += ["--noise", "product_ry", "--project_product_output"]
        if args.force:
            qgan_gen_cmd.append("--force")
        _run(qgan_gen_cmd, env)

    # Stage 5: QDDPM diffusion + train + generate
    qddpm_model_dir = os.path.join(exp_dir, "models", "qddpm")
    qddpm_gen_dir = os.path.join(exp_dir, "gen", "qddpm")

    diff_dir = os.path.join(exp_dir, "diff")
    os.makedirs(diff_dir, exist_ok=True)

    diff_path = os.path.join(
        diff_dir,
        f"mnist01Diff_n{args.n}T{budget.qddpm_T}_N{budget.qddpm_N_diff}_h0.1-2.npy",
    )

    if not args.skip_qddpm:
        diff_cmd = [
            sys.executable,
            _script("mnist01_make_diffusion_qddpm.py"),
            "--data",
            data_dir,
            "--out",
            diff_dir,
            "--n",
            str(args.n),
            "--T",
            str(budget.qddpm_T),
            "--N",
            str(budget.qddpm_N_diff),
            "--h_min",
            "0.1",
            "--h_max",
            "2.0",
            "--seed",
            str(args.seed),
        ]
        if args.force:
            diff_cmd.append("--force")
        _run(diff_cmd, env)

        qddpm_cmd = [
            sys.executable,
            _script("mnist01_train_qddpm.py"),
            "--data",
            data_dir,
            "--out",
            qddpm_model_dir,
            "--diff",
            diff_path,
            "--n",
            str(args.n),
            "--na",
            str(budget.qddpm_na),
            "--T",
            str(budget.qddpm_T),
            "--L",
            str(budget.qddpm_L),
            "--N_train",
            str(budget.qddpm_N_train),
            "--N_diff",
            str(budget.qddpm_N_diff),
            "--epochs",
            str(budget.qddpm_epochs),
            "--lr",
            str(budget.qddpm_lr),
            "--seed",
            str(args.seed),
            "--noise",
            "product_ry",
            "--h_min",
            "0.1",
            "--h_max",
            "2.0",
        ]
        if args.project_product:
            qddpm_cmd.append("--project_product_loss")
        if args.resume_qddpm:
            qddpm_cmd.append("--resume")
        if args.force:
            qddpm_cmd.append("--force")
        _run(qddpm_cmd, env)

        qddpm_prefix = _qddpm_ckpt_prefix(qddpm_model_dir, args.n, budget.qddpm_na, budget.qddpm_T, budget.qddpm_L)
        qddpm_ckpt = qddpm_prefix + ".npz"

        gen_cmd = [
            sys.executable,
            _script("mnist01_generate_qddpm.py"),
            "--ckpt",
            qddpm_ckpt,
            "--out",
            qddpm_gen_dir,
            "--n",
            str(args.n),
            "--na",
            str(budget.qddpm_na),
            "--T",
            str(budget.qddpm_T),
            "--L",
            str(budget.qddpm_L),
            "--n_gen",
            str(budget.n_gen),
            "--seed",
            str(args.seed + 200),
            "--noise",
            "product_ry",
        ]
        if args.project_product:
            gen_cmd.append("--project_product_output")
        if args.force:
            gen_cmd.append("--force")
        _run(gen_cmd, env)

    # Stage 6: evaluation
    if not args.skip_eval:
        if not args.skip_qdt:
            _run(
                [
                    sys.executable,
                    _script("mnist01_eval.py"),
                    "--data",
                    data_dir,
                    "--gen",
                    qdt_gen_dir,
                    "--classifier",
                    classifier_ckpt,
                    "--n_eval",
                    str(budget.n_eval),
                    "--seed",
                    str(args.seed),
                ],
                env,
            )

        if not args.skip_qddpm:
            _run(
                [
                    sys.executable,
                    _script("mnist01_eval.py"),
                    "--data",
                    data_dir,
                    "--gen",
                    qddpm_gen_dir,
                    "--classifier",
                    classifier_ckpt,
                    "--n_eval",
                    str(budget.n_eval),
                    "--seed",
                    str(args.seed),
                ],
                env,
            )

        if not args.skip_qgan:
            _run(
                [
                    sys.executable,
                    _script("mnist01_eval.py"),
                    "--data",
                    data_dir,
                    "--gen",
                    qgan_gen_dir,
                    "--classifier",
                    classifier_ckpt,
                    "--n_eval",
                    str(budget.n_eval),
                    "--seed",
                    str(args.seed),
                ],
                env,
            )

    print("\nDone.")
    print("exp_dir:", exp_dir)
    print("qdt_gen:", qdt_gen_dir)
    print("qddpm_gen:", qddpm_gen_dir)
    print("qgan_gen:", qgan_gen_dir)


if __name__ == "__main__":
    main()
