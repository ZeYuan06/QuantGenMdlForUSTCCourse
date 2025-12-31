#!/usr/bin/env python

"""Plot a run-level metric comparison from eval reports.

This is meant for course presentation: after a single run directory contains
  <exp>/gen/qddpm/eval_report.json
  <exp>/gen/qdt/eval_report.json
  <exp>/gen/qgan/eval_report.json
we generate a couple of PNG figures under <out>/.

Only uses matplotlib + stdlib.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"expected dict in {path}")
    return obj


def _pick_report(gen_dir: Path) -> Optional[Path]:
    for name in ["eval_report.json", "eval_report_qstate.json"]:
        p = gen_dir / name
        if p.exists():
            return p
    return None


def _get_qm(r: Dict[str, Any]) -> Dict[str, Any]:
    qm = r.get("qstate_metrics")
    if isinstance(qm, dict):
        out = dict(qm)
    else:
        out = {}
    if "natural_distance" not in out and "qstate_natural_distance" in r:
        out["natural_distance"] = r.get("qstate_natural_distance")
    return out


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment dir, e.g. data/mnist01_run_YYYY-mm-dd_HHMMSS")
    ap.add_argument("--out", default=None, help="output dir for plots (default: <exp>/plots)")
    args = ap.parse_args()

    exp_dir = Path(args.exp).resolve()
    out_dir = Path(args.out).resolve() if args.out else (exp_dir / "plots")
    _ensure_dir(out_dir)

    models = [
        ("QDDPM", exp_dir / "gen" / "qddpm"),
        ("QDT", exp_dir / "gen" / "qdt"),
        ("QGAN", exp_dir / "gen" / "qgan"),
    ]

    rows = []
    for name, gdir in models:
        rp = _pick_report(gdir)
        if rp is None:
            continue
        r = _load_json(rp)
        qm = _get_qm(r)
        rows.append(
            {
                "name": name,
                "report": str(rp),
                "q_natural": qm.get("natural_distance"),
                "q_mmd_z_zz": qm.get("feature_mmd_rbf_z_zz"),
                "purity": qm.get("single_qubit_purity_mean"),
                "real_pred_frac_1": r.get("real_pred_frac_1"),
                "generated_pred_frac_1": r.get("generated_pred_frac_1"),
            }
        )

    if not rows:
        raise SystemExit(f"No eval_report*.json found under {exp_dir}/gen/*")

    labels = [x["name"] for x in rows]

    # --- Figure 1: quantum primary metrics ---
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.6))

    axs[0].bar(labels, [x["q_natural"] for x in rows])
    axs[0].set_title("natural distance (↓ better)")

    axs[1].bar(labels, [x["q_mmd_z_zz"] for x in rows])
    axs[1].set_title("MMD on ⟨Z⟩/⟨ZZ⟩ (↓ better)")

    axs[2].bar(labels, [x["purity"] for x in rows])
    axs[2].set_ylim(0.0, 1.05)
    axs[2].set_title("mean single-qubit purity (→1 good)")

    fig.suptitle(f"MNIST01 run metrics: {exp_dir.name}")
    fig.tight_layout()
    p1 = out_dir / "quantum_metrics.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # --- Figure 2: class fraction alignment (auxiliary) ---
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.6))
    real = [x["real_pred_frac_1"] for x in rows]
    gen = [x["generated_pred_frac_1"] for x in rows]
    x = list(range(len(labels)))
    w = 0.35
    ax.bar([i - w / 2 for i in x], real, width=w, label="real_pred_frac_1")
    ax.bar([i + w / 2 for i in x], gen, width=w, label="generated_pred_frac_1")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("classifier-predicted class-1 fraction (closer is better)")
    ax.legend()
    fig.tight_layout()
    p2 = out_dir / "class_fraction.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)

    # --- Write a tiny text summary for convenience ---
    summary_path = out_dir / "summary.txt"
    lines = [f"exp: {exp_dir}", "", "reports:"]
    for r in rows:
        lines.append(f"- {r['name']}: {r['report']}")
    lines.append("")
    lines.append("quantum metrics (lower is better for distances):")
    for r in rows:
        lines.append(
            f"- {r['name']}: natural={r['q_natural']}, mmd_z_zz={r['q_mmd_z_zz']}, purity={r['purity']}"
        )
    lines.append("")
    lines.append("saved plots:")
    lines.append(str(p1))
    lines.append(str(p2))
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("saved:")
    print(" ", p1)
    print(" ", p2)
    print(" ", summary_path)


if __name__ == "__main__":
    main()
