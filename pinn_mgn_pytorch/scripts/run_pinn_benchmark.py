"""Benchmark the strong-form PINN against baseline FEA on several geometries, BCs and tractions.

For every case: train an ensemble of PINNs (different seeds) on 10k+ collocation points,
evaluate on the FEA nodes, and report relative L2 displacement error vs FEA
(and vs the analytic solution for the Timoshenko beam).

    python scripts/run_pinn_benchmark.py [--seeds 3] [--adam 3000] [--lbfgs 3000]
"""
import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from elastic_sim import fem, pinn, problems  # noqa: E402

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def plot_case(case, nodes, u_fea, u_pinn, path):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    mag = lambda u: np.linalg.norm(u, axis=1)
    vmax = mag(u_fea).max()
    for a, (title, val) in zip(ax, [("FEA |u|", mag(u_fea)), ("PINN ensemble |u|", mag(u_pinn)),
                                    ("|u_PINN - u_FEA| / max|u_FEA|", mag(u_pinn - u_fea) / vmax)]):
        sc = a.scatter(nodes[:, 0], nodes[:, 1], c=val, s=8, cmap="viridis",
                       vmax=None if "PINN -" in title else vmax)
        a.set_title(title); a.set_aspect("equal"); fig.colorbar(sc, ax=a, shrink=0.8)
    fig.suptitle(case.name)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--adam", type=int, default=3000)
    ap.add_argument("--adam-energy", type=int, default=8000, help="Adam steps for energy-loss cases (no L-BFGS)")
    ap.add_argument("--lbfgs", type=int, default=3000)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--device", default=None, help="cpu | mps | cuda (default: best available)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "pinn_benchmark.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    for case in problems.benchmark_cases():
        if args.cases and case.name not in args.cases:
            continue
        fea = fem.solve(case.problem)
        nodes = case.problem.mesh.nodes
        members, infos = [], []
        for seed in range(args.seeds):
            adam = args.adam_energy if case.pinn_loss == "energy" else args.adam
            m, info = pinn.train_pinn(case, seed=seed, adam_steps=adam, lbfgs_steps=args.lbfgs,
                                      log_every=1000, loss=case.pinn_loss, device=args.device)
            members.append(m)
            infos.append(info)
        ens = pinn.PinnEnsemble(members)
        u_mean, u_std = ens.predict(nodes)
        member_err = [fem.relative_l2(pinn.predict(m, nodes), fea.u) for m in members]
        r = dict(device=str(args.device or pinn.default_device()), geometry=case.meta["geometry"], bc=case.meta["bc"], load=case.meta["load"], loss=case.pinn_loss,
                 n_fea_nodes=int(case.problem.mesh.n_nodes), n_collocation=infos[0]["n_collocation"],
                 member_rel_l2_vs_fea=member_err, ensemble_rel_l2_vs_fea=fem.relative_l2(u_mean, fea.u),
                 ensemble_max_std_over_max_u=float(np.linalg.norm(u_std, axis=1).max() /
                                                    np.linalg.norm(fea.u, axis=1).max()),
                 train_time_s=[i["train_time"] for i in infos], final_loss_parts=infos[0]["final_parts"],
                 fea_time_s=fea.t_total)
        if case.exact is not None:
            ex = case.exact(nodes)
            r["ensemble_rel_l2_vs_exact"] = fem.relative_l2(u_mean, ex)
            r["fea_rel_l2_vs_exact"] = fem.relative_l2(fea.u, ex)
        results[case.name] = r
        print(f"{case.name}: ensemble rel-L2 vs FEA = {r['ensemble_rel_l2_vs_fea']:.4f}  "
              f"members = {np.round(member_err, 4).tolist()}", flush=True)
        plot_case(case, nodes, fea.u, u_mean, os.path.join(OUT, f"pinn_{case.name}.png"))
        with open(os.path.join(OUT, "pinn_benchmark.json"), "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
