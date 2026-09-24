"""Train the physics-informed MGN ensemble on the training pool and benchmark it against FEA.

Reports, on held-out meshes/configurations:
  * relative L2 displacement error of each member and of the ensemble mean vs FEA
  * wall-clock: baseline FEA (assembly + sparse direct solve) vs MGN inference
    (assembly + features + ensemble forward + Ritz amplitude)
  * wall-clock/iterations of Jacobi-preconditioned CG to rtol=1e-8 starting from zero vs
    starting from the MGN prediction (the MGN used as a learned initial guess)

    python scripts/run_mgn_benchmark.py [--seeds 3] [--epochs 150] [--norm layer]
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from elastic_sim import fem, mgn, pool  # noqa: E402

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def mgn_inference_time(models, prob, repeats=3):
    """Time the full MGN pipeline on a new mesh (no FEA solve anywhere in this path)."""
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        K, F = fem.assemble(prob)
        fake = fem.FEAResult(np.zeros((prob.mesh.n_nodes, 2)), K, F, None, None, None, 0.0, 0.0)
        s = pool.make_sample(prob, fea=fake)
        b = pool.collate([s])
        with torch.no_grad():
            u = torch.stack([mgn.predict_displacement(m, b)[0] for m in models]).mean(0).numpy()
        best = min(best, time.perf_counter() - t0)
    return best, u


def fea_time(prob, repeats=3):
    best = np.inf
    for _ in range(repeats):
        r = fem.solve(prob)
        best = min(best, r.t_total)
    return best, r


def pcg(res, x0=None, rtol=1e-8):
    Kff, rhs = fem.reduced_system(res)
    d = Kff.diagonal()
    M = spla.LinearOperator(Kff.shape, matvec=lambda v: v / d)
    it = [0]
    t0 = time.perf_counter()
    x, info = spla.cg(Kff, rhs, x0=x0, rtol=rtol, maxiter=200000, M=M,
                      callback=lambda _: it.__setitem__(0, it[0] + 1))
    return time.perf_counter() - t0, it[0], info


def timing_problems():
    """Larger versions of the held-out configurations for the timing study."""
    return [pool.beam_problem(7.0, 1.0, 140, 20, "end_shear"),
            pool.beam_problem(7.0, 1.0, 280, 40, "end_shear"),
            pool.hole_problem(0.25, 1, 1, 80, 40),
            pool.hole_problem(0.25, 1, 1, 160, 80),
            pool.plate_problem(1.5, 1.0, 60, "left", "gravity"),
            pool.plate_problem(1.5, 1.0, 120, "left", "gravity")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--norm", default="layer", choices=["layer", "batch", "none"])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=15)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--curriculum", action="store_true",
                    help="warm-start on coarse meshes (<=300 nodes) for 30%% of epochs, then the full pool")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = args.tag or f"{args.norm}"

    train = pool.build_pool(pool.training_problems())
    test = pool.build_pool(pool.test_problems())
    curriculum = [(0.3, lambda m: m["n_nodes"] <= 300), (0.7, lambda m: True)] if args.curriculum else None
    models, infos = [], []
    for seed in range(args.seeds):
        m, info = mgn.train_mgn(train, seed=seed, epochs=args.epochs, norm=args.norm, hidden=args.hidden,
                                n_layers=args.layers, device=args.device, curriculum=curriculum)
        models.append(m)
        infos.append(info)
        torch.save(m.state_dict(), os.path.join(OUT, f"mgn_{tag}_seed{seed}.pt"))

    ens = mgn.MGNEnsemble(models)
    acc = {}
    for s in test:
        u_ref = s.u_fea.numpy()
        members = [fem.relative_l2(mgn.MGNEnsemble([m]).predict(s)[0], u_ref) for m in models]
        u_ens, _ = ens.predict(s)
        acc[s.meta["name"]] = dict(geometry=s.meta["geometry"], load=s.meta["load"], n_nodes=s.n_nodes,
                                   member_rel_l2=members, ensemble_rel_l2=fem.relative_l2(u_ens, u_ref))
        print(f"{s.meta['name']:38s} n={s.n_nodes:5d} ensemble={acc[s.meta['name']]['ensemble_rel_l2']:.4f} "
              f"members={np.round(members, 4).tolist()}", flush=True)
    train_err = [fem.relative_l2(ens.predict(s)[0], s.u_fea.numpy()) for s in train]

    timing = {}
    for prob in [p for p in pool.test_problems()] + timing_problems():
        t_fea, res = fea_time(prob)
        t_mgn, u_mgn = mgn_inference_time(models, prob)
        t_cg0, it0, _ = pcg(res)
        t_cgw, itw, _ = pcg(res, x0=u_mgn.reshape(-1)[res.free])
        timing[prob.name] = dict(n_dofs=2 * prob.mesh.n_nodes, fea_direct_s=t_fea, mgn_inference_s=t_mgn,
                                 mgn_rel_l2=fem.relative_l2(u_mgn, res.u),
                                 pcg_zero_s=t_cg0, pcg_zero_iters=it0,
                                 pcg_mgn_s=t_mgn + t_cgw, pcg_mgn_iters=itw,
                                 pcg_time_reduction=1 - (t_mgn + t_cgw) / t_cg0,
                                 pcg_iter_reduction=1 - itw / max(it0, 1))
        print(f"{prob.name:38s} dofs={2 * prob.mesh.n_nodes:6d} FEA={t_fea * 1e3:7.1f}ms MGN={t_mgn * 1e3:7.1f}ms "
              f"PCG0={t_cg0 * 1e3:7.1f}ms({it0}) PCG+MGN={(t_mgn + t_cgw) * 1e3:7.1f}ms({itw})", flush=True)

    out = dict(config=vars(args), train_pool=dict(n_graphs=len(train),
                                                  geometries=sorted({s.meta["geometry"] for s in train}),
                                                  median_rel_l2=float(np.median(train_err))),
               train_time_s=[i["train_time"] for i in infos], history=[i["history"] for i in infos],
               test_accuracy=acc, timing=timing)
    with open(os.path.join(OUT, f"mgn_benchmark_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
