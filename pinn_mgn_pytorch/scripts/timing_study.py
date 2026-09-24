"""Wall-clock study: baseline FEA vs trained MGN inference, with accuracy on the same meshes.

FEA   = assemble K and F + sparse direct solve (scipy/SuperLU, CPU).
MGN   = assemble F only (the learned-amplitude head needs no stiffness matrix) + graph features
        + forward pass, for one model and for the 3-member ensemble, on CPU and on the GPU (MPS).
Run on an otherwise idle machine; each timing is the best of `--repeats`.

    python scripts/timing_study.py --tag layer
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from elastic_sim import fem, mgn, pool  # noqa: E402

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def load_vector(prob):
    m = prob.mesh
    F = fem.body_force_vector(m.nodes, m.elems, prob.body_force) if any(prob.body_force) else np.zeros(2 * m.n_nodes)
    for bname, fn in prob.tractions:
        F += fem.traction_vector(m, m.boundary[bname], fn)
    return F


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="layer")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(OUT, f"mgn_benchmark_{args.tag}.json")))["config"]
    models = []
    for sd in range(cfg["seeds"]):
        m = mgn.MeshGraphNet(hidden=cfg["hidden"], n_layers=cfg["layers"], norm=cfg["norm"])
        m.load_state_dict(torch.load(os.path.join(OUT, f"mgn_{args.tag}_seed{sd}.pt")))
        models.append(m.eval())
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    probs = [p for p in pool.test_problems()] + [
        pool.beam_problem(4.0, 1.0, 96, 24, "end_shear"), pool.beam_problem(4.0, 1.0, 192, 48, "end_shear"),
        pool.hole_problem(0.25, 1, 1, 80, 40), pool.hole_problem(0.25, 1, 1, 160, 80),
        pool.hole_problem(0.25, 1, 0, 224, 112),
        pool.plate_problem(1.0, 1.0, 60, "bottom", "lateral"), pool.plate_problem(1.0, 1.0, 140, "bottom", "lateral")]
    rows = {}
    for p in probs:
        r = dict(n_dofs=2 * p.mesh.n_nodes)
        fea_best, res = np.inf, None
        for _ in range(args.repeats):
            res = fem.solve(p)
            fea_best = min(fea_best, res.t_total)
        r["fea_s"] = fea_best
        for dev in devices:
            ms = [m.to(dev) for m in models]
            best1, bestE = np.inf, np.inf
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                F = load_vector(p)
                fake = fem.FEAResult(np.zeros((p.mesh.n_nodes, 2)), None, F, None, None, None, 0, 0)
                s = pool.make_sample(p, fea=_WithK(fake, p))
                b = pool.collate([s], dev)
                t_prep = time.perf_counter() - t0
                with torch.no_grad():
                    sync(dev); t1 = time.perf_counter()
                    u1 = mgn.predict_displacement(ms[0], b)[0]
                    sync(dev); t2 = time.perf_counter()
                    uE = torch.stack([mgn.predict_displacement(m, b)[0] for m in ms]).mean(0)
                    sync(dev); t3 = time.perf_counter()
                best1, bestE = min(best1, t_prep + t2 - t1), min(bestE, t_prep + t3 - t2)
            r[f"mgn1_{dev}_s"], r[f"mgnE_{dev}_s"] = best1, bestE
            r["mgn1_rel_l2"] = fem.relative_l2(u1.cpu().numpy(), res.u)
            r["mgnE_rel_l2"] = fem.relative_l2(uE.cpu().numpy(), res.u)
        rows[p.name] = r
        print(f"{p.name:36s} dofs={r['n_dofs']:6d} FEA={r['fea_s'] * 1e3:7.1f}ms " +
              " ".join(f"{k[:-2]}={v * 1e3:7.1f}ms" for k, v in r.items() if k.startswith("mgn") and k.endswith("_s")) +
              f"  err1={r['mgn1_rel_l2']:.3f} errE={r['mgnE_rel_l2']:.3f}", flush=True)
    json.dump(rows, open(os.path.join(OUT, f"timing_study_{args.tag}.json"), "w"), indent=2)


class _WithK:
    """make_sample() stores K for the physics loss; inference with the amplitude head does not need it,
    so give it an empty sparse matrix instead of assembling one (keeps the timed path honest)."""

    def __init__(self, fake, prob):
        import scipy.sparse as sp
        n = 2 * prob.mesh.n_nodes
        self.K, self.F, self.u = sp.csr_matrix((n, n)), fake.F, fake.u
        self.t_assemble = self.t_solve = 0.0


if __name__ == "__main__":
    main()
