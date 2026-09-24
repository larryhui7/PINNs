"""Training-evolution deformation GIFs for the PyTorch models (counterpart of the MATLAB make_*_gifs.m).

Each frame: the predicted deformed mesh coloured by |u| (magnified), the FEA deformed mesh dashed black,
the undeformed mesh in light grey, and the loss / error-vs-FEA curves drawn up to the current step.
One model per case is retrained with a snapshot callback (seed 0, same settings as the benchmarks).

    python scripts/make_pytorch_gifs.py pinn     # results/gifs/pinn_<case>.gif      (5 cases)
    python scripts/make_pytorch_gifs.py mgn      # results/gifs/mgn_<test mesh>.gif  (3 held-out meshes)
    python scripts/make_pytorch_gifs.py fsi      # results/gifs/fsi_<test flap>.gif
"""
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
from matplotlib.collections import LineCollection, PolyCollection
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from elastic_sim import fem, fsi, mgn, pinn, pool, problems  # noqa: E402

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "gifs")
C = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e6e5df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2})


# ----------------------------------------------------------------------------- rendering
def _edges(elems):
    e = np.concatenate([elems[:, [i, (i + 1) % 4]] for i in range(4)])
    return np.unique(np.sort(e, axis=1), axis=0)


def _tri(elems):
    return np.concatenate([elems[:, [0, 1, 2]], elems[:, [0, 2, 3]]])


def nice_magnification(u_ref, extent, frac=0.12):
    m = frac * extent / max(np.abs(u_ref).max(), 1e-30)
    p = 10 ** np.floor(np.log10(m))
    return float(min((1, 2, 5, 10), key=lambda k: abs(k * p - m)) * p)


def save_gif(frames, path, ms=80, hold=15):
    frames = frames + [frames[-1]] * hold
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)


def fig_to_image(fig):
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).convert("P", palette=Image.ADAPTIVE)


def render_mesh_gif(nodes, elems, u_ref, snaps, title, path, xlabel="step", curve_label="loss"):
    """snaps: list of (step, u_pred (n,2), loss, rel_err); curves are drawn from all snaps up to the frame."""
    magn = nice_magnification(u_ref, np.ptp(nodes, axis=0).max())
    cmax = np.linalg.norm(u_ref, axis=1).max()
    ed = _edges(elems)
    def_ref = nodes + magn * u_ref
    allpts = np.concatenate([nodes, def_ref] + [nodes + magn * s[1] for s in snaps[-3:]])
    lo, hi = allpts.min(0), allpts.max(0)
    pad = 0.06 * (hi - lo).max()
    steps = np.array([s[0] for s in snaps]); losses = np.array([s[2] for s in snaps])
    errs = np.maximum(np.array([s[3] for s in snaps]), 1e-5)
    pos = losses[losses > 0]
    frames = []
    fig = plt.figure(figsize=(11.5, 4.8), dpi=85)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], hspace=0.35, wspace=0.25)
    for k, (step, u, loss, err) in enumerate(snaps):
        fig.clf()
        ax = fig.add_subplot(gs[:, 0])
        ax.add_collection(LineCollection(nodes[ed], colors="#d9d8d2", linewidths=0.5))
        dp = nodes + magn * u
        tc = ax.tripcolor(mtri.Triangulation(dp[:, 0], dp[:, 1], _tri(elems)), np.linalg.norm(u, axis=1),
                          shading="gouraud", cmap="turbo", vmin=0, vmax=cmax)
        ax.add_collection(LineCollection(dp[ed], colors="#333333", linewidths=0.3, alpha=0.6))
        ax.add_collection(LineCollection(def_ref[ed], colors="k", linewidths=0.9, linestyles=(0, (4, 3))))
        ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad); ax.set_aspect("equal")
        ax.grid(False)
        fig.colorbar(tc, ax=ax, shrink=0.8, label="|u| predicted")
        ax.set_title(f"{title}\n{xlabel} {step}   deflection ×{magn:g}   (dashed: FEA)", color=INK, fontsize=10)
        a1 = fig.add_subplot(gs[0, 1])
        a1.semilogy(steps[:k + 1], np.maximum(losses[:k + 1], 1e-30), color=C[0])
        a1.set_xlim(steps.min(), steps.max()); a1.set_ylim(pos.min() * 0.5, pos.max() * 2)
        a1.set_ylabel(curve_label); a1.set_title("training", loc="left", color=INK)
        a2 = fig.add_subplot(gs[1, 1])
        a2.semilogy(steps[:k + 1], errs[:k + 1], color=C[1])
        a2.set_xlim(steps.min(), steps.max()); a2.set_ylim(errs.min() * 0.5, max(errs.max() * 2, 1.5))
        a2.set_ylabel("rel. L2 error vs FEA"); a2.set_xlabel(xlabel)
        a2.text(0.98, 0.9, f"{100 * err:.2f}%", transform=a2.transAxes, ha="right", color=C[1], fontsize=11)
        frames.append(fig_to_image(fig))
    plt.close(fig)
    save_gif(frames, path)
    print("wrote", os.path.relpath(path), len(frames), "frames", flush=True)


# ----------------------------------------------------------------------------- PINN
def run_pinn(n_frames=120):
    for case in problems.benchmark_cases():
        fea = fem.solve(case.problem)
        nodes, elems = case.problem.mesh.nodes, case.problem.mesh.elems
        energy = case.pinn_loss == "energy"
        adam, lbfgs = (8000, 0) if energy else (3000, 3000)
        total = adam + int(1.25 * lbfgs)
        marks = set(np.unique(np.round(np.logspace(0, np.log10(total), n_frames)).astype(int) - 1).tolist())
        snaps = []

        def cb(step, model, lval):
            if step in marks:
                u = pinn.predict(model, nodes)
                snaps.append((step + 1, u, abs(lval), fem.relative_l2(u, fea.u)))

        pinn.train_pinn(case, seed=0, adam_steps=adam, lbfgs_steps=lbfgs, loss=case.pinn_loss,
                        callback=cb, verbose=False)
        label = "|potential energy| (Adam)" if energy else "residual loss (Adam → L-BFGS)"
        render_mesh_gif(nodes, elems, fea.u, snaps, f"PINN · {case.name} ({case.pinn_loss} loss)",
                        os.path.join(OUT, f"pinn_{case.name}.gif"), xlabel="step", curve_label=label)


# ----------------------------------------------------------------------------- MGN
def run_mgn(epochs=300):
    train = pool.build_pool(pool.training_problems())
    tests = {p.name: p for p in pool.test_problems()}
    chosen = ["hole_a0.25_s1,1_40x20", "beam_end_shear_L4_D1_48x12", "plate_bottom_lateral_1x1_n20"]
    samples = {n: pool.make_sample(tests[n]) for n in chosen}
    snaps = {n: [] for n in chosen}

    def cb(epoch, model, loss):
        with torch.no_grad():
            for n, s in samples.items():
                u = mgn.predict_displacement(model.cpu(), pool.collate([s]))[0].numpy()
                snaps[n].append((epoch, u, loss, fem.relative_l2(u, s.u_fea.numpy())))

    mgn.train_mgn(train, seed=0, epochs=epochs, hidden=64, n_layers=12, norm="layer", callback=cb, verbose=False)
    for n in chosen:
        m = tests[n].mesh
        render_mesh_gif(m.nodes, m.elems, samples[n].u_fea.numpy(), snaps[n][::2] + [snaps[n][-1]],
                        f"MGN (192-mesh pool) · held-out {n}", os.path.join(OUT, f"mgn_{n.replace(',', '_')}.gif"),
                        xlabel="epoch", curve_label="training loss (pool)")


# ----------------------------------------------------------------------------- FSI
def run_fsi(epochs=250, warm_frac=0.3):
    train = [fsi.make_fsi_sample(c) for c in fsi.training_configs()]
    cfg = fsi.test_configs()[0]
    s = fsi.make_fsi_sample(cfg)
    P = fsi.build_problem(cfg)
    snaps = []

    def cb(epoch, model, h):
        with torch.no_grad():
            xf, us, _ = fsi.fsi_forward(model.cpu(), s)
        e = fsi.fsi_errors(model, s)
        snaps.append((epoch, xf.numpy(), us.numpy(), h, e))

    fsi.train_fsi(train, epochs=epochs, hidden=64, n_layers=12, warm_start_frac=warm_frac, callback=cb, verbose=False)
    render_fsi_gif(P, s, snaps[::2] + [snaps[-1]], int(warm_frac * epochs), os.path.join(OUT, f"fsi_{cfg.name}.gif"))


def render_fsi_gif(P, s, snaps, warm_epochs, path):
    m = P.mesh
    nodes, elems = m.nodes, m.elems
    fl_el, so_el = elems[~P.solid], elems[P.solid]
    v_ref = np.linalg.norm(s.xf.numpy()[:, :2], axis=1)
    u_ref = s.us.numpy()
    so_nodes = np.unique(so_el)
    magn = nice_magnification(u_ref[so_nodes], P.cfg.flap_h, frac=0.25)
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], _tri(fl_el))
    ep = np.array([t[0] for t in snaps])
    E = {k: np.maximum([t[4][k] for t in snaps], 1e-4) for k in ("velocity", "pressure", "solid_disp")}
    frames = []
    fig = plt.figure(figsize=(12, 5.2), dpi=85)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], hspace=0.4, wspace=0.22)
    for k, (epoch, xf, us, h, e) in enumerate(snaps):
        fig.clf()
        for row, (vals, ttl) in enumerate([(np.linalg.norm(xf[:, :2], axis=1), "GNN"), (v_ref, "FEA")]):
            ax = fig.add_subplot(gs[row, 0])
            tc = ax.tripcolor(tri, vals, shading="gouraud", cmap="viridis", vmin=0, vmax=v_ref.max())
            flap_u = us if ttl == "GNN" else u_ref
            ax.add_collection(PolyCollection((nodes + magn * flap_u)[so_el], facecolors=C[1], edgecolors="#7a2e10",
                                             linewidths=0.4))
            ax.add_collection(PolyCollection(nodes[so_el], facecolors="none", edgecolors="k", linewidths=0.7,
                                             linestyles=(0, (3, 2))))
            ax.set_xlim(0, P.cfg.length); ax.set_ylim(0, P.cfg.height); ax.set_aspect("equal"); ax.grid(False)
            fig.colorbar(tc, ax=ax, shrink=0.85, label="|v|")
            stage = "fluid-only warm start" if epoch <= warm_epochs else "coupled"
            ax.set_title((f"FSI GNN · epoch {epoch} ({stage}) · flap deflection ×{magn:g}, dashed = undeformed"
                          if ttl == "GNN" else "FEA reference (Stokes + elastic flap)"), loc="left", fontsize=9.5,
                         color=INK)
        a = fig.add_subplot(gs[:, 1])
        a.axvspan(0, warm_epochs, color=GRID, lw=0)
        for i, (key, lab) in enumerate([("velocity", "velocity"), ("pressure", "pressure"), ("solid_disp", "flap disp.")]):
            a.semilogy(ep[:k + 1], E[key][:k + 1], color=C[i], label=f"{lab}: {100 * e[key]:.1f}%")
        a.set_xlim(0, ep.max()); a.set_ylim(min(v.min() for v in E.values()) * 0.5, 3)
        a.set_xlabel("epoch"); a.set_ylabel("rel. L2 error vs FEA (unseen flap)")
        a.legend(frameon=False, loc="upper right"); a.text(3, a.get_ylim()[0] * 1.5, "warm start", color=MUTED)
        frames.append(fig_to_image(fig))
    plt.close(fig)
    save_gif(frames, path)
    print("wrote", os.path.relpath(path), len(frames), "frames", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    {"pinn": run_pinn, "mgn": run_mgn, "fsi": run_fsi}[sys.argv[1]]()
