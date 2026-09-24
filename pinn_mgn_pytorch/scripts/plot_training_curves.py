"""Plot current training curves from the run logs / saved histories (safe to run while training)."""
import glob, json, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

LOGS = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "training_curves.png")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e6e5df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2})
fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

# 1. PINN (strong-form cases; energy-loss cases have a signed objective so are not log-plottable)
a = ax[0, 0]; runs = {}
for line in open(os.path.join(LOGS, "pinn_bench2.log")):
    m = re.search(r"\[(\S+) seed=(\d)\] adam\s+(\d+) loss=(\S+)", line)
    if m: runs.setdefault((m[1], m[2]), []).append((int(m[3]), float(m[4])))
cases = sorted({c for c, _ in runs if c not in ("cantilever_udl", "square_gravity")})
for i, case in enumerate(cases):
    for (c, sd), pts in runs.items():
        if c == case:
            x, y = zip(*pts); a.semilogy(x, y, color=C[i], alpha=1 if sd == "0" else 0.45,
                                         label=case if sd == "0" else None)
a.set_title("PINN, strong-form loss (Adam + cosine LR; 3 seeds each)", loc="left", color=INK)
a.set_xlabel("Adam step"); a.set_ylabel("non-dim. residual loss"); a.legend(frameon=False)

# 2. MGN ensemble members
a = ax[0, 1]
mg = {}
for line in open(os.path.join(LOGS, "mgn_bench_nocurr.log")):
    m = re.search(r"seed=(\d)\] epoch\s+(\d+) graphs=\s*(\d+).*pattern=(\S+) amp=(\S+) phys=(\S+)", line)
    if m: mg.setdefault(m[1], []).append([int(m[2]), float(m[4]), float(m[5]), float(m[6])])
for sd, rows in mg.items():
    r = np.array(rows); al = 1 if sd == "0" else 0.45
    a.semilogy(r[:, 0], r[:, 1], color=C[0], alpha=al, label="pattern MSE" if sd == "0" else None)
    a.semilogy(r[:, 0], r[:, 2], color=C[1], alpha=al, label="log-amplitude MSE" if sd == "0" else None)
    a.semilogy(r[:, 0], r[:, 3], color=C[2], alpha=al, label="FE residual (rel.)" if sd == "0" else None)
a.set_title("MGN ensemble, 192-mesh pool (faded = seeds 1–2)", loc="left", color=INK)
a.set_xlabel("epoch"); a.set_ylabel("loss term"); a.legend(frameon=False)

# 3. FSI: warm start vs from scratch
a = ax[1, 0]
h = json.load(open(glob.glob(os.path.join(os.path.dirname(__file__), "..", "results", "fsi_layer.json"))[0]))["history"]
ep = [r["epoch"] for r in h]
a.semilogy(ep, [r["data_f"] for r in h], color=C[0], label="fluid error (warm start)")
a.semilogy(ep, [r["data_s"] for r in h], color=C[1], label="flap pattern error (warm start)")
nw = [re.search(r"epoch\s+(\d+).*data_f=(\S+) data_s=(\S+)", l) for l in open(os.path.join(LOGS, "fsi_nowarm.log"))]
nw = np.array([[int(m[1]), float(m[2]), float(m[3])] for m in nw if m])
if len(nw):
    a.semilogy(nw[:, 0], nw[:, 1], color=C[0], ls=":", label="fluid error (from scratch)")
    a.semilogy(nw[:, 0], nw[:, 2], color=C[1], ls=":", label="flap pattern error (from scratch)")
a.axvspan(0, 75, color=GRID, alpha=0.6, lw=0); a.text(3, a.get_ylim()[0] * 2, "fluid-only warm start", color=MUTED)
a.set_title("FSI GNN: fluid-only warm start → coupled", loc="left", color=INK)
a.set_xlabel("epoch"); a.set_ylabel("MSE (normalised)"); a.legend(frameon=False, fontsize=8)

# 4. MATLAB PI-MGN (25-node plate)
a = ax[1, 1]
mdir = os.path.join(os.path.dirname(__file__), "..", "..", "pi_fem_mgn_successful_one_batch_test_Jan_30")
for i, (f, lab) in enumerate([("results_cosine_ensemble_layer.mat", "LayerNorm"),
                              ("results_cosine_ensemble_batch.mat", "BatchNorm (encoders)"),
                              ("results_cosine_ensemble_layer_warmstart.mat", "warm start (pretrained)"),
                              ("results_cosine_ensemble_layer_rawinputs.mat", "LayerNorm, no input normalisation")]):
    p = os.path.join(mdir, f)
    if not os.path.exists(p): continue
    hist = sio.loadmat(p)["hist"].ravel()
    for k, hm in enumerate(hist):
        a.semilogy(np.clip(1 - hm[:, 1], 1e-10, None), color=C[i], alpha=1 if k == 0 else 0.45,
                   label=lab if k == 0 else None)
a.set_title("MATLAB PI-MGN, FE-residual loss, cosine LR (faded = other members)", loc="left", color=INK)
a.set_xlabel("epoch"); a.set_ylabel("1 − R² vs FEM"); a.legend(frameon=False)
fig.savefig(OUT, dpi=130)
print(os.path.abspath(OUT))
