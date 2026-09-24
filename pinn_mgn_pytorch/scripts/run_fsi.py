"""Train the differentiable FSI GNN (fluid-only warm start -> coupled) and evaluate on unseen flaps.

    python scripts/run_fsi.py [--epochs 200] [--norm layer] [--no-warm-start]
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from elastic_sim import fsi  # noqa: E402

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--norm", default="layer", choices=["layer", "batch", "none"])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--no-warm-start", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    train = [fsi.make_fsi_sample(c) for c in fsi.training_configs()]
    test = [fsi.make_fsi_sample(c) for c in fsi.test_configs()]
    frac = 0.0 if args.no_warm_start else 0.3
    model, info = fsi.train_fsi(train, epochs=args.epochs, norm=args.norm, hidden=args.hidden,
                                n_layers=args.layers, warm_start_frac=frac, device=args.device)
    tag = f"{args.norm}{'_nowarm' if args.no_warm_start else ''}"
    torch.save(model.state_dict(), os.path.join(OUT, f"fsi_{tag}.pt"))
    res = {s.meta["name"]: dict(fsi.fsi_errors(model, s), n_nodes=s.n_nodes) for s in test}
    tr = [fsi.fsi_errors(model, s) for s in train]
    for k, v in res.items():
        print(f"{k:32s} " + " ".join(f"{m}={v[m]:.4f}" for m in ("velocity", "pressure", "solid_disp")))
    out = dict(config=vars(args), n_train=len(train), train_time_s=info["train_time"],
               train_median={m: float(np.median([t[m] for t in tr])) for m in tr[0]},
               test=res, history=info["history"])
    with open(os.path.join(OUT, f"fsi_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
