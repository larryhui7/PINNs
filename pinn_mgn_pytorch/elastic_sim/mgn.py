"""Physics-informed MeshGraphNet for linear elasticity (PyTorch port of dlnetworkMGN/).

Architecture mirrors meshGraphNetwork.m: node/edge MLP encoders, N message-passing
blocks (edge update from [e, h_s, h_r], node update from [h, sum_e]) with residual
connections, and an MLP decoder. Normalisation after each MLP is configurable:
"layer" (LayerNorm, as in the MATLAB code), "batch" (BatchNorm over nodes/edges) or "none".

Prediction pipeline for one mesh (output normalisation):
  1. MGN predicts a displacement *pattern* w on the free DOFs, RMS-normalised per graph, so one
     network covers beams, plates and holes whose magnitudes differ by orders of magnitude.
  2. A pooled "amplitude head" predicts the dimensionless log-amplitude
     log10( rms(u) * E / sum|F_i| ), standardised by a Normalizer; u = u_d + amplitude * w.
     (amplitude="ritz" instead uses the Rayleigh-Ritz amplitude w.(F - K u_d) / (w.K w).)
Training loss = pattern MSE + log-amplitude MSE vs FEA (supervised)
              + lambda * relative FE residual ||K u - F||^2/||F||^2 on the free DOFs
              (physics-informed, as in global_fem_large_mesh_test_loss_adjust.m).
"""
import copy
import math
import time

import numpy as np
import torch
from torch import nn

from .pool import EDGE_FEATURES, NODE_FEATURES, collate


class Normalizer(nn.Module):
    """Online feature standardisation (accumulates mean/std over the first pass of the pool)."""

    def __init__(self, size, max_accum=10 ** 7, eps=1e-8):
        super().__init__()
        self.register_buffer("count", torch.zeros(()))
        self.register_buffer("sum", torch.zeros(size))
        self.register_buffer("sumsq", torch.zeros(size))
        self.max_accum, self.eps = max_accum, eps

    def accumulate(self, x):
        if self.count < self.max_accum:
            self.count += x.shape[0]
            self.sum += x.sum(0)
            self.sumsq += (x ** 2).sum(0)

    def forward(self, x):
        mean = self.sum / self.count.clamp(min=1)
        std = (self.sumsq / self.count.clamp(min=1) - mean ** 2).clamp(min=0).sqrt()
        return (x - mean) / (std + self.eps)


def mlp(d_in, d_hidden, d_out, n_hidden=2, norm="layer"):
    layers, d = [], d_in
    for _ in range(n_hidden):
        layers += [nn.Linear(d, d_hidden), nn.SiLU()]      # SiLU == MATLAB "swish"
        d = d_hidden
    layers.append(nn.Linear(d, d_out))
    if norm == "layer":
        layers.append(nn.LayerNorm(d_out))
    elif norm == "batch":
        layers.append(nn.BatchNorm1d(d_out))
    return nn.Sequential(*layers)


class GraphBlock(nn.Module):
    def __init__(self, h, norm, aggregate="sum"):
        super().__init__()
        self.edge_mlp = mlp(3 * h, h, h, norm=norm)
        self.node_mlp = mlp(2 * h, h, h, norm=norm)
        self.aggregate = aggregate

    def forward(self, h, e, snd, rcv):
        e = e + self.edge_mlp(torch.cat([e, h[snd], h[rcv]], 1))
        agg = torch.zeros_like(h).index_add_(0, rcv, e)
        if self.aggregate == "mean":
            deg = torch.zeros(h.shape[0], 1, device=h.device).index_add_(0, rcv, torch.ones_like(e[:, :1]))
            agg = agg / deg.clamp(min=1)
        h = h + self.node_mlp(torch.cat([h, agg], 1))
        return h, e


class MeshGraphNet(nn.Module):
    def __init__(self, hidden=128, n_layers=15, norm="layer", aggregate="sum",
                 node_features=NODE_FEATURES, edge_features=EDGE_FEATURES, out_dim=2):
        super().__init__()
        self.node_norm = Normalizer(node_features)
        self.edge_norm = Normalizer(edge_features)
        self.node_enc = mlp(node_features, hidden, hidden, norm=norm)
        self.edge_enc = mlp(edge_features, hidden, hidden, norm=norm)
        self.blocks = nn.ModuleList([GraphBlock(hidden, norm, aggregate) for _ in range(n_layers)])
        self.decoder = mlp(hidden, hidden, out_dim, norm="none")
        self.amp_head = mlp(hidden, hidden, 1, norm="none")
        self.register_buffer("amp_stats", torch.tensor([0.0, 1.0]))   # mean/std of log10 amplitude
        self.config = dict(hidden=hidden, n_layers=n_layers, norm=norm, aggregate=aggregate,
                           node_features=node_features, edge_features=edge_features, out_dim=out_dim)

    def forward(self, b):
        h = self.node_enc(self.node_norm(b["node_x"]))
        e = self.edge_enc(self.edge_norm(b["edge_x"]))
        for blk in self.blocks:
            h, e = blk(h, e, b["senders"], b["receivers"])
        if "graph_id" in b:
            gid, n = b["graph_id"], b["n_graphs"]
            cnt = torch.zeros(n, 1, device=h.device).index_add_(0, gid, torch.ones_like(h[:, :1]))
            pooled = torch.zeros(n, h.shape[1], device=h.device).index_add_(0, gid, h) / cnt
            self.last_log_amp = self.amp_head(pooled)[:, 0]           # standardised log10 amplitude
        return self.decoder(h)


# ----------------------------------------------------------------------------- physics
def k_matvec(b, u_flat):
    i, j = b["K_idx"]
    return torch.zeros_like(u_flat).index_add_(0, i, b["K_val"] * u_flat[j])


def segment_sum(x, seg, n):
    return torch.zeros(n, dtype=x.dtype, device=x.device).index_add_(0, seg, x)


def ritz_scale(b, w):
    """Energy-optimal amplitude per graph: u = u_d + alpha * w (w vanishes on Dirichlet DOFs)."""
    gid = b["graph_id"].repeat_interleave(2)
    w_flat, ud_flat = w.reshape(-1), b["u_dir"].reshape(-1)
    Kw = k_matvec(b, w_flat)
    rhs = b["F"] - k_matvec(b, ud_flat)
    num = segment_sum(w_flat * rhs, gid, b["n_graphs"])
    den = segment_sum(w_flat * Kw, gid, b["n_graphs"]).clamp(min=1e-30)
    alpha = (num / den)[b["graph_id"]]
    return b["u_dir"] + alpha[:, None] * w


def rms_normalise(b, w):
    free = 1 - b["dir_mask"]
    gid = b["graph_id"]
    ms = segment_sum((w ** 2).sum(1), gid, b["n_graphs"]) / segment_sum(free.sum(1), gid, b["n_graphs"])
    return w / ms.sqrt().clamp(min=1e-12)[gid][:, None]


def predict_displacement(model, b, amplitude="head"):
    w = rms_normalise(b, model(b) * (1 - b["dir_mask"]))
    if amplitude == "ritz":
        return ritz_scale(b, w), w
    log_amp = model.last_log_amp * model.amp_stats[1] + model.amp_stats[0]
    amp = 10 ** log_amp * b["load_scale"]
    return b["u_dir"] + amp[b["graph_id"]][:, None] * w, w


def losses(model, b, phys_weight=0.1):
    u, w = predict_displacement(model, b)
    free = 1 - b["dir_mask"]
    gid = b["graph_id"]
    # RMS-normalised FEA pattern per graph (output normalisation)
    target = (b["u_fea"] - b["u_dir"]) * free
    rms_t = (segment_sum((target ** 2).sum(1), gid, b["n_graphs"]) /
             segment_sum(free.sum(1), gid, b["n_graphs"])).sqrt().clamp(min=1e-30)
    data = (((w - target / rms_t[gid][:, None]) * free) ** 2).sum() / free.sum()
    log_amp_t = (torch.log10(rms_t / b["load_scale"]) - model.amp_stats[0]) / model.amp_stats[1]
    amp = ((model.last_log_amp - log_amp_t) ** 2).mean()
    # physics: relative residual on free DOFs, per graph
    r = (k_matvec(b, u.reshape(-1)) - b["F"]) * free.reshape(-1)
    gid2 = gid.repeat_interleave(2)
    rel = segment_sum(r ** 2, gid2, b["n_graphs"]) / segment_sum((b["F"] * free.reshape(-1)) ** 2, gid2,
                                                                  b["n_graphs"]).clamp(min=1e-30)
    phys = rel.mean()
    return data + amp + phys_weight * phys, dict(data=data.item(), amp=amp.item(), phys=phys.item())


# ----------------------------------------------------------------------------- training
def fit_normalizers(model, pool):
    logs = []
    for s in pool:
        model.node_norm.accumulate(s.node_x)
        model.edge_norm.accumulate(s.edge_x)
        free = 1 - s.dir_mask
        rms = (((s.u_fea - s.u_dir) * free) ** 2).sum().div(free.sum()).sqrt()
        logs.append(float(torch.log10(rms / s.load_scale)))
    model.amp_stats.copy_(torch.tensor([np.mean(logs), max(np.std(logs), 1e-3)]))


def train_mgn(pool, seed=0, epochs=300, batch_graphs=4, lr=1e-3, lr_min=1e-6, phys_weight=0.01,
              warm_start=None, curriculum=None, device="cpu", verbose=True, callback=None, **model_kw):
    """Train one MGN on the pool with Adam + cosine annealing.

    warm_start : a trained MeshGraphNet whose weights initialise this one (transfer / gradual training).
    curriculum : optional list of (epoch_fraction, predicate(meta)) stages; each stage trains on the
                 subset of the pool whose metadata satisfies the predicate, warm-starting from the
                 previous stage (e.g. coarse meshes first, then everything).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MeshGraphNet(**model_kw)
    if warm_start is not None:
        model.load_state_dict(warm_start.state_dict())
    else:
        fit_normalizers(model, pool)
    model.to(device)
    stages = curriculum or [(1.0, lambda m: True)]
    total_epochs = epochs
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps_per_epoch = [math.ceil(sum(pred(s.meta) for s in pool) / batch_graphs) for _, pred in stages]
    total_steps = sum(int(f * total_epochs) * n for (f, _), n in zip(stages, steps_per_epoch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=lr_min)
    history, t0 = [], time.perf_counter()
    epoch_global = 0
    for (frac, pred) in stages:
        subset = [s for s in pool if pred(s.meta)]
        for _ in range(int(frac * total_epochs)):
            model.train()
            order = rng.permutation(len(subset))
            ep_loss, ep_parts = [], []
            for k in range(0, len(order), batch_graphs):
                b = collate([subset[i] for i in order[k:k + batch_graphs]], device)
                loss, parts = losses(model, b, phys_weight)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                ep_loss.append(loss.item())
                ep_parts.append(parts)
            epoch_global += 1
            if callback is not None:
                model.eval()
                callback(epoch_global, model, float(np.mean(ep_loss)))
            history.append(dict(epoch=epoch_global, loss=float(np.mean(ep_loss)), n_graphs=len(subset),
                                lr=sched.get_last_lr()[0],
                                data=float(np.mean([p["data"] for p in ep_parts])),
                                amp=float(np.mean([p["amp"] for p in ep_parts])),
                                phys=float(np.mean([p["phys"] for p in ep_parts]))))
            if verbose and (epoch_global % 25 == 0 or epoch_global == 1):
                h = history[-1]
                print(f"  [mgn seed={seed}] epoch {epoch_global:4d} graphs={len(subset):3d} loss={h['loss']:.3e} "
                      f"pattern={h['data']:.3e} amp={h['amp']:.3e} phys={h['phys']:.3e} lr={h['lr']:.1e}", flush=True)
    model.eval()
    return model.cpu(), dict(history=history, train_time=time.perf_counter() - t0)


class MGNEnsemble:
    """Averages the (Ritz-scaled) displacement of independently seeded MGNs."""

    def __init__(self, models):
        self.models = models

    @torch.no_grad()
    def predict(self, sample, device="cpu"):
        b = collate([sample], device)
        us = torch.stack([predict_displacement(m.to(device).eval(), b)[0] for m in self.models])
        return us.mean(0).cpu().numpy(), us.std(0).cpu().numpy()


def copy_model(m):
    return copy.deepcopy(m)
