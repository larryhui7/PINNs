"""Differentiable GNN solver for a steady fluid-solid interaction (FSI) problem.

Setup (sketch from "4 ideas next steps", page 3: flexible fin on the floor of a channel):
  * Fluid: steady Stokes flow in a channel [0, Lc] x [0, H], parabolic inflow at x = 0,
    no-slip on the walls and on the flap, traction-free outlet. Q1-Q1 velocity/pressure with
    Brezzi-Pitkaranta pressure stabilisation (all unknowns are nodal -> natural for a GNN).
  * Solid: plane-stress linear-elastic flap clamped to the channel floor.
  * Coupling (one-way, small deformation): the fluid traction sigma_f . n on the flap surface
    is a *linear* operator C applied to the fluid unknowns, F_s = C x_f.

The GNN runs on the joint fluid+solid graph and outputs [vx, vy, p, ux, uy] per node.
Loss = supervised field-pattern MSE + physics residuals of BOTH discrete systems,
  ||K_f x_f - F_f|| / ||K_f x_d||   and   ||K_s u - C x_f|| / ||C x_f||,
where the solid residual is evaluated with the *predicted* fluid state, so gradients from the
solid equations flow back through C into the fluid prediction (end-to-end differentiable).
Output normalisation: fluid unknowns are predicted in reference units (v / U_in, p / p_ref with
p_ref = 12 mu U_in Lc / H^2, the Poiseuille pressure drop); the solid displacement is an
RMS-normalised pattern times a pooled amplitude head predicting log10(rms(u) E_s / sum|F_s|),
where F_s = C x_f is the traction of the *predicted* flow.
"""
import math
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch import nn

from . import fem, meshes
from .mgn import MeshGraphNet, segment_sum

FSI_NODE_FEATURES = 15
FSI_EDGE_FEATURES = 4


# ----------------------------------------------------------------------------- geometry
@dataclass
class FSIConfig:
    length: float = 4.0
    height: float = 1.0
    flap_x0: float = 1.0
    flap_w: float = 0.1
    flap_h: float = 0.5
    ny: int = 20
    mu: float = 1.0
    u_in: float = 1.0          # mean inflow velocity
    E_s: float = 5.0e5
    nu_s: float = 0.3
    beta: float = 0.05         # pressure stabilisation

    @property
    def name(self):
        return f"fsi_h{self.flap_h:g}_w{self.flap_w:g}_x{self.flap_x0:g}_ny{self.ny}"


def build_mesh(cfg):
    h = cfg.height / cfg.ny
    nx = int(round(cfg.length / h))
    m = meshes.rectangle(cfg.length, cfg.height, nx, cfg.ny, name="channel")
    c = m.nodes[m.elems].mean(1)
    solid = (c[:, 0] > cfg.flap_x0) & (c[:, 0] < cfg.flap_x0 + cfg.flap_w) & (c[:, 1] < cfg.flap_h)
    assert solid.sum() > 0, "flap not resolved by the mesh"
    return m, solid, h


# ----------------------------------------------------------------------------- fluid FEM
def assemble_stokes(nodes, elems, mu, beta, h):
    """Q1-Q1 Stokes with Brezzi-Pitkaranta stabilisation; DOFs (vx, vy, p) = 3*node + {0,1,2}."""
    n_el = len(elems)
    Ke = np.zeros((n_el, 12, 12))
    for xi, wi in zip(fem.GP, fem.GW):
        for eta, wj in zip(fem.GP, fem.GW):
            N, dNdx, detJ = fem.q4_gradients(nodes, elems, xi, eta)
            w = (detJ * wi * wj)[:, None, None]
            lap = np.einsum("eai,ebi->eab", dNdx, dNdx)
            for c in range(2):                                    # viscous block, per component
                Ke[:, c::3, c::3] += mu * lap * w
            for c in range(2):                                    # B: -int q dv_c/dx_c
                Bc = -np.einsum("a,eb->eab", N, dNdx[:, :, c]) * w
                Ke[:, 2::3, c::3] += Bc
                Ke[:, c::3, 2::3] += np.transpose(Bc, (0, 2, 1))
            Ke[:, 2::3, 2::3] -= beta * h ** 2 / mu * lap * w      # pressure stabilisation
    dofs = np.stack([3 * elems + k for k in range(3)], 2).reshape(n_el, 12)
    rows = np.repeat(dofs, 12, axis=1).ravel()
    cols = np.tile(dofs, (1, 12)).ravel()
    n = 3 * len(nodes)
    return sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()


def traction_operator(nodes, fluid_elems, iface_edges, solid_elem_centroids, mu):
    """Sparse C (2n x 3n): nodal forces on the solid from fluid stress sigma = -pI + mu(grad v + grad v^T).

    For each interface edge, the adjacent fluid element supplies sigma at two Gauss points of the edge;
    the normal n points out of the solid (into the fluid)."""
    rows, cols, vals = [], [], []
    fl_nodes = nodes[fluid_elems]
    fl_cent = fl_nodes.mean(1)
    for a, b in iface_edges:
        pa, pb = nodes[a], nodes[b]
        mid = 0.5 * (pa + pb)
        # owning fluid element = fluid element whose centroid is nearest the edge midpoint
        e = int(np.argmin(((fl_cent - mid) ** 2).sum(1)))
        el = fluid_elems[e]
        t = pb - pa
        length = np.linalg.norm(t)
        n = np.array([t[1], -t[0]]) / length
        if np.dot(fl_cent[e] - mid, n) < 0:
            n = -n                                                # point into the fluid
        # local (xi, eta) of the edge endpoints inside the axis-aligned fluid element
        lo, hi = fl_nodes[e].min(0), fl_nodes[e].max(0)
        loc = lambda p: 2 * (p - lo) / (hi - lo) - 1
        la, lb = loc(pa), loc(pb)
        for s, w in zip(fem.GP, fem.GW):
            Na_e, Nb_e = 0.5 * (1 - s), 0.5 * (1 + s)
            xi, eta = Na_e * la + Nb_e * lb
            N, dNdx, _ = fem.q4_gradients(nodes, el[None], xi, eta)
            dNdx = dNdx[0]
            jac = 0.5 * length * w
            for nd_edge, Ne in ((a, Na_e), (b, Nb_e)):
                for i in range(2):                                # force component on solid node
                    for k, nd in enumerate(el):
                        # pressure part: -p n_i
                        rows.append(2 * nd_edge + i); cols.append(3 * nd + 2)
                        vals.append(-N[k] * n[i] * Ne * jac)
                        # viscous part: mu (dv_i/dx_j + dv_j/dx_i) n_j
                        for j in range(2):
                            rows.append(2 * nd_edge + i); cols.append(3 * nd + i)
                            vals.append(mu * dNdx[k, j] * n[j] * Ne * jac)
                            rows.append(2 * nd_edge + i); cols.append(3 * nd + j)
                            vals.append(mu * dNdx[k, i] * n[j] * Ne * jac)
    n_nodes = len(nodes)
    return sp.coo_matrix((vals, (rows, cols)), shape=(2 * n_nodes, 3 * n_nodes)).tocsr()


def interface_edges(mesh, solid):
    """Element sides shared by a solid and a fluid element."""
    side = {}
    for e, el in enumerate(mesh.elems):
        for k in range(4):
            key = tuple(sorted((el[k], el[(k + 1) % 4])))
            side.setdefault(key, []).append(e)
    return [k for k, es in side.items() if len(es) == 2 and solid[es[0]] != solid[es[1]]]


@dataclass
class FSIProblem:
    cfg: FSIConfig
    mesh: object
    solid: np.ndarray
    Kf: sp.csr_matrix
    Ks: sp.csr_matrix
    C: sp.csr_matrix
    f_fixed: np.ndarray          # fluid DOFs with Dirichlet data
    f_vals: np.ndarray
    s_fixed: np.ndarray          # solid DOFs fixed (clamp + inactive nodes)
    node_type: np.ndarray        # 0 fluid, 1 solid, 2 interface
    flags: dict


def build_problem(cfg):
    mesh, solid, h = build_mesh(cfg)
    x, y = mesh.nodes.T
    n = mesh.n_nodes
    fl_el, so_el = mesh.elems[~solid], mesh.elems[solid]
    in_fluid = np.zeros(n, bool); in_fluid[fl_el.ravel()] = True
    in_solid = np.zeros(n, bool); in_solid[so_el.ravel()] = True
    node_type = np.where(in_fluid & in_solid, 2, np.where(in_solid, 1, 0))
    eps = 1e-9
    inlet, outlet = x < eps, x > cfg.length - eps
    wall = (y < eps) | (y > cfg.height - eps)
    clamp = in_solid & (y < eps)

    Kf = assemble_stokes(mesh.nodes, fl_el, cfg.mu, cfg.beta, h)
    Ks = fem.assemble_stiffness(mesh.nodes, so_el, fem.d_matrix(cfg.E_s, cfg.nu_s))
    iface = interface_edges(mesh, solid)
    C = traction_operator(mesh.nodes, fl_el, iface, None, cfg.mu)

    # fluid Dirichlet: no-slip on walls + flap (interface), inflow profile, everything on solid-only nodes
    u_max = 1.5 * cfg.u_in
    prof = 4 * u_max * y * (cfg.height - y) / cfg.height ** 2
    fixed, vals = [], []
    for i in range(n):
        if node_type[i] == 1:                                    # solid-only: fluid DOFs inactive
            fixed += [3 * i, 3 * i + 1, 3 * i + 2]; vals += [0, 0, 0]
        elif inlet[i]:
            fixed += [3 * i, 3 * i + 1]; vals += [prof[i], 0]
        elif wall[i] or node_type[i] == 2:
            fixed += [3 * i, 3 * i + 1]; vals += [0, 0]
    s_fixed = []
    for i in range(n):
        if node_type[i] == 0 or clamp[i]:
            s_fixed += [2 * i, 2 * i + 1]
    order = np.argsort(fixed)
    flags = dict(inlet=inlet, outlet=outlet, wall=wall, clamp=clamp, profile=prof / u_max, h=h)
    return FSIProblem(cfg, mesh, solid, Kf, Ks, C, np.asarray(fixed)[order], np.asarray(vals, float)[order],
                      np.asarray(s_fixed), node_type, flags)


def solve_fsi(P):
    """Reference solution: fluid saddle-point solve, then solid solve with F_s = C x_f."""
    t0 = time.perf_counter()
    nf = P.Kf.shape[0]
    xf = np.zeros(nf); xf[P.f_fixed] = P.f_vals
    free = np.setdiff1d(np.arange(nf), P.f_fixed)
    rhs = -P.Kf[free][:, P.f_fixed] @ P.f_vals
    xf[free] = spla.spsolve(P.Kf[free][:, free].tocsc(), rhs)
    t1 = time.perf_counter()
    Fs = P.C @ xf
    ns = P.Ks.shape[0]
    us = np.zeros(ns)
    sfree = np.setdiff1d(np.arange(ns), P.s_fixed)
    us[sfree] = spla.spsolve(P.Ks[sfree][:, sfree].tocsc(), Fs[sfree])
    t2 = time.perf_counter()
    return xf.reshape(-1, 3), us.reshape(-1, 2), dict(t_fluid=t1 - t0, t_solid=t2 - t1)


# ----------------------------------------------------------------------------- graph sample
def _coo(M, device="cpu"):
    M = M.tocoo()
    return (torch.as_tensor(np.stack([M.row, M.col]), dtype=torch.long),
            torch.as_tensor(M.data, dtype=torch.float32))


@dataclass
class FSISample:
    node_x: torch.Tensor
    edge_x: torch.Tensor
    senders: torch.Tensor
    receivers: torch.Tensor
    f_free: torch.Tensor         # (n, 3)
    f_dir: torch.Tensor          # (n, 3)
    s_free: torch.Tensor         # (n, 2)
    Kf: tuple
    Ks: tuple
    C: tuple
    p_ref: float
    u_in: float
    E_s: float
    xf: torch.Tensor             # (n, 3) reference fluid
    us: torch.Tensor             # (n, 2) reference solid
    meta: dict

    @property
    def n_nodes(self):
        return self.node_x.shape[0]


def make_fsi_sample(cfg):
    P = build_problem(cfg)
    xf, us, times = solve_fsi(P)
    m, cfgc = P.mesh, P.cfg
    n = m.n_nodes
    x, y = m.nodes.T
    fl = P.flags
    # signed-ish distance to the flap rectangle
    dx = np.maximum(np.maximum(cfgc.flap_x0 - x, x - cfgc.flap_x0 - cfgc.flap_w), 0)
    dy = np.maximum(y - cfgc.flap_h, 0)
    dist = np.hypot(dx, dy) / cfgc.height
    onehot = np.eye(3)[P.node_type]
    glob = np.array([cfgc.flap_h / cfgc.height, cfgc.flap_w / cfgc.height, cfgc.flap_x0 / cfgc.length,
                     math.log10(n) / 4])
    node_x = np.concatenate([
        np.stack([(x - cfgc.length / 2) / cfgc.height, (y - cfgc.height / 2) / cfgc.height], 1),
        onehot, np.stack([fl["inlet"], fl["outlet"], fl["wall"], fl["clamp"], fl["profile"] * fl["inlet"],
                          dist], 1).astype(float),
        np.tile(glob, (n, 1))], 1)
    assert node_x.shape[1] == FSI_NODE_FEATURES
    e = m.edges()
    d = m.nodes[e[0]] - m.nodes[e[1]]
    same = (P.node_type[e[0]] == P.node_type[e[1]]).astype(float)
    edge_x = np.concatenate([d / cfgc.height, np.linalg.norm(d, axis=1, keepdims=True) / cfgc.height,
                             same[:, None]], 1)
    f_free = np.ones(3 * n); f_free[P.f_fixed] = 0
    f_dir = np.zeros(3 * n); f_dir[P.f_fixed] = P.f_vals
    s_free = np.ones(2 * n); s_free[P.s_fixed] = 0
    t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt)
    p_ref = 12 * cfgc.mu * cfgc.u_in * cfgc.length / cfgc.height ** 2
    meta = dict(name=cfgc.name, n_nodes=n, flap_h=cfgc.flap_h, flap_w=cfgc.flap_w, flap_x0=cfgc.flap_x0,
                ny=cfgc.ny, n_interface_edges=int((P.C.getnnz(1) > 0).sum() // 2), **times)
    return FSISample(t(node_x), t(edge_x), t(e[0], torch.long), t(e[1], torch.long),
                     t(f_free.reshape(-1, 3)), t(f_dir.reshape(-1, 3)), t(s_free.reshape(-1, 2)),
                     _coo(P.Kf), _coo(P.Ks), _coo(P.C), p_ref, cfgc.u_in, cfgc.E_s, t(xf), t(us), meta)


def _mv(coo, v, n_out):
    idx, val = coo
    return torch.zeros(n_out, dtype=v.dtype, device=v.device).index_add_(0, idx[0], val * v[idx[1]])


def to_device(s, device):
    if device == "cpu":
        return s
    mv = lambda c: (c[0].to(device), c[1].to(device))
    return FSISample(s.node_x.to(device), s.edge_x.to(device), s.senders.to(device), s.receivers.to(device),
                     s.f_free.to(device), s.f_dir.to(device), s.s_free.to(device), mv(s.Kf), mv(s.Ks), mv(s.C),
                     s.p_ref, s.u_in, s.E_s, s.xf.to(device), s.us.to(device), s.meta)


def fsi_forward(model, s):
    """GNN -> fluid state (reference units) -> traction F_s = C x_f -> solid pattern * learned amplitude."""
    b = dict(node_x=s.node_x, edge_x=s.edge_x, senders=s.senders, receivers=s.receivers,
             graph_id=torch.zeros(s.n_nodes, dtype=torch.long, device=s.node_x.device), n_graphs=1)
    out = model(b)
    n = s.n_nodes
    chan = torch.tensor([s.u_in, s.u_in, s.p_ref], device=out.device)
    xf = s.f_dir + out[:, :3] * chan * s.f_free
    Fs = _mv(s.C, xf.reshape(-1), 2 * n) * s.s_free.reshape(-1)
    sf = s.s_free.reshape(-1)
    ws = (out[:, 3:] * s.s_free).reshape(-1)
    ws = ws / ws.pow(2).sum().div(sf.sum()).sqrt().clamp(min=1e-12)
    load = Fs.reshape(-1, 2).norm(dim=1).sum() / s.E_s
    log_amp = model.last_log_amp[0] * model.amp_stats[1] + model.amp_stats[0]
    us = 10 ** log_amp * load * ws
    return xf, us.reshape(-1, 2), (ws, Fs, load)


def fsi_losses(model, s, solid_weight=1.0, phys_weight=0.1):
    xf, us, (ws, Fs, load) = fsi_forward(model, s)
    n = s.n_nodes
    ff, sf = s.f_free.reshape(-1), s.s_free.reshape(-1)
    chan = torch.tensor([s.u_in, s.u_in, s.p_ref], device=xf.device)
    data_f = ((((xf - s.xf) / chan).reshape(-1) * ff) ** 2).sum() / ff.sum()
    ts = s.us.reshape(-1) * sf
    rms_t = ts.pow(2).sum().div(sf.sum()).sqrt().clamp(min=1e-30)
    data_s = ((ws - ts / rms_t) ** 2 * sf).sum() / sf.sum()
    load_ref = (_mv(s.C, s.xf.reshape(-1), 2 * n) * sf).reshape(-1, 2).norm(dim=1).sum() / s.E_s
    log_t = (torch.log10(rms_t / load_ref) - model.amp_stats[0]) / model.amp_stats[1]
    amp = (model.last_log_amp[0] - log_t) ** 2
    # physics residuals of both discrete systems (solid one uses the *predicted* fluid traction)
    r0 = _mv(s.Kf, s.f_dir.reshape(-1), 3 * n) * ff
    rf = (_mv(s.Kf, xf.reshape(-1), 3 * n) * ff).pow(2).sum() / r0.pow(2).sum().clamp(min=1e-30)
    rs = ((_mv(s.Ks, us.reshape(-1), 2 * n) - Fs) * sf).pow(2).sum() / Fs.pow(2).sum().clamp(min=1e-30)
    # log(1 + r): the solid residual starts ~1e4-1e5 when coupling switches on; a raw quadratic term
    # would push large gradients back through C and destroy the warm-started fluid solution.
    loss = data_f + phys_weight * torch.log1p(rf) + solid_weight * (data_s + amp + phys_weight * torch.log1p(rs))
    return loss, dict(data_f=data_f.item(), data_s=data_s.item(), amp=amp.item(), res_f=rf.item(), res_s=rs.item())


def fsi_errors(model, s):
    with torch.no_grad():
        xf, us, _ = fsi_forward(model, s)
    ref_v, ref_p = s.xf[:, :2], s.xf[:, 2]
    fl = s.f_free[:, 2] > 0                                      # nodes carrying fluid pressure
    rel = lambda a, b: float((a - b).norm() / b.norm())
    return dict(velocity=rel(xf[:, :2], ref_v), pressure=rel(xf[fl, 2], ref_p[fl]),
                solid_disp=rel(us, s.us))


def train_fsi(pool, epochs=200, lr=1e-3, lr_min=1e-6, seed=0, warm_start_frac=0.3, device="cpu",
              verbose=True, callback=None, **model_kw):
    """Gradual training with a warm start: the first `warm_start_frac` of the epochs train the fluid
    branch only (solid_weight=0); the network is then warm-started into fully coupled training."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MeshGraphNet(node_features=FSI_NODE_FEATURES, edge_features=FSI_EDGE_FEATURES, out_dim=5, **model_kw)
    logs = []
    for s in pool:
        model.node_norm.accumulate(s.node_x)
        model.edge_norm.accumulate(s.edge_x)
        n = s.n_nodes
        sf = s.s_free.reshape(-1)
        rms = (s.us.reshape(-1) * sf).pow(2).sum().div(sf.sum()).sqrt()
        load = (_mv(s.C, s.xf.reshape(-1), 2 * n) * sf).reshape(-1, 2).norm(dim=1).sum() / s.E_s
        logs.append(float(torch.log10(rms / load)))
    model.amp_stats.copy_(torch.tensor([np.mean(logs), max(np.std(logs), 1e-3)]))
    model.to(device)
    dev_pool = [to_device(s, device) for s in pool]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(pool), eta_min=lr_min)
    hist, t0 = [], time.perf_counter()
    for ep in range(epochs):
        stage = "fluid-only warm start" if ep < warm_start_frac * epochs else "coupled"
        sw = 0.0 if stage != "coupled" else 1.0
        model.train()
        parts_ep = []
        for i in rng.permutation(len(dev_pool)):
            loss, parts = fsi_losses(model, dev_pool[i], solid_weight=sw)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            parts_ep.append(parts)
        h = {k: float(np.mean([p[k] for p in parts_ep])) for k in parts_ep[0]}
        h.update(epoch=ep + 1, stage=stage)
        hist.append(h)
        if callback is not None:
            model.eval()
            callback(ep + 1, model, h)
        if verbose and ((ep + 1) % 20 == 0 or ep == 0):
            print(f"  [fsi] epoch {ep + 1:4d} {stage:22s} data_f={h['data_f']:.2e} data_s={h['data_s']:.2e} amp={h['amp']:.2e} "
                  f"res_f={h['res_f']:.2e} res_s={h['res_s']:.2e}", flush=True)
    model.eval()
    return model.cpu(), dict(history=hist, train_time=time.perf_counter() - t0)


def training_configs():
    cfgs = []
    for ny in (16, 20):
        h_el = 1.0 / ny
        for fh in (0.3, 0.4, 0.5, 0.6):
            for fw_el in (2, 3):
                for x0 in (1.0, 1.5):
                    cfgs.append(FSIConfig(flap_x0=x0, flap_w=fw_el * h_el, flap_h=fh, ny=ny))
    return cfgs


def test_configs():
    return [FSIConfig(flap_x0=1.25, flap_w=0.1, flap_h=0.45, ny=20),     # unseen height/position
            FSIConfig(flap_x0=1.0, flap_w=0.125, flap_h=0.55, ny=24),    # unseen, finer mesh
            FSIConfig(flap_x0=1.2, flap_w=0.15, flap_h=0.35, ny=20)]
