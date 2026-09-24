"""Training pool of FEA-labelled graphs with mesh metadata (Gen-FVGN-style data pool).

Each ``Sample`` stores the graph inputs for the MeshGraphNet, the sparse stiffness
system used by the physics loss, the FEA displacement, and a metadata dict
(geometry, BC, load, mesh size, ...) used for curriculum / stratified sampling.
"""
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from . import fem, meshes
from .fem import Problem
from .problems import zero_traction

GEOMETRIES = ["beam", "plate_with_hole", "plate"]
NODE_FEATURES = 24
EDGE_FEATURES = 4


@dataclass
class Sample:
    node_x: torch.Tensor        # (n, NODE_FEATURES)
    edge_x: torch.Tensor        # (e, EDGE_FEATURES)
    senders: torch.Tensor       # (e,)
    receivers: torch.Tensor     # (e,)
    dir_mask: torch.Tensor      # (n, 2) 1 where the DOF is prescribed
    u_dir: torch.Tensor         # (n, 2) prescribed values (0 elsewhere)
    K_idx: torch.Tensor         # (2, nnz) COO of the full stiffness (DOF indices)
    K_val: torch.Tensor         # (nnz,)
    F: torch.Tensor             # (2n,)
    u_fea: torch.Tensor         # (n, 2)
    meta: dict
    load_scale: float = 1.0     # sum_i |F_i| / E  (displacement units): amplitude normaliser

    @property
    def n_nodes(self):
        return self.node_x.shape[0]


def _node_features(mesh, prob, F, dir_mask):
    lo, hi = mesh.nodes.min(0), mesh.nodes.max(0)
    ext = float((hi - lo).max())
    xy = (mesh.nodes - (lo + hi) / 2) / ext
    f = F.reshape(-1, 2)
    f = f / max(np.abs(f).max(), 1e-30)
    on_bnd = np.zeros(mesh.n_nodes)
    for edges in mesh.boundary.values():
        on_bnd[np.asarray(edges).ravel()] = 1.0
    dnodes = np.where(dir_mask.any(1))[0]
    d_dir = cKDTree(mesh.nodes[dnodes]).query(mesh.nodes)[0] / ext
    md = mesh.metadata()
    geo = np.zeros(len(GEOMETRIES))
    geo[GEOMETRIES.index(prob.meta["geometry"])] = 1.0
    glob = np.array([np.log10(mesh.n_nodes) / 4, md["h_mean"] / ext, (hi - lo)[1] / (hi - lo)[0]])
    # Global load / constraint descriptors broadcast to every node: loads live only on a few boundary
    # nodes, which message passing cannot reach from the far side of the mesh within N hops.
    fraw = F.reshape(-1, 2)
    fmag = np.linalg.norm(fraw, axis=1)
    tot = max(fmag.sum(), 1e-30)
    load_c = (fmag[:, None] * xy).sum(0) / tot
    dir_c = xy[dnodes].mean(0)
    glob_load = np.array([fraw[:, 0].sum() / tot, fraw[:, 1].sum() / tot, np.abs(fraw[:, 0]).sum() / tot,
                          np.abs(fraw[:, 1]).sum() / tot, load_c[0], load_c[1], (fmag > 1e-12 * fmag.max()).mean(),
                          dir_c[0], dir_c[1]])
    cols = [xy, dir_mask, f, on_bnd[:, None], d_dir[:, None], np.full((mesh.n_nodes, 1), prob.nu),
            np.tile(glob, (mesh.n_nodes, 1)), np.tile(geo, (mesh.n_nodes, 1)), np.tile(glob_load, (mesh.n_nodes, 1))]
    return np.concatenate(cols, 1), ext, md


def make_sample(prob, fea=None):
    mesh = prob.mesh
    fea = fea or fem.solve(prob)
    dofs, vals = prob.dirichlet_arrays()
    dir_mask = np.zeros(2 * mesh.n_nodes)
    dir_mask[dofs] = 1
    u_dir = np.zeros(2 * mesh.n_nodes)
    u_dir[dofs] = vals
    dir_mask, u_dir = dir_mask.reshape(-1, 2), u_dir.reshape(-1, 2)
    node_x, ext, md = _node_features(mesh, prob, fea.F, dir_mask)
    e = mesh.edges()
    d = mesh.nodes[e[0]] - mesh.nodes[e[1]]
    ln = np.linalg.norm(d, axis=1, keepdims=True)
    edge_x = np.concatenate([d / ext, ln / ext, ln / md["h_mean"]], 1)
    K = fea.K.tocoo()
    t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt)
    meta = dict(md, **prob.meta, name=prob.name, E=prob.E, nu=prob.nu,
                t_fea_assemble=fea.t_assemble, t_fea_solve=fea.t_solve)
    load_scale = float(np.linalg.norm(fea.F.reshape(-1, 2), axis=1).sum() / prob.E)
    return Sample(t(node_x), t(edge_x), t(e[0], torch.long), t(e[1], torch.long), t(dir_mask), t(u_dir),
                  t(np.stack([K.row, K.col]), torch.long), t(K.data), t(fea.F), t(fea.u), meta, load_scale)


# ----------------------------------------------------------------------------- problem families
def _clamp_left(mesh):
    d = {}
    for nd in mesh.boundary_nodes("left"):
        d[2 * nd] = 0.0
        d[2 * nd + 1] = 0.0
    return d


def beam_problem(L, D, nx, ny, load, E=1e3, nu=0.3, P=1.0):
    mesh = meshes.check_mesh(meshes.rectangle(L, D, nx, ny, 0.0, -D / 2, name="beam"))
    I = D ** 3 / 12
    tr, body = [("top", zero_traction), ("bottom", zero_traction), ("right", zero_traction)], (0.0, 0.0)
    if load == "end_shear":      # Timoshenko parabolic end shear
        tr[2] = ("right", lambda x, y, n: np.stack([0 * y, -P / (2 * I) * (D ** 2 / 4 - y ** 2)], 1))
    elif load == "udl":
        tr[0] = ("top", lambda x, y, n: np.stack([0 * x, -P * np.ones_like(x)], 1))
    elif load == "gravity":
        body = (0.0, -P)
    elif load == "axial":
        tr[2] = ("right", lambda x, y, n: np.stack([P * np.ones_like(y), 0 * y], 1))
    return Problem(mesh, E, nu, _clamp_left(mesh), tr, body, name=f"beam_{load}_L{L:g}_D{D:g}_{nx}x{ny}",
                   meta=dict(geometry="beam", bc="clamped_left", load=load, aspect=L / D))


def hole_problem(a, sx, sy, n_theta, n_r, W=1.0, E=1e3, nu=0.3):
    mesh = meshes.check_mesh(meshes.plate_with_hole(W, a, n_theta, n_r))
    d = {}
    for nd in mesh.boundary_nodes("sym_x0"):
        d[2 * nd] = 0.0
    for nd in mesh.boundary_nodes("sym_y0"):
        d[2 * nd + 1] = 0.0
    tr = [("right", lambda x, y, n: np.stack([sx * np.ones_like(x), 0 * x], 1)),
          ("top", lambda x, y, n: np.stack([0 * x, sy * np.ones_like(x)], 1)),
          ("hole", zero_traction)]
    return Problem(mesh, E, nu, d, tr, name=f"hole_a{a:g}_s{sx:g},{sy:g}_{n_theta}x{n_r}",
                   meta=dict(geometry="plate_with_hole", bc="symmetry", load=f"tension({sx:g},{sy:g})", radius=a))


def plate_problem(lx, ly, n, fixed="left", load="gravity", E=7.0, nu=0.25, g=0.1):
    mesh = meshes.check_mesh(meshes.rectangle(lx, ly, n, max(2, int(round(n * ly / lx))), name="plate"))
    d = {}
    for nd in mesh.boundary_nodes(fixed):
        d[2 * nd] = 0.0
        d[2 * nd + 1] = 0.0
    free = [b for b in ("top", "bottom", "right", "left") if b != fixed]
    tr = [(b, zero_traction) for b in free]
    body = (0.0, -g) if load == "gravity" else (g, 0.0)
    return Problem(mesh, E, nu, d, tr, body, name=f"plate_{fixed}_{load}_{lx:g}x{ly:g}_n{n}",
                   meta=dict(geometry="plate", bc=f"clamped_{fixed}", load=load))


def training_problems():
    probs = []
    for L_D in (3, 3.5, 4.5, 5, 6, 6.5, 8, 9):        # L/D = 4 and 7 are held out for testing
        for load in ("end_shear", "udl", "gravity", "axial"):
            for ny in (4, 8):
                probs.append(beam_problem(L_D * 1.0, 1.0, int(L_D * ny), ny, load))
    for a in (0.1, 0.15, 0.2, 0.3, 0.35):
        for sx, sy in ((1, 0), (0, 1), (1, 1), (1, -0.5), (1, 0.25), (1, 0.75), (0.5, 1), (1, -1)):
            for nt, nr in ((16, 8), (32, 16)):
                probs.append(hole_problem(a, sx, sy, nt, nr))
    for lx, ly in ((1, 1), (1, 0.5), (2, 1), (1, 1.5)):
        for fixed in ("left", "bottom"):
            for load in ("gravity", "lateral"):
                for n in (8, 12, 16):
                    probs.append(plate_problem(lx, ly, n, fixed, load))
    return probs


def test_problems():
    """Held-out configurations: unseen aspect ratios, hole radii, load mixes and finer meshes."""
    return [
        beam_problem(7.0, 1.0, 70, 10, "end_shear"),    # Timoshenko beam, unseen L/D
        beam_problem(4.0, 1.0, 48, 12, "end_shear"),    # finer than any training mesh
        beam_problem(7.0, 1.0, 56, 8, "udl"),
        beam_problem(5.0, 1.0, 50, 10, "gravity"),
        hole_problem(0.25, 1, 0, 28, 14),                # central hole, unseen radius
        hole_problem(0.25, 1, 1, 40, 20),
        hole_problem(0.2, 1, 0.5, 36, 18),               # unseen load ratio (between 0.25 and 0.75)
        plate_problem(1.5, 1.0, 18, "left", "gravity"),
        plate_problem(1.0, 1.0, 20, "bottom", "lateral"),
    ]


def build_pool(problems):
    return [make_sample(p) for p in problems]


# ----------------------------------------------------------------------------- batching
def collate(samples, device="cpu"):
    """Concatenate graphs into one disconnected graph (PyG style)."""
    node_off = np.cumsum([0] + [s.n_nodes for s in samples[:-1]])
    cat = lambda f: torch.cat([getattr(s, f) for s in samples]).to(device)
    batch = dict(node_x=cat("node_x"), edge_x=cat("edge_x"), dir_mask=cat("dir_mask"), u_dir=cat("u_dir"),
                 F=cat("F"), u_fea=cat("u_fea"), K_val=cat("K_val"))
    batch["senders"] = torch.cat([s.senders + o for s, o in zip(samples, node_off)]).to(device)
    batch["receivers"] = torch.cat([s.receivers + o for s, o in zip(samples, node_off)]).to(device)
    batch["K_idx"] = torch.cat([s.K_idx + 2 * o for s, o in zip(samples, node_off)], 1).to(device)
    batch["graph_id"] = torch.cat([torch.full((s.n_nodes,), i, dtype=torch.long)
                                   for i, s in enumerate(samples)]).to(device)
    batch["n_graphs"] = len(samples)
    batch["load_scale"] = torch.tensor([s.load_scale for s in samples], device=device)
    return batch
