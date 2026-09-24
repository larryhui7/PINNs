"""Benchmark boundary-value problems (several geometries, BCs and tractions).

Every case exposes both an FEA ``Problem`` on a Q4 mesh and a ``PinnSpec`` with
the continuous description (samplers, normals, tractions, hard Dirichlet lift)
used by the mesh-free PINN.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import meshes
from .fem import Problem


def _stack(a, b):
    """np.stack or torch.stack along axis 1, so lifts/distances work for FEA (numpy) and PINN (torch)."""
    if isinstance(a, np.ndarray):
        return np.stack([a, b], 1)
    import torch
    return torch.stack([a, b], 1)


# ----------------------------------------------------------------------------- analytic
def timoshenko_exact(x, y, L, D, E, nu, P):
    """Exact plane-stress cantilever with parabolic end shear (Timoshenko & Goodier;
    Augarde & Deeks 2008). Beam occupies x in [0, L], y in [-D/2, D/2]."""
    I = D ** 3 / 12.0
    ux = -P * y / (6 * E * I) * ((6 * L - 3 * x) * x + (2 + nu) * (y ** 2 - D ** 2 / 4))
    uy = P / (6 * E * I) * (3 * nu * y ** 2 * (L - x) + (4 + 5 * nu) * D ** 2 * x / 4 + (3 * L - x) * x ** 2)
    return _stack(ux, uy)


# ----------------------------------------------------------------------------- PINN spec
@dataclass
class BoundarySpec:
    name: str
    sample: Callable            # n -> (points (n,2), outward normals (n,2), length of this boundary)
    traction: Callable = None   # (x, y, n) -> (n,2); None = fully Dirichlet (handled by hard constraint)
    tangential_only: bool = False  # symmetry line: only shear traction must vanish


@dataclass
class PinnSpec:
    sample_interior: Callable   # n -> (n,2)
    boundaries: list
    lift: Callable              # x (n,2) -> g(x) (n,2) prescribed Dirichlet lift
    distance: Callable          # x (n,2) -> phi(x) (n,2), zero where u_i is prescribed
    length_scale: float
    disp_scale: tuple           # characteristic (u_x, u_y) magnitudes for output normalisation
    body_force: tuple = (0.0, 0.0)
    bbox: tuple = ((0, 0), (1, 1))


@dataclass
class Case:
    name: str
    problem: Problem
    pinn: PinnSpec
    exact: Callable = None      # optional analytic displacement (n,2) -> (n,2)
    meta: dict = field(default_factory=dict)
    # Fully clamped edges meeting traction-free edges create corner stress singularities; the
    # strong-form residual then prefers a near-zero field, so those cases use the energy loss.
    pinn_loss: str = "strong"


def _unit(n, rng, d=2):
    """Scrambled Sobol points in [0,1)^d: low-discrepancy collocation / quadrature points."""
    from scipy.stats import qmc
    m = int(np.ceil(np.log2(max(n, 2))))
    return qmc.Sobol(d=d, scramble=True, seed=rng).random_base2(m)[:n]


def _segment_sampler(p0, p1, normal):
    p0, p1, normal = map(np.asarray, (p0, p1, normal))
    length = float(np.linalg.norm(p1 - p0))

    def sample(n, rng):
        s = (np.arange(n) + rng.random()) / n            # stratified along the edge
        return p0 + s[:, None] * (p1 - p0), np.tile(normal, (n, 1)), length
    return sample


def _dirichlet_on(mesh, bname, fn):
    """Prescribe fn(x, y) -> (m,2) displacement on every node of a boundary."""
    nodes = mesh.boundary_nodes(bname)
    vals = fn(mesh.nodes[nodes, 0], mesh.nodes[nodes, 1])
    d = {}
    for k, nd in enumerate(nodes):
        d[2 * nd] = float(vals[k, 0])
        d[2 * nd + 1] = float(vals[k, 1])
    return d


def zero_traction(x, y, n):
    return np.zeros((len(x), 2))


# ----------------------------------------------------------------------------- cases
def timoshenko_beam(nx=32, ny=8, L=48.0, D=12.0, E=3.0e7, nu=0.3, P=1000.0):
    I = D ** 3 / 12.0
    exact = lambda x, y: timoshenko_exact(x, y, L, D, E, nu, P)
    # With this displacement field sigma_xy = +P/(2I)(D^2/4 - y^2) (upward end shear); n = (1, 0) at x = L.
    shear = lambda x, y, n: np.stack([np.zeros_like(y), P / (2 * I) * (D ** 2 / 4 - y ** 2)], 1)
    mesh = meshes.check_mesh(meshes.timoshenko_beam(L, D, nx, ny))
    prob = Problem(mesh, E, nu, _dirichlet_on(mesh, "left", exact),
                   [("right", shear), ("top", zero_traction), ("bottom", zero_traction)],
                   name="timoshenko_beam")
    tip = abs(exact(np.array([L]), np.array([0.0]))[0, 1])
    spec = PinnSpec(
        sample_interior=lambda n, rng: (_unit(n, rng) - [0, 0.5]) * [L, D],
        boundaries=[
            BoundarySpec("right", _segment_sampler((L, -D / 2), (L, D / 2), (1, 0)), shear),
            BoundarySpec("top", _segment_sampler((0, D / 2), (L, D / 2), (0, 1)), zero_traction),
            BoundarySpec("bottom", _segment_sampler((0, -D / 2), (L, -D / 2), (0, -1)), zero_traction),
        ],
        lift=lambda X: exact(0 * X[:, 0], X[:, 1]),
        distance=lambda X: _stack(X[:, 0] / L, X[:, 0] / L),
        length_scale=L, disp_scale=(tip * 3 * D / (2 * L), tip),
        bbox=((0, -D / 2), (L, D / 2)))
    return Case("timoshenko_beam", prob, spec, exact=lambda X: exact(X[:, 0], X[:, 1]),
                meta=dict(geometry="beam", bc="prescribed_exact_left", load="parabolic_end_shear"))


def cantilever_udl(nx=40, ny=8, L=10.0, D=2.0, E=1.0e3, nu=0.3, q=1.0):
    mesh = meshes.check_mesh(meshes.rectangle(L, D, nx, ny, 0.0, -D / 2, name="cantilever_udl"))
    pressure = lambda x, y, n: np.stack([np.zeros_like(x), -q * np.ones_like(x)], 1)
    prob = Problem(mesh, E, nu, _dirichlet_on(mesh, "left", lambda x, y: np.zeros((len(x), 2))),
                   [("top", pressure), ("right", zero_traction), ("bottom", zero_traction)],
                   name="cantilever_udl")
    I = D ** 3 / 12.0
    tip = q * L ** 4 / (8 * E * I)
    spec = PinnSpec(
        sample_interior=lambda n, rng: (_unit(n, rng) - [0, 0.5]) * [L, D],
        boundaries=[
            BoundarySpec("top", _segment_sampler((0, D / 2), (L, D / 2), (0, 1)), pressure),
            BoundarySpec("right", _segment_sampler((L, -D / 2), (L, D / 2), (1, 0)), zero_traction),
            BoundarySpec("bottom", _segment_sampler((0, -D / 2), (L, -D / 2), (0, -1)), zero_traction),
        ],
        lift=lambda X: 0 * X,
        distance=lambda X: _stack(X[:, 0] / L, X[:, 0] / L),
        length_scale=L, disp_scale=(tip * 4 * D / (3 * L), tip),
        bbox=((0, -D / 2), (L, D / 2)))
    return Case("cantilever_udl", prob, spec,
                meta=dict(geometry="beam", bc="clamped_left", load="uniform_pressure_top"), pinn_loss="energy")


def plate_hole(W=1.0, a=0.2, E=1.0e3, nu=0.3, sx=1.0, sy=0.0, n_theta=32, n_r=16, name=None):
    """Quarter plate with central hole, symmetry BCs, far-field tension (sx, sy)."""
    mesh = meshes.check_mesh(meshes.plate_with_hole(W, a, n_theta, n_r))
    dirichlet = {}
    for nd in mesh.boundary_nodes("sym_x0"):
        dirichlet[2 * nd] = 0.0
    for nd in mesh.boundary_nodes("sym_y0"):
        dirichlet[2 * nd + 1] = 0.0
    tx = lambda x, y, n: np.stack([sx * np.ones_like(x), np.zeros_like(x)], 1)
    ty = lambda x, y, n: np.stack([np.zeros_like(x), sy * np.ones_like(x)], 1)
    name = name or ("plate_hole_uniaxial" if sy == 0 else "plate_hole_biaxial")
    prob = Problem(mesh, E, nu, dirichlet,
                   [("right", tx), ("top", ty), ("hole", zero_traction)], name=name)

    def sample_interior(n, rng):
        pts = np.empty((0, 2))
        while len(pts) < n:
            p = _unit(2 * n, rng) * W
            pts = np.concatenate([pts, p[(p ** 2).sum(1) > a ** 2]])
        return pts[:n]

    def hole_sampler(n, rng):
        th = (np.arange(n) + rng.random()) / n * np.pi / 2
        p = a * np.stack([np.cos(th), np.sin(th)], 1)
        return p, -p / a, np.pi * a / 2      # outward from the solid points into the hole

    s = max(abs(sx), abs(sy))
    spec = PinnSpec(
        sample_interior=sample_interior,
        boundaries=[
            BoundarySpec("right", _segment_sampler((W, 0), (W, W), (1, 0)), tx),
            BoundarySpec("top", _segment_sampler((0, W), (W, W), (0, 1)), ty),
            BoundarySpec("hole", hole_sampler, zero_traction),
            BoundarySpec("sym_x0", _segment_sampler((0, a), (0, W), (-1, 0)), zero_traction, tangential_only=True),
            BoundarySpec("sym_y0", _segment_sampler((a, 0), (W, 0), (0, -1)), zero_traction, tangential_only=True),
        ],
        lift=lambda X: 0 * X,
        distance=lambda X: _stack(X[:, 0] / W, X[:, 1] / W),
        length_scale=W, disp_scale=(s * W / E, s * W / E),
        bbox=((0, 0), (W, W)))
    return Case(name, prob, spec,
                meta=dict(geometry="plate_with_hole", bc="symmetry", load=f"tension sx={sx} sy={sy}"))


def square_gravity(n=16, E=7.0, nu=0.25, g=-0.1, rho=1.0):
    """The original MATLAB case (global_fem_large_mesh_test_loss_adjust.m): unit square,
    left edge fixed, gravity body load."""
    mesh = meshes.check_mesh(meshes.rectangle(1.0, 1.0, n, n, name="square_gravity"))
    prob = Problem(mesh, E, nu, _dirichlet_on(mesh, "left", lambda x, y: np.zeros((len(x), 2))),
                   [("top", zero_traction), ("right", zero_traction), ("bottom", zero_traction)],
                   body_force=(0.0, rho * g), name="square_gravity")
    scale = abs(rho * g) / E
    spec = PinnSpec(
        sample_interior=lambda n_, rng: _unit(n_, rng),
        boundaries=[
            BoundarySpec("top", _segment_sampler((0, 1), (1, 1), (0, 1)), zero_traction),
            BoundarySpec("right", _segment_sampler((1, 0), (1, 1), (1, 0)), zero_traction),
            BoundarySpec("bottom", _segment_sampler((0, 0), (1, 0), (0, -1)), zero_traction),
        ],
        lift=lambda X: 0 * X,
        distance=lambda X: _stack(X[:, 0], X[:, 0]),
        length_scale=1.0, disp_scale=(0.5 * scale, scale), body_force=(0.0, rho * g),
        bbox=((0, 0), (1, 1)))
    return Case("square_gravity", prob, spec,
                meta=dict(geometry="square", bc="clamped_left", load="gravity_body_force"), pinn_loss="energy")


def benchmark_cases():
    return [
        timoshenko_beam(),
        cantilever_udl(),
        plate_hole(sx=1.0, sy=0.0),
        plate_hole(sx=1.0, sy=1.0),
        square_gravity(),
    ]
