"""Structured bilinear-quad (Q4) mesh generators and boundary bookkeeping.

Node ordering inside every element is counter-clockwise starting bottom-left,
matching ``processGmsh.m`` in the MATLAB code (IX = QUADS(:, [2,3,4,1])).
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Mesh:
    nodes: np.ndarray                 # (n_nodes, 2)
    elems: np.ndarray                 # (n_elems, 4) CCW node ids
    boundary: dict                    # name -> list of (n0, n1) boundary edges, oriented CCW
    name: str = "mesh"
    meta: dict = field(default_factory=dict)

    @property
    def n_nodes(self):
        return self.nodes.shape[0]

    @property
    def n_elems(self):
        return self.elems.shape[0]

    def boundary_nodes(self, name):
        return np.unique(np.asarray(self.boundary[name]).ravel())

    def edges(self):
        """Unique undirected element edges, returned bidirectionally (2, 2E) like processGmsh.m."""
        e = np.concatenate([self.elems[:, [i, (i + 1) % 4]] for i in range(4)])
        e = np.unique(np.sort(e, axis=1), axis=0)
        return np.concatenate([e.T, e[:, ::-1].T], axis=1)

    def metadata(self):
        """Mesh metadata stored alongside each training-pool sample."""
        e = self.edges()
        lengths = np.linalg.norm(self.nodes[e[0]] - self.nodes[e[1]], axis=1)
        lo, hi = self.nodes.min(0), self.nodes.max(0)
        return dict(self.meta, n_nodes=self.n_nodes, n_elems=self.n_elems,
                    h_min=float(lengths.min()), h_mean=float(lengths.mean()),
                    h_max=float(lengths.max()), bbox=(lo.tolist(), hi.tolist()))


def _grid_elems(nx, ny):
    """CCW connectivity for an (nx+1) x (ny+1) node grid indexed as j*(nx+1)+i."""
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    n0 = (j * (nx + 1) + i).ravel()
    return np.stack([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1], axis=1)


def _grid_boundary(nx, ny):
    idx = lambda i, j: j * (nx + 1) + i
    return {
        "bottom": [(idx(i, 0), idx(i + 1, 0)) for i in range(nx)],
        "right": [(idx(nx, j), idx(nx, j + 1)) for j in range(ny)],
        "top": [(idx(i + 1, ny), idx(i, ny)) for i in range(nx)],
        "left": [(idx(0, j + 1), idx(0, j)) for j in range(ny)],
    }


def rectangle(lx, ly, nx, ny, x0=0.0, y0=0.0, name="rectangle"):
    xs = np.linspace(x0, x0 + lx, nx + 1)
    ys = np.linspace(y0, y0 + ly, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    nodes = np.stack([X.ravel(), Y.ravel()], axis=1)
    return Mesh(nodes, _grid_elems(nx, ny), _grid_boundary(nx, ny), name,
                meta=dict(geometry=name, lx=lx, ly=ly))


def timoshenko_beam(length=48.0, depth=12.0, nx=32, ny=8):
    """Cantilever x in [0, L], y in [-D/2, D/2] (Timoshenko & Goodier setup)."""
    m = rectangle(length, depth, nx, ny, 0.0, -depth / 2, name="timoshenko_beam")
    m.meta.update(geometry="timoshenko_beam", length=length, depth=depth)
    return m


def plate_with_hole(width=1.0, radius=0.2, n_theta=24, n_r=12, grading=1.6):
    """Quarter model of a square plate [0,W]^2 with a central hole of radius a.

    Single structured block: angle theta in [0, pi/2] maps the hole arc onto the
    outer square boundary (right edge for theta<=pi/4, top edge above). Radial
    spacing is graded towards the hole where the stress concentrates.
    """
    theta = np.linspace(0.0, np.pi / 2, n_theta + 1)
    s = np.linspace(0.0, 1.0, n_r + 1) ** grading
    inner = radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    outer = np.where((theta <= np.pi / 4)[:, None],
                     np.stack([np.full_like(theta, width), width * np.tan(np.minimum(theta, np.pi / 4))], 1),
                     np.stack([width / np.tan(np.maximum(theta, np.pi / 4)), np.full_like(theta, width)], 1))
    # Node (i = theta index, j = radial index) -> j*(n_theta+1)+i
    nodes = np.concatenate([inner + sj * (outer - inner) for sj in s])
    nodes[np.abs(nodes) < 1e-14] = 0.0
    nx, ny = n_theta, n_r
    elems = _grid_elems(nx, ny)
    # grid "bottom" is the hole arc, grid "top" is the outer square, i=0 is y=0 line, i=nx is x=0 line.
    # Element orientation (theta, r) is clockwise in physical space -> flip.
    elems = elems[:, [0, 3, 2, 1]]
    g = _grid_boundary(nx, ny)
    flip = lambda edges: [(b, a) for a, b in edges]
    outer_edges = flip(g["top"])
    right = [e for e in outer_edges if nodes[list(e), 0].min() > width - 1e-9]
    top = [e for e in outer_edges if nodes[list(e), 1].min() > width - 1e-9]
    boundary = {
        "hole": flip(g["bottom"]),
        "sym_y0": flip(g["left"]),    # theta = 0 line, y = 0 (u_y = 0)
        "sym_x0": flip(g["right"]),   # theta = pi/2 line, x = 0 (u_x = 0)
        "right": right,
        "top": top,
    }
    m = Mesh(nodes, elems, boundary, "plate_with_hole",
             meta=dict(geometry="plate_with_hole", width=width, radius=radius))
    return m


def check_mesh(mesh):
    """All element Jacobians at the centre must be positive (CCW, non-inverted)."""
    p = mesh.nodes[mesh.elems]
    d1 = p[:, 2] - p[:, 0]
    d2 = p[:, 3] - p[:, 1]
    area2 = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    assert (area2 > 0).all(), f"{(area2 <= 0).sum()} inverted elements in {mesh.name}"
    return mesh
