"""Baseline FEA: vectorised Q4 plane-stress linear elasticity (and scalar Laplace).

Python port of the MATLAB FEM folder (element.m, globalAssembly.m, solveFEM.m):
2x2 Gauss quadrature, bilinear shape functions, DOF numbering ID = [2i, 2i+1].
Adds edge tractions, non-zero Dirichlet values and a sparse solver.
"""
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)
GW = np.array([1.0, 1.0])


def d_matrix(E, nu):
    """Plane-stress constitutive matrix (D_mat.m)."""
    return E / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])


def shape(xi, eta):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dN = 0.25 * np.array([[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                          [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]])
    return N, dN


def q4_gradients(nodes, elems, xi, eta):
    """Physical shape-function gradients (n_el, 4, 2) and det(J) (n_el,) at (xi, eta)."""
    N, dN = shape(xi, eta)
    xe = nodes[elems]                                   # (n_el, 4, 2)
    J = np.einsum("ak,eki->eai", dN, xe)                # (n_el, 2, 2): d(x,y)/d(xi,eta)
    detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    Jinv = np.linalg.inv(J)
    dNdx = np.einsum("eia,ak->eki", Jinv, dN)           # (n_el, 4, 2)
    return N, dNdx, detJ


def b_matrix(dNdx):
    n_el = dNdx.shape[0]
    B = np.zeros((n_el, 3, 8))
    B[:, 0, 0::2] = dNdx[:, :, 0]
    B[:, 1, 1::2] = dNdx[:, :, 1]
    B[:, 2, 0::2] = dNdx[:, :, 1]
    B[:, 2, 1::2] = dNdx[:, :, 0]
    return B


def element_dofs(elems):
    return np.stack([2 * elems, 2 * elems + 1], axis=2).reshape(len(elems), 8)


def assemble_stiffness(nodes, elems, D):
    """Global K (sparse CSR) - vectorised equivalent of globalAssembly.m."""
    Ke = np.zeros((len(elems), 8, 8))
    for xi, wi in zip(GP, GW):
        for eta, wj in zip(GP, GW):
            _, dNdx, detJ = q4_gradients(nodes, elems, xi, eta)
            assert (detJ > 0).all(), "inverted element"
            B = b_matrix(dNdx)
            Ke += np.einsum("eai,ab,ebj->eij", B, D, B) * (detJ * wi * wj)[:, None, None]
    dofs = element_dofs(elems)
    rows = np.repeat(dofs, 8, axis=1).ravel()
    cols = np.tile(dofs, (1, 8)).ravel()
    n = 2 * len(nodes)
    return sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()


def body_force_vector(nodes, elems, b):
    """Consistent nodal loads from a constant body force b = (bx, by) (e.g. rho*g)."""
    F = np.zeros(2 * len(nodes))
    for xi, wi in zip(GP, GW):
        for eta, wj in zip(GP, GW):
            N, _, detJ = q4_gradients(nodes, elems, xi, eta)
            w = detJ * wi * wj
            for a in range(4):
                np.add.at(F, 2 * elems[:, a], N[a] * b[0] * w)
                np.add.at(F, 2 * elems[:, a] + 1, N[a] * b[1] * w)
    return F


def outward_normals(mesh, edges):
    """Unit outward normal of each boundary edge, using the adjacent element centroid."""
    edges = np.asarray(edges)
    p0, p1 = mesh.nodes[edges[:, 0]], mesh.nodes[edges[:, 1]]
    t = p1 - p0
    n = np.stack([t[:, 1], -t[:, 0]], axis=1) / np.linalg.norm(t, axis=1, keepdims=True)
    centroids = mesh.nodes[mesh.elems].mean(1)
    # nearest element centroid to each edge midpoint is the owning element
    mid = 0.5 * (p0 + p1)
    owner = np.argmin(((mid[:, None, :] - centroids[None]) ** 2).sum(-1), axis=1)
    inward = centroids[owner] - mid
    n *= np.where((n * inward).sum(1) > 0, -1.0, 1.0)[:, None]
    return n


def traction_vector(mesh, edges, traction_fn):
    """Nodal loads from a traction t(x, y, n) -> (m, 2) integrated with 2-pt Gauss on each edge."""
    F = np.zeros(2 * mesh.n_nodes)
    if len(edges) == 0:
        return F
    edges = np.asarray(edges)
    n = outward_normals(mesh, edges)
    p0, p1 = mesh.nodes[edges[:, 0]], mesh.nodes[edges[:, 1]]
    half_len = 0.5 * np.linalg.norm(p1 - p0, axis=1)
    for s, w in zip(GP, GW):
        N0, N1 = 0.5 * (1 - s), 0.5 * (1 + s)
        xg = N0 * p0 + N1 * p1
        t = np.asarray(traction_fn(xg[:, 0], xg[:, 1], n))
        for Na, nd in ((N0, edges[:, 0]), (N1, edges[:, 1])):
            np.add.at(F, 2 * nd, Na * t[:, 0] * half_len * w)
            np.add.at(F, 2 * nd + 1, Na * t[:, 1] * half_len * w)
    return F


@dataclass
class Problem:
    """A fully specified linear-elastic boundary value problem on a mesh."""
    mesh: object
    E: float
    nu: float
    dirichlet: dict            # global dof -> prescribed value
    tractions: list            # [(boundary_name, fn(x, y, n) -> (m, 2))]
    body_force: tuple = (0.0, 0.0)
    name: str = "problem"
    meta: dict = None

    def dirichlet_arrays(self):
        dofs = np.fromiter(self.dirichlet.keys(), dtype=int)
        vals = np.fromiter(self.dirichlet.values(), dtype=float)
        order = np.argsort(dofs)
        return dofs[order], vals[order]


@dataclass
class FEAResult:
    u: np.ndarray              # (n_nodes, 2)
    K: sp.csr_matrix           # full stiffness
    F: np.ndarray              # full load vector
    free: np.ndarray
    fixed: np.ndarray
    fixed_vals: np.ndarray
    t_assemble: float
    t_solve: float

    @property
    def t_total(self):
        return self.t_assemble + self.t_solve


def assemble(problem):
    m = problem.mesh
    K = assemble_stiffness(m.nodes, m.elems, d_matrix(problem.E, problem.nu))
    F = body_force_vector(m.nodes, m.elems, problem.body_force) if any(problem.body_force) else np.zeros(2 * m.n_nodes)
    for bname, fn in problem.tractions:
        F += traction_vector(m, m.boundary[bname], fn)
    return K, F


def solve(problem):
    """Baseline FEA solve with timing (assembly and sparse direct solve separately)."""
    t0 = time.perf_counter()
    K, F = assemble(problem)
    fixed, fixed_vals = problem.dirichlet_arrays()
    free = np.setdiff1d(np.arange(K.shape[0]), fixed)
    t1 = time.perf_counter()
    u = np.zeros(K.shape[0])
    u[fixed] = fixed_vals
    Kff = K[free][:, free]
    rhs = F[free] - K[free][:, fixed] @ fixed_vals
    u[free] = spla.spsolve(Kff.tocsc(), rhs)
    t2 = time.perf_counter()
    return FEAResult(u.reshape(-1, 2), K, F, free, fixed, fixed_vals, t1 - t0, t2 - t1)


def reduced_system(res):
    """K_ff, rhs for R(u_f) = K_ff u_f - (F_f - K_fd u_d); used by the physics-informed losses."""
    Kff = res.K[res.free][:, res.free]
    rhs = res.F[res.free] - res.K[res.free][:, res.fixed] @ res.fixed_vals
    return Kff.tocsr(), rhs


def assemble_laplace(nodes, elems, k=1.0):
    """Scalar Q4 Laplace stiffness (used for the potential-flow fluid in the FSI demo)."""
    Ke = np.zeros((len(elems), 4, 4))
    for xi, wi in zip(GP, GW):
        for eta, wj in zip(GP, GW):
            _, dNdx, detJ = q4_gradients(nodes, elems, xi, eta)
            Ke += k * np.einsum("eai,ebi->eab", dNdx, dNdx) * (detJ * wi * wj)[:, None, None]
    rows = np.repeat(elems, 4, axis=1).ravel()
    cols = np.tile(elems, (1, 4)).ravel()
    n = len(nodes)
    return sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()


def relative_l2(u_pred, u_ref):
    return float(np.linalg.norm(u_pred - u_ref) / np.linalg.norm(u_ref))
