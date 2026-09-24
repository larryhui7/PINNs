"""Strong-form PINN for 2D plane-stress linear elasticity (PyTorch).

u(x) = g(x) + phi(x) * s * N(x_hat)
  g    : lift that satisfies the Dirichlet data exactly ("strong Dirichlet")
  phi  : distance-like function that vanishes where u_i is prescribed
  s    : characteristic displacement per component (output normalisation)
  x_hat: coordinates normalised to [-1, 1] over the bounding box (input normalisation)

Loss = mean |div(sigma) + b|^2 over interior collocation points
     + mean |sigma.n - t|^2 over traction boundaries (tangential part only on symmetry lines),
non-dimensionalised by the applied stress level S (max traction, or |b|*L for body loads) and
the smallest domain extent l: interior residuals by S/l, traction residuals by S.
Optimiser: Adam with cosine-annealed learning rate, then an optional L-BFGS polish.

loss="energy" instead minimises the total potential energy (Deep Energy Method)
  Pi = int_Omega 1/2 sigma:eps dA - int_Gamma_t t.u ds - int_Omega b.u dA,
estimated on the same collocation points (traction-free and symmetry edges are natural BCs).
It only needs first derivatives and does not get trapped in the near-zero local minimum the
strong form falls into for bending-dominated problems (e.g. a clamped cantilever under pressure).
loss="energy+strong" warm-starts with the energy loss (Adam) and refines with the strong form (L-BFGS).
With loss="energy" the L-BFGS stage is skipped: on a fixed Monte-Carlo/Sobol quadrature, L-BFGS lowers
the *discrete* energy by exploiting the gaps between collocation points (gravity-loaded square:
0.44% error after Adam, 36% after an additional 3000 L-BFGS iterations), so Adam + cosine annealing only,
and the quadrature points are re-drawn (fresh scrambled Sobol set) every `resample_every` steps.
"""
import copy
import time

import numpy as np
import torch
from torch import nn

DTYPE = torch.float32


def default_device():
    return torch.device("mps" if torch.backends.mps.is_available() else
                        "cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, width=64, depth=4, act=nn.Tanh):
        super().__init__()
        layers, d = [], 2
        for _ in range(depth):
            layers += [nn.Linear(d, width), act()]
            d = width
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ElasticityPINN(nn.Module):
    def __init__(self, spec, width=64, depth=4):
        super().__init__()
        self.spec = spec
        lo, hi = map(lambda v: torch.tensor(v, dtype=DTYPE), spec.bbox)
        self.register_buffer("center", (lo + hi) / 2)
        self.register_buffer("halfwidth", (hi - lo) / 2)
        self.register_buffer("scale", torch.tensor(spec.disp_scale, dtype=DTYPE))
        self.mlp = MLP(width, depth)

    def forward(self, x):
        n = self.mlp((x - self.center) / self.halfwidth)
        return self.spec.lift(x) + self.spec.distance(x) * self.scale * n


def _grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def stresses(model, x, E, nu):
    """sigma_xx, sigma_yy, sigma_xy (plane stress) via automatic differentiation."""
    u = model(x)
    du = _grad(u[:, 0], x)
    dv = _grad(u[:, 1], x)
    c = E / (1 - nu ** 2)
    sxx = c * (du[:, 0] + nu * dv[:, 1])
    syy = c * (dv[:, 1] + nu * du[:, 0])
    sxy = c * (1 - nu) / 2 * (du[:, 1] + dv[:, 0])
    return sxx, syy, sxy


class CollocationSet:
    """Fixed collocation points (interior + each traction boundary) with precomputed targets."""

    def __init__(self, spec, n_interior=12000, n_boundary=1000, seed=0, device="cpu", area=None):
        rng = np.random.default_rng(seed)
        self.area = area
        t = lambda a: torch.tensor(a, dtype=DTYPE, device=device)
        self.interior = t(spec.sample_interior(n_interior, rng))
        self.boundaries = []
        for b in spec.boundaries:
            pts, normals, length = b.sample(n_boundary, rng)
            target = b.traction(pts[:, 0], pts[:, 1], normals)
            self.boundaries.append((b, t(pts), t(normals), t(target)))
            b.length = length
        self.scales = stress_scale(spec, self)

    @property
    def n_points(self):
        return len(self.interior) + sum(len(p) for _, p, _, _ in self.boundaries)


def stress_scale(spec, colloc):
    lo, hi = np.asarray(spec.bbox[0]), np.asarray(spec.bbox[1])
    ell = float((hi - lo).min())
    S = max([float(t.abs().max()) for _, _, _, t in colloc.boundaries] +
            [float(np.hypot(*spec.body_force)) * float((hi - lo).max())])
    return S, ell


def pinn_loss(model, colloc, E, nu, spec, traction_weight=1.0):
    S, ell = colloc.scales
    s_int, s_bnd = ell / S, 1.0 / S
    x = colloc.interior.clone().requires_grad_(True)
    sxx, syy, sxy = stresses(model, x, E, nu)
    dsxx, dsyy, dsxy = _grad(sxx, x), _grad(syy, x), _grad(sxy, x)
    rx = (dsxx[:, 0] + dsxy[:, 1] + spec.body_force[0]) * s_int
    ry = (dsxy[:, 0] + dsyy[:, 1] + spec.body_force[1]) * s_int
    losses = {"pde": (rx ** 2 + ry ** 2).mean()}
    for b, pts, n, target in colloc.boundaries:
        xb = pts.clone().requires_grad_(True)
        bxx, byy, bxy = stresses(model, xb, E, nu)
        tx = bxx * n[:, 0] + bxy * n[:, 1] - target[:, 0]
        ty = bxy * n[:, 0] + byy * n[:, 1] - target[:, 1]
        if b.tangential_only:
            r = (-n[:, 1] * tx + n[:, 0] * ty) ** 2
        else:
            r = tx ** 2 + ty ** 2
        losses[b.name] = (r * s_bnd ** 2).mean()
    total = losses["pde"] + traction_weight * sum(v for k, v in losses.items() if k != "pde")
    return total, losses


def energy_loss(model, colloc, E, nu, spec):
    """Total potential energy, non-dimensionalised by E * U^2 (U = characteristic displacement)."""
    x = colloc.interior.clone().requires_grad_(True)
    u = model(x)
    du, dv = _grad(u[:, 0], x), _grad(u[:, 1], x)
    exx, eyy, gxy = du[:, 0], dv[:, 1], du[:, 1] + dv[:, 0]
    c = E / (1 - nu ** 2)
    w = 0.5 * c * (exx ** 2 + eyy ** 2 + 2 * nu * exx * eyy + (1 - nu) / 2 * gxy ** 2)
    pi = colloc.area * w.mean()
    bx, by = spec.body_force
    if bx or by:
        pi = pi - colloc.area * (bx * u[:, 0] + by * u[:, 1]).mean()
    for b, pts, n, target in colloc.boundaries:
        if b.tangential_only or not bool(target.abs().max() > 0):
            continue                                   # natural boundary condition
        ub = model(pts)
        pi = pi - b.length * (target * ub).sum(1).mean()
    U = float(max(spec.disp_scale))
    return pi / (E * U ** 2), {"energy": pi}


def train_pinn(case, seed=0, adam_steps=4000, lbfgs_steps=500, lr=1e-3, lr_min=1e-5,
               n_interior=12000, n_boundary=1000, width=64, depth=4, warm_start=None, log_every=500,
               verbose=True, device=None, loss="strong", resample_every=500, callback=None):
    """callback(step, model, loss_value) is called after every Adam step and every L-BFGS evaluation."""
    device = device or default_device()
    torch.manual_seed(seed)
    spec, prob = case.pinn, case.problem
    model = ElasticityPINN(spec, width, depth).to(device, DTYPE)
    if warm_start is not None:
        model.load_state_dict(warm_start.state_dict())
    pe = prob.mesh.nodes[prob.mesh.elems]
    d1, d2 = pe[:, 2] - pe[:, 0], pe[:, 3] - pe[:, 1]
    area = float(0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]).sum())  # domain area from the FEA mesh
    colloc = CollocationSet(spec, n_interior, n_boundary, seed, device, area)
    adam_loss = (lambda: energy_loss(model, colloc, prob.E, prob.nu, spec)) if loss.startswith("energy") else \
        (lambda: pinn_loss(model, colloc, prob.E, prob.nu, spec))
    lbfgs_loss = (lambda: energy_loss(model, colloc, prob.E, prob.nu, spec)) if loss == "energy" else \
        (lambda: pinn_loss(model, colloc, prob.E, prob.nu, spec))
    history = []
    t0 = time.perf_counter()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=adam_steps, eta_min=lr_min)
    for it in range(adam_steps):
        if loss == "energy" and resample_every and it and it % resample_every == 0:
            colloc = CollocationSet(spec, n_interior, n_boundary, seed + 1000 * it, device, area)
        opt.zero_grad()
        lval, parts = adam_loss()
        lval.backward()
        opt.step()
        sched.step()
        if callback is not None:
            callback(it, model, lval.item())
        if it % log_every == 0 or it == adam_steps - 1:
            history.append((it, lval.item(), sched.get_last_lr()[0]))
            if verbose:
                print(f"  [{case.name} seed={seed}] adam {it:5d} loss={lval.item():.3e} lr={sched.get_last_lr()[0]:.1e}")

    if lbfgs_steps and loss != "energy":
        opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_steps, history_size=50,
                                tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")

        n_eval = [adam_steps]

        def closure():
            opt.zero_grad()
            l, _ = lbfgs_loss()
            l.backward()
            if callback is not None:
                callback(n_eval[0], model, l.item())
            n_eval[0] += 1
            return l
        opt.step(closure)
    lval, parts = pinn_loss(model, colloc, prob.E, prob.nu, spec)       # always report strong-form residuals
    history.append((adam_steps + lbfgs_steps, lval.item(), 0.0))
    if verbose:
        print(f"  [{case.name} seed={seed}] final strong-form loss={lval.item():.3e}")
    model = model.cpu()
    return model, dict(history=history, train_time=time.perf_counter() - t0, loss=loss,
                       n_collocation=colloc.n_points,
                       final_parts={k: v.item() for k, v in parts.items()})


@torch.no_grad()
def predict(model, points):
    dev = next(model.parameters()).device
    return model(torch.tensor(points, dtype=DTYPE, device=dev)).cpu().numpy()


class PinnEnsemble:
    """Average of independently seeded PINNs; the spread is a cheap uncertainty estimate."""

    def __init__(self, models):
        self.models = models

    def predict(self, points):
        preds = np.stack([predict(m, points) for m in self.models])
        return preds.mean(0), preds.std(0)

    def clone_member(self, i=0):
        return copy.deepcopy(self.models[i])
