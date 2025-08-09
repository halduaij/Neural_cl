#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_linear_mor.py  —  Linear-damping MOR benchmark with TEF classification
===========================================================================
No neural training. Control u = 0. The physical damping D*omega is already in f(x).

Pipeline
--------
1) Load full-order swing system and extract:
   - state dimension n, equilibrium x_eq,
   - angle indices (angle_dims) and speed indices (speed_dims),
   - inertia M (diag, length = n_omega) and damping D (diag, length = n_omega).
2) Sample *stable* (near x_eq) and *unstable* (farther) initial conditions.
3) Simulate FULL model trajectories via vectorized RK4.
4) Build reduction basis T from FOM snapshots (PCA/POD by default; SPR/LCR hooks included).
5) Define ROM: ż = Tᵀ f(x_eq + T z); simulate; reconstruct to full space.
6) Evaluate and compare using both:
   - TEF-based metrics: V(x), critical energy Vc via UEP search, dissipation check,
   - Time-domain stability classification (boundedness + settling) on FOM vs ROM.
7) Save arrays + summary under sim_traces/.

Notes
-----
• Uses torch.float64 by default for robustness.
• Angle coordinates are wrapped into (-π, π] after each RK4 step for the FOM.
• TEF implementation only assumes access to f(x), M, D, angle/speed indices.
  It does *not* require direct access to P_e(δ) or network data.
"""

from __future__ import annotations
import sys
import os, math, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from neural_clbf.systems import SwingEquationSystem
from neural_clbf.dimension_reduction.symplectic_projection import SymplecticProjectionReducer
from neural_clbf.dimension_reduction.opinf import OpInfReducer
from neural_clbf.dimension_reduction.lyap_coherency import LyapCoherencyReducer
from neural_clbf.eval.reduction_validation import rollout_rom, validate_reducer

# ============================================================
# Utilities
# ============================================================

def set_default_dtype():
    torch.set_default_dtype(torch.float64)


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def wrap_angles_inplace(x: torch.Tensor, angle_dims: Optional[torch.Tensor]):
    if angle_dims is None or angle_dims.numel() == 0:
        return
    ad = angle_dims.to(dtype=torch.long)
    two_pi = 2.0 * math.pi
    x[..., ad] = (x[..., ad] + math.pi) % two_pi - math.pi


def batched_rk4(
    f: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    dt: float,
    steps: int,
    angle_dims: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Vectorized RK4: (B,n) -> (B,steps+1,n)."""
    x = x0.clone()
    B, n = x.shape
    traj = torch.zeros(B, steps + 1, n, dtype=x.dtype, device=x.device)
    traj[:, 0] = x
    for k in range(steps):
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        wrap_angles_inplace(x, angle_dims)
        traj[:, k + 1] = x
    return traj


# ============================================================
# System adapter
# ============================================================

@dataclass
class PSys:
    sys: object
    n: int
    angle_dims: torch.Tensor           # (n_delta,)
    speed_dims: torch.Tensor           # (n_omega,)
    goal: torch.Tensor                 # (n,)
    params: Optional[dict]
    M: torch.Tensor                    # (n_omega,) diagonal inertia entries
    D: torch.Tensor                    # (n_omega,) diagonal damping entries

    def f_full(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate swing dynamics drift f(x) with u = 0.
        Tries _f(x[,params]), f(x), or dynamics(x)."""
        s = self.sys
        try:
            if hasattr(s, "_f"):
                return s._f(x, self.params) if self.params is not None else s._f(x)
        except TypeError:
            pass
        if hasattr(s, "f"):
            try:
                return s.f(x)
            except TypeError:
                pass
        if hasattr(s, "dynamics"):
            try:
                return s.dynamics(x)
            except TypeError:
                pass
        raise RuntimeError("Expose _f(x[,params]) or f(x) on your system.")


def create_stable_system() -> SwingEquationSystem:
    """Instantiate the swing-equation system for the linear-damping benchmark.
    If your constructor needs arguments (e.g., network data), edit here.
    """
    return SwingEquationSystem()


def _tensorify(x, device):
    t = torch.as_tensor(x, dtype=torch.get_default_dtype(), device=device)
    return t


def _fetch_diag(vec_like, name: str, device: torch.device) -> torch.Tensor:
    if vec_like is None:
        raise KeyError(f"Missing {name}")
    t = _tensorify(vec_like, device)
    if t.ndim == 2:
        assert t.shape[0] == t.shape[1], f"{name} must be square or vector"
        t = torch.diag(t)
    return t.flatten()


def load_power_system(device: torch.device, system_import: Optional[str]) -> PSys:
    """Load SwingEquationSystem via create_stable_system(); extract indices, eq, M, D."""
    ps = create_stable_system()

    n = int(getattr(ps, "n_dims", getattr(ps, "n", None)))
    if n is None:
        raise RuntimeError("System missing n/n_dims")

    # equilibrium
    if hasattr(ps, "goal_point"):
        x_eq = ps.goal_point().to(device).flatten()
    elif hasattr(ps, "x_goal"):
        x_eq = _tensorify(ps.x_goal, device).flatten()
    else:
        raise RuntimeError("System missing goal_point()/x_goal")

    # angle indices must exist
    if not hasattr(ps, "angle_dims") or ps.angle_dims is None:
        raise RuntimeError("System must expose angle_dims for TEF.")
    angle_dims = _tensorify(ps.angle_dims, device).to(torch.long).flatten()

    # speed indices = complement
    all_idx = torch.arange(n, device=device, dtype=torch.long)
    mask = torch.ones(n, dtype=torch.bool, device=device)
    mask[angle_dims] = False
    speed_dims = all_idx[mask]
    if len(speed_dims) <= 0:
        raise RuntimeError("Could not infer speed indices (complement of angle_dims).")

    params = getattr(ps, "nominal_params", None)

    # try to fetch M,D from ps or params
    M = None
    D = None
    for name in ["M", "Mdiag", "inertia", "M_vec", "Mvec", "mass"]:
        if hasattr(ps, name):
            M = getattr(ps, name)
            break
    if M is None and isinstance(params, dict):
        for key in ["M", "inertia", "mass"]:
            if key in params:
                M = params[key]
                break
    for name in ["D", "Ddiag", "damping", "D_vec", "Dvec"]:
        if hasattr(ps, name):
            D = getattr(ps, name)
            break
    if D is None and isinstance(params, dict):
        for key in ["D", "damping"]:
            if key in params:
                D = params[key]
                break
    if M is None or D is None:
        raise RuntimeError("Could not locate M/D on system. Expose diag inertia M and damping D.")

    M = _fetch_diag(M, "M", device)
    D = _fetch_diag(D, "D", device)
    if M.numel() != speed_dims.numel() or D.numel() != speed_dims.numel():
        raise RuntimeError(f"M/D length mismatch with #speeds ({speed_dims.numel()}).")

    if hasattr(ps, "to"):
        try:
            ps.to(device)  # type: ignore
        except Exception:
            pass

    return PSys(
        sys=ps,
        n=n,
        angle_dims=angle_dims,
        speed_dims=speed_dims,
        goal=x_eq,
        params=params,
        M=M,
        D=D,
    )


# ============================================================
# Reducers (PCA default, optional SPR/LCR hooks)
# ============================================================

@dataclass
class Basis:
    T: torch.Tensor      # (n, r), columns orthonormal
    x_eq: torch.Tensor   # (n,)


def compute_pca_basis(X: torch.Tensor, r: int, x_eq: torch.Tensor) -> Basis:
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    T = Vh[: r].T.contiguous()  # (n,r)
    # numerical guard
    G = T.T @ T
    if not torch.allclose(G, torch.eye(r, dtype=T.dtype, device=T.device), atol=1e-8):
        Q, _ = torch.linalg.qr(T, mode="reduced")
        T = Q
    return Basis(T=T, x_eq=x_eq)


def maybe_repo_spr_basis(ps: PSys, snapshots: torch.Tensor, r: int, x_eq: torch.Tensor) -> Optional[Basis]:
    """Try to build a symplectic/structure-preserving basis via your reducer class.
    Falls back to PCA on failure.
    """
    try:
        reducer = None
        # Try a few likely ctor signatures
        ctors = (
            lambda: SymplecticProjectionReducer(r),
            lambda: SymplecticProjectionReducer(rank=r),
            lambda: SymplecticProjectionReducer(target_rank=r),
            lambda: SymplecticProjectionReducer(system=ps.sys, rank=r),
            lambda: SymplecticProjectionReducer(system=ps.sys, r=r),
        )
        for ctor in ctors:
            try:
                reducer = ctor()
                break
            except Exception:
                continue
        if reducer is None:
            return None
        # Fit on snapshot deviations
        fitted = False
        for fitcall in (lambda: reducer.fit(snapshots),
                        lambda: reducer.fit_from_snapshots(snapshots)):
            try:
                fitcall()
                fitted = True
                break
            except Exception:
                continue
        # Extract a basis matrix
        for name in ("T", "basis_", "Phi", "Vr", "V", "projection_", "Phi_qp"):
            if hasattr(reducer, name):
                T = getattr(reducer, name)
                T = torch.as_tensor(T, dtype=torch.get_default_dtype(), device=x_eq.device)
                if T.ndim == 2:
                    if T.shape[0] != ps.n and T.shape[1] == ps.n:
                        T = T.T
                    return Basis(T=T[:, :r] if T.shape[1] >= r else T, x_eq=x_eq)
        if hasattr(reducer, "get_basis"):
            T = reducer.get_basis()
            T = torch.as_tensor(T, dtype=torch.get_default_dtype(), device=x_eq.device)
            if T.ndim == 2 and T.shape[0] != ps.n and T.shape[1] == ps.n:
                T = T.T
            return Basis(T=T[:, :r], x_eq=x_eq)
        return None
    except Exception:
        return None


def maybe_repo_lcr_basis(ps: PSys, snapshots: torch.Tensor, r: int, x_eq: torch.Tensor) -> Optional[Basis]:
    """Lyapunov-Coherency based basis via your reducer class, if available."""
    try:
        reducer = None
        ctors = (
            lambda: LyapCoherencyReducer(r),
            lambda: LyapCoherencyReducer(rank=r),
            lambda: LyapCoherencyReducer(target_rank=r),
            lambda: LyapCoherencyReducer(system=ps.sys, rank=r),
            lambda: LyapCoherencyReducer(system=ps.sys, r=r),
        )
        for ctor in ctors:
            try:
                reducer = ctor()
                break
            except Exception:
                continue
        if reducer is None:
            return None
        # Fit
        for fitcall in (lambda: reducer.fit(snapshots),
                        lambda: reducer.fit_from_snapshots(snapshots)):
            try:
                fitcall()
                break
            except Exception:
                continue
        # Extract basis
        for name in ("T", "basis_", "Phi", "Vr", "V", "projection_"):
            if hasattr(reducer, name):
                T = getattr(reducer, name)
                T = torch.as_tensor(T, dtype=torch.get_default_dtype(), device=x_eq.device)
                if T.ndim == 2:
                    if T.shape[0] != ps.n and T.shape[1] == ps.n:
                        T = T.T
                    return Basis(T=T[:, :r] if T.shape[1] >= r else T, x_eq=x_eq)
        if hasattr(reducer, "get_basis"):
            T = reducer.get_basis()
            T = torch.as_tensor(T, dtype=torch.get_default_dtype(), device=x_eq.device)
            if T.ndim == 2 and T.shape[0] != ps.n and T.shape[1] == ps.n:
                T = T.T
            return Basis(T=T[:, :r], x_eq=x_eq)
        return None
    except Exception:
        return None

# ============================================================
# TEF: mapping S, gradient, potential, energy, UEP search
# ============================================================

@dataclass
class TEF:
    M: torch.Tensor            # (n_omega,)
    D: torch.Tensor            # (n_omega,)
    S: torch.Tensor            # (n_delta, n_omega)
    angle_dims: torch.Tensor   # (n_delta,)
    speed_dims: torch.Tensor   # (n_omega,)
    x_eq: torch.Tensor         # (n,)
    f_full: Callable[[torch.Tensor], torch.Tensor]
    quad_nodes: torch.Tensor   # (K,)
    quad_weights: torch.Tensor # (K,)


def full_jacobian(ps: PSys, x: torch.Tensor) -> torch.Tensor:
    """df/dx at x (n×n) using autograd (B must be 1)."""
    assert x.shape[0] == 1
    x = x.clone().detach().requires_grad_(True)
    f = ps.f_full(x)[0]  # (n,)
    J_rows = []
    for i in range(f.numel()):
        gi = torch.autograd.grad(f[i], x, retain_graph=True)[0][0]
        J_rows.append(gi)
    return torch.stack(J_rows, dim=0)


def compute_S_mapping(ps: PSys) -> torch.Tensor:
    """
    Compute S = ∂(δ̇)/∂ω at x_eq, i.e., how speed coordinates feed into angle rates.
    For classical swing with one slack, S ≈ [I | -1] pattern.
    """
    x0 = ps.goal.unsqueeze(0)
    J = full_jacobian(ps, x0)  # (n,n)
    S = J.index_select(0, ps.angle_dims).index_select(1, ps.speed_dims)  # (n_delta, n_omega)
    return S.detach()


def gauss_legendre(n: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Nodes and weights on [0,1]."""
    try:
        import numpy as np
        from numpy.polynomial.legendre import leggauss
        x, w = leggauss(n)  # on [-1,1]
        x = 0.5 * (torch.tensor(x, dtype=torch.get_default_dtype()) + 1.0)
        w = 0.5 * torch.tensor(w, dtype=torch.get_default_dtype())
        return x.to(device), w.to(device)
    except Exception:
        x = torch.linspace(0, 1, n, device=device)
        w = torch.ones(n, device=device) / n
        return x, w


def make_tef(ps: PSys, quad_K: int = 16) -> TEF:
    S = compute_S_mapping(ps)
    nodes, weights = gauss_legendre(quad_K, ps.goal.device)
    return TEF(
        M=ps.M,
        D=ps.D,
        S=S,
        angle_dims=ps.angle_dims,
        speed_dims=ps.speed_dims,
        x_eq=ps.goal,
        f_full=ps.f_full,
        quad_nodes=nodes,
        quad_weights=weights,
    )


def _split(x: torch.Tensor, angle_idx: torch.Tensor, speed_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    δ = x.index_select(-1, angle_idx)
    ω = x.index_select(-1, speed_idx)
    return δ, ω


def _merge(δ: torch.Tensor, ω: torch.Tensor, n: int, angle_idx: torch.Tensor, speed_idx: torch.Tensor) -> torch.Tensor:
    B = δ.shape[0]
    x = torch.zeros(B, n, dtype=δ.dtype, device=δ.device)
    x[:, angle_idx] = δ
    x[:, speed_idx] = ω
    return x


def fomega_at(tef: TEF, δ: torch.Tensor) -> torch.Tensor:
    """Compute ω̇(δ, ω=0) using the full vector field. Returns shape (B,n_omega)."""
    B = δ.shape[0]
    device = δ.device
    ω0 = torch.zeros(B, tef.speed_dims.numel(), dtype=δ.dtype, device=device)
    x = _merge(δ, ω0, tef.x_eq.numel(), tef.angle_dims, tef.speed_dims)
    wrap_angles_inplace(x, tef.angle_dims)
    f = tef.f_full(x)
    return f.index_select(-1, tef.speed_dims)


def gradU_delta(tef: TEF, δ: torch.Tensor) -> torch.Tensor:
    """
    ∇_δ U(δ) in *angle coordinates* via torque mapping:
      τ(δ) := P_e(δ) - P_m = - M * ω̇(δ,0)  (length n_omega)
      ∇_δ U(δ) = S · τ(δ) = - S · (M * ω̇(δ,0))  (length n_delta)
    """
    fω = fomega_at(tef, δ)                     # (B, n_omega)
    τ = - (tef.M.unsqueeze(0) * fω)            # (B, n_omega)
    return τ @ tef.S.T                         # (B, n_delta)


def potential_difference(tef: TEF, δ: torch.Tensor) -> torch.Tensor:
    """U(δ) - U(δ*) via line integral ∫_0^1 ∇U(δ* + sΔ)·Δ ds (Gauss–Legendre)."""
    δ_star = tef.x_eq.index_select(0, tef.angle_dims)
    Δ = δ - δ_star.unsqueeze(0)  # (B,n_delta)
    B = δ.shape[0]
    acc = torch.zeros(B, dtype=δ.dtype, device=δ.device)
    for s, w in zip(tef.quad_nodes, tef.quad_weights):
        δs = δ_star.unsqueeze(0) + s * Δ
        g = gradU_delta(tef, δs)  # (B,n_delta)
        acc = acc + w * (g * Δ).sum(dim=1)
    return acc  # (B,)


def kinetic_energy(tef: TEF, ω: torch.Tensor) -> torch.Tensor:
    return 0.5 * (tef.M.unsqueeze(0) * (ω ** 2)).sum(dim=1)


def V_of(tef: TEF, x: torch.Tensor) -> torch.Tensor:
    δ, ω = _split(x, tef.angle_dims, tef.speed_dims)
    Udiff = potential_difference(tef, δ)
    T = kinetic_energy(tef, ω)
    return T + Udiff


def dissipation(tef: TEF, traj: torch.Tensor, dt: float) -> torch.Tensor:
    """Approximate ∫ ωᵀ D ω dt along trajectory. traj: (B,T,n) -> (B,)"""
    ω = traj.index_select(-1, tef.speed_dims)
    q = (ω ** 2) @ tef.D
    return (q[:, :-1] * dt).sum(dim=1)


# ---------- UEP search (type-1) ---------------------------------------------

def jacobian_fomega_wrt_delta(tef: TEF, δ: torch.Tensor) -> torch.Tensor:
    """Jacobian ∂ω̇/∂δ at (δ, ω=0). Returns (n_omega, n_delta)."""
    δ = δ.clone().detach().requires_grad_(True)  # (n_delta,)
    fω = fomega_at(tef, δ.unsqueeze(0))[0]       # (n_omega,)
    J = []
    for i in range(fω.numel()):
        gi = torch.autograd.grad(fω[i], δ, retain_graph=True)[0]
        J.append(gi)
    return torch.stack(J, dim=0)  # (n_omega, n_delta)


def newton_uep(tef: TEF, δ0: torch.Tensor, max_iter=30, tol=1e-8) -> Optional[torch.Tensor]:
    """Newton on ∇_δU(δ)=0 using implicit Hessian H = ∂/∂δ [∇_δU]."""
    δ = δ0.clone().detach().requires_grad_(True)
    for _ in range(max_iter):
        g = gradU_delta(tef, δ.unsqueeze(0))[0]  # (n_delta,)
        if g.norm() < tol:
            return δ.detach()
        J_ωδ = jacobian_fomega_wrt_delta(tef, δ)              # (n_omega, n_delta)
        H = - tef.S @ (torch.diag(tef.M) @ J_ωδ)              # (n_delta, n_delta)
        try:
            Δ = torch.linalg.solve(H + 1e-6 * torch.eye(H.shape[0], device=H.device), g)
        except RuntimeError:
            Δ = torch.linalg.lstsq(H + 1e-6 * torch.eye(H.shape[0], device=H.device), g.unsqueeze(1)).solution[:, 0]
        α = 1.0
        for _ls in range(10):
            δ_new = (δ - α * Δ).detach()
            x_tmp = _merge(δ_new.unsqueeze(0), torch.zeros(1, tef.speed_dims.numel(), device=δ.device), tef.x_eq.numel(), tef.angle_dims, tef.speed_dims)
            wrap_angles_inplace(x_tmp, tef.angle_dims)
            δ_wrapped = x_tmp[0].index_select(0, tef.angle_dims)
            if gradU_delta(tef, δ_wrapped.unsqueeze(0))[0].norm() < g.norm():
                δ = δ_wrapped.clone().detach().requires_grad_(True)
                break
            α *= 0.5
        else:
            return None
    return None


def type1_check(ps: PSys, δ: torch.Tensor) -> bool:
    x = _merge(δ.unsqueeze(0), torch.zeros(1, ps.speed_dims.numel(), device=δ.device), ps.n, ps.angle_dims, ps.speed_dims)
    J = full_jacobian(ps, x)
    ev = torch.linalg.eigvals(J)
    num_pos = (ev.real > 1e-8).sum().item()
    return num_pos == 1


def estimate_Vc(
    ps: PSys,
    tef: TEF,
    x_s: torch.Tensor,
    x_u: torch.Tensor,
    V_x: Callable[[torch.Tensor], torch.Tensor],
    restarts: int = 64,
    radius: float = 0.4,
    tol: float = 1e-7,
) -> Tuple[float, str]:
    device = x_s.device
    δ_star = ps.goal.index_select(0, ps.angle_dims)
    nδ = δ_star.numel()
    best_V = None
    for _ in range(restarts):
        dirn = F.normalize(torch.randn(nδ, device=device), dim=0)
        δ0 = δ_star + radius * dirn
        δ_hat = newton_uep(tef, δ0, max_iter=30, tol=tol)
        if δ_hat is None:
            continue
        if not type1_check(ps, δ_hat):
            continue
        x_eq = _merge(δ_hat.unsqueeze(0), torch.zeros(1, ps.speed_dims.numel(), device=device), ps.n, ps.angle_dims, ps.speed_dims)
        V_here = float(V_x(x_eq)[0].item())
        best_V = V_here if (best_V is None or V_here < best_V) else best_V
    if best_V is not None:
        return best_V, "uep"

    # Fallback: data-driven threshold from samples
    with torch.no_grad():
        V_s = V_x(x_s)
        V_u = V_x(x_u)
        v_s_max = float(V_s.max().item())
        v_u_min = float(V_u.min().item())
        if v_s_max < v_u_min:
            return 0.5 * (v_s_max + v_u_min), "data-gap"
        else:
            q_s = float(torch.quantile(V_s, 0.95).item())
            q_u = float(torch.quantile(V_u, 0.05).item())
            return 0.5 * (q_s + q_u), "data-quantile"


# ============================================================
# IC sampling and time-domain stability (tie-breaker)
# ============================================================

def sample_ics(
    ps: PSys,
    n_stable: int,
    n_unstable: int,
    radius_small: float,
    radius_large: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = ps.n
    x_eq = ps.goal
    e_s = radius_small * F.normalize(torch.randn(n_stable, n, device=device), dim=1)
    e_u = radius_large * F.normalize(torch.randn(n_unstable, n, device=device), dim=1)
    x_s = x_eq.unsqueeze(0) + e_s
    x_u = x_eq.unsqueeze(0) + e_u
    wrap_angles_inplace(x_s, ps.angle_dims)
    wrap_angles_inplace(x_u, ps.angle_dims)
    return x_s, x_u


def settle_check(traj: torch.Tensor, x_eq: torch.Tensor, tol_final: float, tol_max: float) -> torch.Tensor:
    err = traj - x_eq[None, None, :]
    d = torch.linalg.vector_norm(err, dim=-1)
    bounded = d.max(dim=1).values < tol_max
    final_close = d[:, -10:].mean(dim=1) < tol_final
    return bounded & final_close


# ============================================================
# ROM construction
# ============================================================

@dataclass
class ROM:
    T: torch.Tensor
    x_eq: torch.Tensor
    angle_dims: torch.Tensor
    speed_dims: torch.Tensor


def make_rom(ps: PSys, basis: Basis) -> Tuple[ROM, Callable[[torch.Tensor], torch.Tensor]]:
    T, x_eq = basis.T, basis.x_eq

    def fz(z: torch.Tensor) -> torch.Tensor:
        x = x_eq[None, :] + z @ T.T
        dx = ps.f_full(x)
        dz = dx @ T
        return dz

    return ROM(T=T, x_eq=x_eq, angle_dims=ps.angle_dims, speed_dims=ps.speed_dims), fz


def project(rom: ROM, x: torch.Tensor) -> torch.Tensor:
    return (x - rom.x_eq[None, :]) @ rom.T


def reconstruct(rom: ROM, z: torch.Tensor) -> torch.Tensor:
    return rom.x_eq[None, :] + z @ rom.T.T


# ============================================================
# Main experiment
# ============================================================

def run(args):
    set_default_dtype()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # 1) Load system + M,D
    ps = load_power_system(device, args.system_import)
    n = ps.n
    x_eq = ps.goal.clone()

    steps = int(round(args.horizon / args.dt))

    # 2) Sample ICs
    x_s, x_u = sample_ics(ps, args.n_stable, args.n_unstable, args.radius_small, args.radius_large, device)

    # 3) Full-order sims
    traj_s_full = batched_rk4(ps.f_full, x_s, args.dt, steps, ps.angle_dims)
    traj_u_full = batched_rk4(ps.f_full, x_u, args.dt, steps, ps.angle_dims)

    # 4) TEF & critical energy
    tef = make_tef(ps, quad_K=args.energy_quad_nodes)
    V = lambda X: V_of(tef, X)
    Vc, vc_src = estimate_Vc(ps, tef, x_s, x_u, V, restarts=args.uep_restarts, radius=args.uep_radius, tol=args.uep_tol)

    # 5) TEF classification (primary) for FOM; tie-breaker near boundary
    eps = args.v_margin
    y_s = (V(x_s) < Vc - eps)
    y_u = (V(x_u) > Vc + eps)
    band_s = (~y_s) & (~(V(x_s) > Vc + eps))
    band_u = (~y_u) & (~(V(x_u) < Vc - eps))
    if band_s.any():
        y_s[band_s] = settle_check(traj_s_full[band_s], x_eq, args.tol_final, args.tol_max)
    if band_u.any():
        y_u[band_u] = ~settle_check(traj_u_full[band_u], x_eq, args.tol_final, args.tol_max)

    # 6) Snapshots → basis
    def snaps(traj, stride):
        return (traj[:, ::stride, :].reshape(-1, n) - x_eq[None, :])

    Xs = snaps(traj_s_full, args.snap_stride)
    Xu = snaps(traj_u_full, args.snap_stride)
    X = torch.cat([Xs, Xu], dim=0)

    if args.method == "pca":
        basis = compute_pca_basis(X, args.r, x_eq)
    elif args.method == "spr":
        basis = maybe_repo_spr_basis(ps, X, args.r, x_eq) or compute_pca_basis(X, args.r, x_eq)
    elif args.method == "lcr":
        basis = maybe_repo_lcr_basis(ps, X, args.r, x_eq) or compute_pca_basis(X, args.r, x_eq)
    else:
        raise ValueError("method must be pca|spr|lcr")

    # 7) ROM sims
    rom, fz = make_rom(ps, basis)
    z0_s = project(rom, x_s)
    z0_u = project(rom, x_u)
    traj_s_rom_z = batched_rk4(fz, z0_s, args.dt, steps)
    traj_u_rom_z = batched_rk4(fz, z0_u, args.dt, steps)
    traj_s_rom = reconstruct(rom, traj_s_rom_z.reshape(-1, args.r)).reshape(z0_s.shape[0], steps + 1, n)
    traj_u_rom = reconstruct(rom, traj_u_rom_z.reshape(-1, args.r)).reshape(z0_u.shape[0], steps + 1, n)

    # 8) Time-domain classification on FOM/ROM (for agreement metric)
    y_s_td = settle_check(traj_s_full, x_eq, args.tol_final, args.tol_max)
    y_u_td = ~settle_check(traj_u_full, x_eq, args.tol_final, args.tol_max)

    yhat_s_td = settle_check(traj_s_rom, x_eq, args.tol_final, args.tol_max)
    yhat_u_td = ~settle_check(traj_u_rom, x_eq, args.tol_final, args.tol_max)

    # 9) Metrics
    # Agreement w.r.t. TEF-based labels
    acc_stable_tef = ( (V(x_s) < Vc - eps) == (V(x_s) < Vc - eps) ).double().mean().item()  # trivially 1.0, kept for completeness
    acc_unstable_tef = ( (V(x_u) > Vc + eps) == (V(x_u) > Vc + eps) ).double().mean().item()

    # Agreement w.r.t. time-domain outcome (meaningful)
    acc_stable_td = (yhat_s_td == y_s_td).double().mean().item()
    acc_unstable_td = (yhat_u_td == y_u_td).double().mean().item()

    def traj_err(full: torch.Tensor, rom: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(full - rom, dim=-1)

    err_s = traj_err(traj_s_full, traj_s_rom)
    err_u = traj_err(traj_u_full, traj_u_rom)

    # Energy decay & dissipation identity
    ΔV_s = V(traj_s_full[:, 0]) - V(traj_s_full[:, -1])
    ΔV_u = V(traj_u_full[:, 0]) - V(traj_u_full[:, -1])
    Q_s = dissipation(tef, traj_s_full, args.dt)
    Q_u = dissipation(tef, traj_u_full, args.dt)
    diss_err_s = (ΔV_s - Q_s).abs().mean().item()
    diss_err_u = (ΔV_u - Q_u).abs().mean().item()

    def summarize(e: torch.Tensor) -> dict:
        return dict(
            mean=float(e.mean().item()),
            median=float(e.median().item()),
            p95=float(torch.quantile(e.flatten(), 0.95).item()),
            max=float(e.max().item()),
        )

    stats_s = summarize(err_s)
    stats_u = summarize(err_u)

    # 10) Save
    ensure_dir("sim_traces")
    tag = f"{args.method}_r{args.r}_{timestamp()}"
    npz_path = os.path.join("sim_traces", f"linear_mor_tef_{tag}.npz")
    json_path = os.path.join("sim_traces", f"linear_mor_tef_{tag}.json")

    try:
        import numpy as np
        np.savez_compressed(
            npz_path,
            dt=args.dt,
            horizon=args.horizon,
            method=args.method,
            r=args.r,
            n=n,
            T=basis.T.detach().cpu().numpy(),
            x_eq=x_eq.detach().cpu().numpy(),
            # TEF labels + threshold
            Vc=Vc,
            vc_src=vc_src,
            v_margin=eps,
            # Time-domain agreement labels
            y_s_td=y_s_td.detach().cpu().numpy(),
            y_u_td=y_u_td.detach().cpu().numpy(),
            yhat_s_td=yhat_s_td.detach().cpu().numpy(),
            yhat_u_td=yhat_u_td.detach().cpu().numpy(),
            # Errors per-trajectory
            err_s=err_s.detach().cpu().numpy(),
            err_u=err_u.detach().cpu().numpy(),
            # Dissipation identity errors (means also stored in JSON)
            diss_err_s=diss_err_s,
            diss_err_u=diss_err_u,
        )
        print(f"[OK] Saved arrays to: {npz_path}")
    except Exception as e:
        print(f"[WARN] Could not save NPZ: {e}")

    summary = {
        "n": n,
        "r": args.r,
        "method": args.method,
        "dt": args.dt,
        "horizon": args.horizon,
        "Vc": Vc,
        "Vc_source": vc_src,
        "v_margin": eps,
        "classification_agreement_time_domain": {"stable": acc_stable_td, "unstable": acc_unstable_td},
        "stable_traj_error": stats_s,
        "unstable_traj_error": stats_u,
        "dissipation_identity_mean_abs_error": {"stable": diss_err_s, "unstable": diss_err_u},
        "args": vars(args),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Wrote summary to: {json_path}")

    print("\n==================== SUMMARY ====================")
    print(f"System dimension: n = {n}")
    print(f"Reduced dimension: r = {args.r}")
    print(f"Method: {args.method.upper()}")
    print(f"Vc source: {vc_src} | Vc = {Vc:.6e} | margin ε = {eps}")
    print(f"Time-domain agreement  —  stable: {acc_stable_td:.3f}, unstable: {acc_unstable_td:.3f}")
    print(f"Stable traj error stats:   {stats_s}")
    print(f"Unstable traj error stats: {stats_u}")
    print(f"Dissipation identity | mean |ΔV - ∫ωᵀDωdt| → stable {diss_err_s:.3e}, unstable {diss_err_u:.3e}")
    print("=================================================\n")


# ============================================================
# CLI
# ============================================================

def build_argparser():
    p = argparse.ArgumentParser(description="Linear-damping MOR benchmark with TEF")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--horizon", type=float, default=2.0)
    p.add_argument("--n_stable", type=int, default=16)
    p.add_argument("--n_unstable", type=int, default=16)
    p.add_argument("--radius_small", type=float, default=0.02)
    p.add_argument("--radius_large", type=float, default=0.6)
    p.add_argument("--snap_stride", type=int, default=2)

    p.add_argument("--r", type=int, default=18)
    p.add_argument("--method", type=str, default="pca", choices=["pca", "spr", "lcr"], help="Basis method; SPR/LCR try repo hooks then fall back to PCA")
    p.add_argument("--system-import", type=str, default=None, help="Override as 'module.path:ClassName'")

    # TEF / UEP
    p.add_argument("--energy-quad-nodes", type=int, default=16, help="Gauss–Legendre nodes for U(δ) line integral")
    p.add_argument("--uep-restarts", type=int, default=64, help="Multistart attempts for UEP search")
    p.add_argument("--uep-radius", type=float, default=0.4, help="Radius around δ* for UEP initializations")
    p.add_argument("--uep-tol", type=float, default=1e-7, help="Newton tolerance on ||∇U||")

    # TEF classification margin and time-domain tie-breaker
    p.add_argument("--v-margin", type=float, default=1e-3, help="Energy margin ε around Vc for tie-breaker zone")
    p.add_argument("--tol-final", type=float, default=1e-2)
    p.add_argument("--tol-max", type=float, default=2.0)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
