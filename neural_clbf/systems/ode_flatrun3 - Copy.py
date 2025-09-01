#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flat-run time-domain simulator for IEEE39ControlAffineDAE.
"Flat" means starting from equilibrium where dx/dt = 0 (flat derivative).

This script:
- integrates the control-affine IEEE39 model with RK4,
- supports a DAE-style ALTER event (per-bus load increase) by scaling sP/sQ,
- optionally applies a PV active-power step (u component) at a given time,
- audits KCL and derivative norms over time and can plot results.

Usage examples:
  # 2 seconds from equilibrium
  python -m neural_clbf.systems.ode_flatrun2 --T 2.0

  # +5% load at original bus 39 (mapped to reduced index 9) at t=2.0s (hard step)
  python -m neural_clbf.systems.ode_flatrun2 --T 6.0 --dt 0.001 --alter-amount 1.05 --orig-buses 39 --alter-t 2.0

  # +5% load at reduced bus index 6 (0..9) at t=2.0s
  python -m neural_clbf.systems.ode_flatrun2 --T 6.0 --dt 0.001 --alter-amount 1.05 --alter-buses 6 --alter-t 2.0

  # Smooth +5% at bus 39 with tau=0.05s
  python -m neural_clbf.systems.ode_flatrun2 --T 6.0 --dt 0.001 --alter-amount 1.05 --orig-buses 39 --alter-t 2.0 --alter-tau 0.05
"""
import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pytorch_lightning as pl

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
torch.autograd.set_detect_anomaly(False)  # must be ON before building the forward graph
import torch, torch.nn as nn, torch.nn.functional as F
from neural_clbf.controllers.clf_controller import CLFController
from neural_clbf.systems.utils import ScenarioList
from neural_clbf.experiments import ExperimentSuite

# Import your system class (control-affine IEEE39 with algebraic KCL solve)
from neural_clbf.systems.IEEE39ControlAffineDAE298 import IEEE39ControlAffineDAE
# ---- OPTIONAL TRAINING HARNESS (Neural CLBF v2) ----
from neural_clbf.controllers.neural_clbf_controller import NeuralCLBFController  # v2
from neural_clbf.systems.utils import ScenarioList
from neural_clbf.experiments import ExperimentSuite

def _V_JV_f_g(ctrl, x):
    """Compute V, ∂V/∂x, f, g once."""
    V, JV = ctrl.V_with_jacobian(x)                         # V: (B,), JV: (B,1,D)
    f, g  = ctrl.dynamics_model.control_affine_dynamics(x)  # f: (B,D,[1]), g: (B,D,m)
    if f.dim() == 3 and f.size(-1) == 1:
        f = f.squeeze(-1)                                   # f: (B,D)
    return V, JV, f, g

def solve_clf_qp_fast(ctrl, x, u_ref=None, mode="cf", eps=1e-7):
    """
    Fast CLF-QP for the standard 1-constraint problem you are solving:
        min_u  0.5||u - u_ref||^2 + 0.5*p*ρ^2
        s.t.   LfV(x) + LgV(x) u + λ V(x) <= ρ ,   ρ >= 0
    Returns: (u, ρ) where ρ has shape (B,1).

    mode:
      - "cf"     : pure closed-form (exact when no box is embedded in the QP)
      - "hybrid" : closed-form, clamp to model box (if present), then
                   fallback to the original cvxpylayers QP only for items
                   still violating the CLF inequality by > eps
    """
    B = x.shape[0]
    if u_ref is None:
        u_ref = ctrl.dynamics_model.u_eq.expand(B, -1).to(x)

    # Lie derivatives
    V, JV, f, g = _V_JV_f_g(ctrl, x)
    LfV = torch.bmm(JV, f.unsqueeze(-1)).squeeze(-1).squeeze(-1)   # (B,)
    LgV = torch.bmm(JV, g).squeeze(1)                              # (B,m)

    a = LgV                                                        # (B,m)
    b = -(LfV + ctrl.clf_lambda * V)                               # (B,)
    p = float(ctrl.clf_relaxation_penalty)

    # ----- closed-form half-space projection (exact) -----
    # If a^T u_ref <= b: u*=u_ref, ρ*=0
    # Else μ = (a^T u_ref - b) / (||a||^2 + 1/p), u*=u_ref - μ a, ρ*=μ/p
    dot   = (a * u_ref).sum(dim=1) - b
    denom = (a * a).sum(dim=1) + (1.0 / max(p, 1e-12))
    mu    = torch.clamp(dot / denom, min=0.0)
    u_cf  = u_ref - mu.unsqueeze(1) * a
    rho   = (mu / p).unsqueeze(1)                                   # (B,1)

    if mode == "cf":
        return u_cf, rho

    # ----- hybrid: clamp to model box, then verify CLF and fallback if needed -----
    u_try = u_cf
    if hasattr(ctrl.dynamics_model, "control_limits"):
        u_hi, u_lo = ctrl.dynamics_model.control_limits
        u_hi = u_hi.view(1, -1).to(u_try)
        u_lo = u_lo.view(1, -1).to(u_try)
        u_try = torch.min(torch.max(u_try, u_lo), u_hi)

    # Check the CLF inequality after clamping: a^T u <= b + ρ (take minimal ρ >= 0)
    viol = ((a * u_try).sum(dim=1) - b).clamp_min(0.0)              # (B,)
    ok = (viol <= eps)
    if ok.all():
        rho_box = viol.unsqueeze(1)
        return u_try, rho_box

    # Fallback to your original cvxpylayers QP only for violating batch items
    bad = (~ok).nonzero(as_tuple=False).flatten()
    u_out = u_try.clone()
    r_out = viol.unsqueeze(1)

    u_fb, r_fb = ctrl.solve_CLF_QP(
        x[bad], u_ref=u_ref[bad], requires_grad=False
    )
    u_out[bad] = u_fb
    r_out[bad] = r_fb
    return u_out, r_out
def train_mixed_clbf(
    sys,
    epochs: int = 30,
    batch_state: int = 512,
    batch_roll: int = 64,
    horizon: float = 0.8,
    lr: float = 1e-4,
    pretrain_epochs: int = 5,
    events_per_epoch: int = 3,
    alter_amt_range=(0.8, 1.25),
    alter_tau: float = 0.02,
):
    torch.autograd.set_detect_anomaly(False)
    sys.compute_linearized_controller([sys.nominal_params])
    ctrl = make_neural_clbf(sys)
    opt = torch.optim.AdamW(ctrl.V_net.parameters(), lr=lr, weight_decay=1e-4)
    
    dt = float(sys.dt)
    K = int(round(horizon / dt))
    
    def sample_state_batch(B: int):
        device = sys.goal_point.device
        up, lo = sys.state_limits
        span = (up - lo)
        m_goal = max(1, int(0.2 * B))
        Xg = sys.goal_point.expand(m_goal, -1) + 0.05 * span * torch.randn(m_goal, sys.n_dims).type_as(sys.goal_point)
        m_safe = max(1, int(0.5 * B))
        m_bdry = B - m_goal - m_safe
        Xs = sys.sample_safe(m_safe).type_as(sys.goal_point)
        Xb = sys.sample_boundary(max(1, m_bdry)).type_as(sys.goal_point)
        X = torch.cat([Xg, Xs, Xb], dim=0)[:B]
        X = torch.max(torch.min(X, up.type_as(X)), lo.type_as(X))
        return X.to(device)
    
    for ep in range(epochs):
        # Stage A: state-only
        X = sample_state_batch(batch_state)
        losses = []

        # Compute V once for each usage
        if ep < pretrain_epochs:
            V_pretrain = ctrl.V(X)
            P = sys.P.type_as(X)
            P_b = P.unsqueeze(0).expand(X.size(0), -1, -1)
            x0 = sys.goal_point.type_as(X)
            Vp = 0.5 * torch.bmm((X - x0).unsqueeze(1), torch.bmm(P_b, (X - x0).unsqueeze(2))).squeeze()
            losses.append(("init_mse", F.mse_loss(V_pretrain, Vp)))

        # Boundary losses
        V = ctrl.V(X)
        c = 1.0
        goal_term = 10.0 * ctrl.V(sys.goal_point.type_as(X)).mean()
        safe_mask = sys.safe_mask(X)
        unsafe_mask = sys.unsafe_mask(X)
        
        safe_violation = F.relu(1e-2 + V[safe_mask] - c) if safe_mask.any() else V.new_tensor(0.0)
        unsafe_violation = F.relu(1e-2 + c - V[unsafe_mask]) if unsafe_mask.any() else V.new_tensor(0.0)
        
        losses += [
            ("goal", goal_term),
            ("safe", 100.0 * (safe_violation.mean() if safe_mask.any() else V.new_tensor(0.0))),
            ("unsafe", 100.0 * (unsafe_violation.mean() if unsafe_mask.any() else V.new_tensor(0.0))),
        ]

        # QP descent
        u_qp, r_qp = solve_clf_qp_fast(
    ctrl, X, u_ref=sys.u_eq.expand(X.size(0), -1), mode="hybrid"
)
        r_mean = r_qp.mean()
        
        V_descent = ctrl.V(X)
        Lf_V, Lg_V = ctrl.V_lie_derivatives(X)
        V_lin = (Lf_V[:, 0, :].squeeze(-1)
                 + torch.bmm(Lg_V[:, 0, :].unsqueeze(1), u_qp.unsqueeze(2)).squeeze())

        eps_lin = 0.1
        descent_lin = torch.relu(eps_lin + V_lin + ctrl.clf_lambda*V_descent).mean()
        losses += [("qp_relax", r_mean), ("descent_lin", descent_lin)]

        loss_A = sum(v for _, v in losses)
        opt.zero_grad()
        loss_A.backward()
        torch.nn.utils.clip_grad_norm_(ctrl.V_net.parameters(), max_norm=1.0)


        opt.step()
        print(f"[ep {ep+1:03d}] Stage A done", flush=True)

        # Stage B: rollouts (your original logic)
        alter = AlterManager(sys)
        import random
        for _ in range(events_per_epoch):
            bus = random.randrange(sys.n_gen)
            t0 = random.uniform(0.1*horizon, 0.9*horizon)
            amt = random.uniform(alter_amt_range[0], alter_amt_range[1])
            alter.add(bus_idx=bus, t=t0, amount=amt, tau=alter_tau)

        x = sample_state_batch(batch_roll)
        t = 0.0
        total_relax = x.new_tensor(0.0)
        total_desc_lin = x.new_tensor(0.0)
        total_desc_sim = x.new_tensor(0.0)

        for k in range(K):
# Make sure the QP “sees” the correct plant at the current time
            alter.apply(t)

            # Control is piecewise-constant on [t, t+dt]
            u, r = solve_clf_qp_fast(
    ctrl, x, u_ref=sys.u_eq.expand(x.size(0), -1), mode="hybrid"
)

            # One Runge–Kutta 4 step; applies ALTER inside at t, t+dt/2, t+dt
            x_next = rk4_step(sys, x, u, dt, t_now=t, alter=alter)
            # linearized decrease at x
            Vx = ctrl.V(x)
            Lf_V, Lg_V = ctrl.V_lie_derivatives(x)
            Vdot_lin = (Lf_V[:, 0, :].squeeze(-1)
                        + torch.bmm(Lg_V[:, 0, :].unsqueeze(1), u.unsqueeze(2)).squeeze())
            total_desc_lin = total_desc_lin + torch.relu(1.0 + Vdot_lin + ctrl.clf_lambda * Vx).mean()

            # simulated one-step decrease
            V_next = ctrl.V(x_next)
            eps_sim = 0.0
            total_desc_sim = total_desc_sim + torch.relu(eps_sim + (V_next - Vx) + ctrl.clf_lambda*dt*Vx).mean()
            total_relax = total_relax + r.mean()
            x = x_next.detach()

            t += dt

        loss_B = (total_relax + total_desc_lin + total_desc_sim) / K
        opt.zero_grad()
        loss_B.backward()
        opt.step()

        alter.reset()
        print(f"[ep {ep+1:03d}] StageA={float(loss_A):.3e}  StageB={float(loss_B):.3e}")
def train_clbf_through_alter(
    sys,
    epochs: int = 10,
    batch: int = 64,
    lr: float = 1e-4,
    horizon: float = 1.0,
    events_per_epoch: int = 3,
    alter_amt_range=(0.7, 1.3),
    tau: float = 0.02,
):
    # Build a small neural CLF around your CLF‑QP


    # Ensure P,K exist
    sys.compute_linearized_controller([sys.nominal_params])

    # QP layer (fixed structure; V will be learned)
    base = CLFController(
        dynamics_model=sys,
        scenarios=[sys.nominal_params], 
        experiment_suite=ExperimentSuite([]),
        clf_lambda=1.0,
        clf_relaxation_penalty=10.0,
        controller_period=float(sys.dt),
        disable_gurobi=True,
    )

    # Replace quadratic V with a tiny MLP for training
    import torch.nn as nn
    n = sys.n_dims
    V_net = nn.Sequential(
        nn.Linear(n, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 1),
    ).to(sys.goal_point.device)
    for layer in ctrl.V_net:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            nn.init.zeros_(layer.bias)
    # Hook: override V() and V_with_jacobian() to use the net, keeping QP, Lf/Lg wrapped
    def V_with_jacobian_nn(x: torch.Tensor):
        x = x.requires_grad_(True)
        V = V_net(x).squeeze(-1)
        JV = torch.autograd.grad(
            V.sum(), x, retain_graph=True, create_graph=True
        )[0].reshape(x.shape[0], 1, n)
        return V, JV
    base.V_with_jacobian = V_with_jacobian_nn  # monkey‑patch
    base.V = lambda x: V_with_jacobian_nn(x)[0]

    opt = torch.optim.AdamW(V_net.parameters(), lr=lr, weight_decay=1e-4)

    dt = float(sys.dt)
    K = int(round(horizon / dt))

    for ep in range(epochs):
        # 1) sample initial states near the goal (perturbed)
        X = get_initial_states_batch(sys, batch, mode="perturbed", perturbation=0.1)

        # 2) build a fresh multi‑event ALTER schedule (shared by the batch)
        alter = AlterManager(sys)
        import random
        for _ in range(events_per_epoch):
            bus = random.randrange(sys.n_gen)            # 0..9
            t0  = random.uniform(0.1*horizon, 0.9*horizon)
            amt = random.uniform(amt_range[0], amt_range[1])
            alter.add(bus_idx=bus, t=t0, amount=amt, float32=float32)

        # 3) unroll with differentiable QP
        x = X.clone()
        t = 0.0
        total_relax = 0.0
        total_viol  = 0.0
        for k in range(K):
            alter.apply(t)  # writes time‑varying sP/sQ to sys

            u, r = base.solve_CLF_QP(
                x, u_ref=sys.u_eq.expand(x.shape[0], -1), requires_grad=True
            )                               # r: (B, n_scenarios)
            f, g = sys.control_affine_dynamics(x)
            if f.dim() == 3 and f.size(-1) == 1:
                f = f.squeeze(-1)
            xdot = f + torch.bmm(g, u.unsqueeze(-1)).squeeze(-1)
            x = x + dt * xdot
            t += dt

            # CLF decrease (linearized) and QP relaxation
            LfV, LgV = base.V_lie_derivatives(x)
            V = base.V(x)
            Vdot_lin = LfV[:, 0, :].squeeze(-1) + torch.bmm(
                LgV[:, 0, :].unsqueeze(1), u.unsqueeze(2)
            ).squeeze(-1).squeeze(-1)
            viol = torch.relu(1.0 + Vdot_lin + base.clf_lambda * V)  # eps=1.0
            total_viol  = total_viol + viol.mean()
            total_relax = total_relax + r.mean()

        loss = total_viol / K + total_relax / K
        opt.zero_grad(); loss.backward(); opt.step()
        alter.reset()
        print(f"[train ep {ep+1}/{epochs}] loss={float(loss):.4e}")

class _NullDataModule:
    """Minimal stub so we can instantiate NeuralCLBFController v2 without Lightning loops."""
    def prepare_data(self): pass
    def setup(self, *args, **kwargs): pass
    def train_dataloader(self): return None
    def val_dataloader(self): return None
    def test_dataloader(self): return None

def make_neural_clbf(sys):
    # Compute P matrix
    sys.compute_linearized_controller([sys.nominal_params])
    
    # Use regular CLFController (no Lightning dependencies)
    ctrl = CLFController(
        dynamics_model=sys,
        scenarios=[sys.nominal_params],
        experiment_suite=ExperimentSuite([]),
        clf_lambda=1.0,
        clf_relaxation_penalty=100.0,
        controller_period=float(sys.dt),
        disable_gurobi=True,
    )
    
    # Add a neural network for V
    n = sys.n_dims
    ctrl.V_net = nn.Sequential(
        nn.Linear(n, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 1),
    ).to(sys.goal_point.device)
    
    # Override V methods to use neural net + quadratic
    def V_with_jacobian_nn(x: torch.Tensor):
        x = x.requires_grad_(True)
        v_nn = ctrl.V_net(x).squeeze(-1)
        v_nn = 0.5 * v_nn.pow(2)  # Always positive
        
        # Add quadratic term
        P = sys.P.type_as(x)
        x0 = sys.goal_point.type_as(x)
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        P_b = P.unsqueeze(0).expand(x.shape[0], -1, -1)
        v_quad = 0.5 * torch.bmm((x - x0).unsqueeze(1), torch.bmm(P_b, (x - x0).unsqueeze(2))).squeeze()
        
        V = v_nn + v_quad
        JV = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
        return V, JV.reshape(x.shape[0], 1, n)
    
    ctrl.V_with_jacobian = V_with_jacobian_nn
    ctrl.V = lambda x: V_with_jacobian_nn(x)[0]
    
    return ctrl
def train_neural_clbf_epoch(
    sys, ctrl: NeuralCLBFController, batch_size: int, steps: int, optimizer: torch.optim.Optimizer
):
    """
    One epoch of instantaneous CLBF training:
    - sample B states around the goal (optionally perturb),
    - compute CLF-QP descent loss at those points (no BPTT through time),
    - backprop to V-net parameters.
    """
    ctrl.train()
    device = sys.goal_point.device
    # Sample batch of initial states
    X = get_initial_states_batch(sys, batch_size, which="perturbed", perturbation=0.1).to(device)

    optimizer.zero_grad(set_to_none=True)

    # Descent loss (asks cvxpylayers QP for u, differentiably)
    # v2: losses = ctrl.descent_loss(x, accuracy=False, requires_grad=True)
    losses = ctrl.descent_loss(X, accuracy=False, requires_grad=True)
    total = sum(v for (_, v) in losses)

    total.backward()
    optimizer.step()

    return float(total.detach().cpu()), {name: float(val.detach().cpu()) for (name, val) in losses}

# -------------------- Helpers --------------------

def ensure_batch(t: torch.Tensor) -> torch.Tensor:
    """Make (B,D) by adding batch dim if needed."""
    return t if t.dim() == 2 else t.unsqueeze(0)

from dataclasses import dataclass

@dataclass
class DroopConfig:
    kpf: float = 0.3     # P–f droop slope  (pu command per pu freq)
    kvv: float = 3.0     # Q–V droop slope  (pu command per pu voltage)
    uP_max: float = 0.1  # max |u_P| (command magnitude, pu)
    uQ_max: float = 0.1  # max |u_Q| (command magnitude, pu)
    use_vref: bool = False  # False: reference = sys.Vset; True: use sys.Vref

def monotone_linear_u(sys: IEEE39ControlAffineDAE, x: torch.Tensor, cfg: DroopConfig):
    """
    Monotone linear control law with saturation:
       u_P = -kpf * (omega - 1)
       u_Q = -kvv * (V - Vref)    with V from algebraic solve
    Returns: (u_cmd[1,2n], V[n], theta[n])
    """
    # Unpack current state (batch==1)
    delta_rel, omega, Eqp, Efd, Pm, Pvalve, Ppv, Qpv = sys._unpack_state(x[0])
    delta = sys._angle_reconstruct(delta_rel)

    # Solve algebraic network to get bus voltages/angles
    V, theta = sys._solve_kcl_newton(delta, Eqp, Ppv, Qpv)

    # Reference voltages for droop (choose Vset or current Vref)
    vref = (sys.Vref if (cfg.use_vref and hasattr(sys, "Vref")) else sys.Vset).to(V.dtype).to(V.device)

    # Monotone laws (negative slope) + symmetric saturation on command
    uP = (-cfg.kpf * (omega - 1.0)).clamp(min=-cfg.uP_max, max=cfg.uP_max)
    uQ = (-cfg.kvv * (V - vref)).clamp(min=-cfg.uQ_max, max=cfg.uQ_max)

    u_cmd = torch.cat([uP, uQ], dim=-1).unsqueeze(0)  # shape (1, 2n)
    return u_cmd, V, theta

def trace_grad_nan_controls(sys, T: float = 0.05):
    """
    Progressive backward: identifies the earliest RK step whose backward produces
    non-finite grads in dL/dU. Keep T small (e.g., 0.05) for speed.
    """
    torch.autograd.set_detect_anomaly(False)

    dt = float(sys.dt)
    steps = int(round(T/dt))
    x = (sys.goal_point.clone() if sys.goal_point.dim()==2 else sys.goal_point.unsqueeze(0)).detach()
    x.requires_grad_(False)

    n_ctrl = sys.n_controls
    U = nn.Parameter(torch.zeros(steps, 1, n_ctrl, dtype=x.dtype, device=x.device))

    per_step_losses = []
    t = 0.0
    for k in range(steps):
        u = sys.u_eq.clone() + U[k]         # [1, n_ctrl]
        x = rk4_step(sys, x, u, dt, t_now=t, alter=None)
        _, omega, *_ = sys._unpack_state(x[0])
        per_step_losses.append((omega - 1.0).pow(2).mean() * dt)
        t += dt

    # Walk backward step-by-step to find the earliest failing step.
    for k in reversed(range(steps)):
        if U.grad is not None:
            U.grad.zero_()
        try:
            per_step_losses[k].backward(retain_graph=True)
        except RuntimeError as e:
            print(f"[TRACE] backward threw at step k={k} : {e}")
            return k
        gk = U.grad[k]
        if gk is None:
            print(f"[TRACE] no grad reached U[{k}] (allow_unused?)")
            continue
        if not torch.isfinite(gk).all():
            bad = (~torch.isfinite(gk)).nonzero(as_tuple=False).T
            print(f"[TRACE] non-finite grad at step k={k}; bad indices in dL/dU[{k}]: {bad if bad.numel() else 'n/a'}")
            return k
    print("[TRACE] all per-step backprops finite; if sum(losses).backward() NaNs, it’s cross-step interaction.")
    return None
def get_initial_state(sys: IEEE39ControlAffineDAE, which: str = "equilibrium", perturbation: float = 0.0) -> torch.Tensor:
    """
    Get initial state for simulation.

    which='equilibrium': Start from the equilibrium point (dx/dt = 0)
    which='perturbed'  : equilibrium plus small angle perturbation

    Returns a batch of size 1: x0 with shape (1, D).
    """
    # Start from the equilibrium point stored in the system
    x0 = ensure_batch(sys.goal_point.clone())

    # Debug snapshot of the starting state
    delta_rel, omega, Eqp, Efd, Pm, Pvalve, Ppv, Qpv = sys._unpack_state(x0[0])
    print(f"\n=== Initial State Debug ===")
    print(f"Starting from: {which}")
    print(f"||delta_rel||: {float(delta_rel.norm()):.4e}")
    print(f"||omega-1||:  {float((omega-1).norm()):.4e}")
    print(f"||Eqp||:      {float(Eqp.norm()):.4e}")
    print(f"||Efd||:      {float(Efd.norm()):.4e}")
    print(f"||Pm||:       {float(Pm.norm()):.4e}")
    print(f"||Ppv||:      {float(Ppv.norm()):.4e}")
    print(f"||Qpv||:      {float(Qpv.norm()):.4e}")

    # Quick equilibrium check (should be ~0)
    f, g = sys.control_affine_dynamics(x0, params=None)
    xdot_test = f + torch.bmm(g, sys.u_eq.unsqueeze(-1)).squeeze(-1)
    print(f"||xdot|| at initial state: {float(xdot_test.norm()):.4e}")
    print(f"===========================\n")

    if which == "perturbed" and perturbation != 0.0:
        # Add small perturbation to rotor angles and a tiny speed nudge
        delta_rel, omega, Eqp, Efd, Pm, Pvalve, Ppv, Qpv = sys._unpack_state(x0[0])
        delta_rel_perturbed = delta_rel + perturbation * torch.randn_like(delta_rel)
        omega_perturbed = omega + 0.01 * perturbation * torch.randn_like(omega)
        x0 = sys._pack_state(
            delta_rel=delta_rel_perturbed,
            omega=omega_perturbed,
            Eqp=Eqp.clone(),
            Efd=Efd.clone(),
            Pm=Pm.clone(),
            Pvalve=Pvalve.clone(),
            Ppv=Ppv.clone(),
            Qpv=Qpv.clone(),
        ).unsqueeze(0)
        print(f"Applied perturbation: max angle deviation = {float(delta_rel_perturbed.abs().max()):.4f} rad")

    return x0


# -------------------- ALTER (per-bus load change) --------------------

# Reduced generator buses retained by the model (original IEEE-39 numbering, 1-based):
# index:  0   1   2   3   4   5   6   7   8   9
# bus #: 31, 30, 32, 33, 34, 35, 36, 37, 38, 39
GEN_BUS_ORDER = [31, 30, 32, 33, 34, 35, 36, 37, 38, 39]  # 1-based original ids

@dataclass
class AlterEvent:
    bus_idx: int     # reduced index 0..9
    t: float         # event time [s]
    amount: float    # multiplicative factor; 1.05 => +5%
    tau: float  # duration of load change
    float32: float = 0.0 # 0 => hard step; >0 => smooth logistic with width tau

class AlterManager:
    """
    DAE-style ALTER for the control-affine model: scale the calibrated ZIP multipliers
    sP and sQ at selected reduced buses. The runtime ZIP law uses sP/sQ * PL/QL_base,
    so this is the direct analog of "increase load at bus i by X%".
    """
    def __init__(self, sys: IEEE39ControlAffineDAE):
        if not hasattr(sys, "sP") or not hasattr(sys, "sQ"):
            raise AttributeError("IEEE39ControlAffineDAE must expose 'sP' and 'sQ'.")
        self.sys = sys
        self.sP_base = sys.sP.detach().clone()
        self.sQ_base = sys.sQ.detach().clone()
        self.dtype = self.sP_base.dtype
        self.device = self.sP_base.device
        assert self.sP_base.shape == self.sQ_base.shape
        self.events: List[AlterEvent] = []

    def add(self, bus_idx: int, t: float, amount: float, tau: float = 0.0):
        if not (0 <= bus_idx < self.sP_base.numel()):
            raise IndexError(f"Reduced-bus index {bus_idx} out of range [0..{self.sP_base.numel()-1}]")
        self.events.append(AlterEvent(bus_idx=int(bus_idx), t=float(t), amount=float(amount), tau=float(tau)))
        print(f"[ALTER] scheduled: reduced-bus {bus_idx}  t={t}s  amount={amount}  tau={tau}")

    @staticmethod
    def _gate_scalar(t_now: float, t0: float, tau: float) -> float:
        if tau <= 0.0:
            return 1.0 if (t_now >= t0) else 0.0
        import math
        return 1.0 / (1.0 + math.exp(-(t_now - t0) / max(tau, 1e-9)))

    def apply(self, t_now: float):
        # start with identity scale (vector length 10)
        scale = torch.ones_like(self.sP_base, dtype=self.dtype, device=self.device)
        for e in self.events:
            g = self._gate_scalar(t_now, e.t, e.tau)         # scalar in [0,1]
            inc = 1.0 + g * (e.amount - 1.0)                 # multiplicative increase
            scale[e.bus_idx] = scale[e.bus_idx] * inc
        # write time-varying sP, sQ onto the system
        self.sys.sP = self.sP_base * scale
        self.sys.sQ = self.sQ_base * scale     # PD+QD scaling; if PD-only, set sQ to base

    def reset(self):
        self.sys.sP = self.sP_base
        self.sys.sQ = self.sQ_base

def monotone_linear_u_batched(sys: IEEE39ControlAffineDAE, X: torch.Tensor, cfg: DroopConfig):
    """
    Batched version of monotone droop controller.
    X: (B, D) states. Returns:
      u_cmd: (B, 2n), V: (B, n), theta: (B, n)
    """
    B = X.shape[0]
    n = sys.n_gen
    u_cmd_list, V_list, th_list = [], [], []

    # droop refs
    # If use_vref=True and sys has Vref, use it; else use Vset
    Vref_vec = (sys.Vref if (cfg.use_vref and hasattr(sys, "Vref")) else sys.Vset)

    for b in range(B):
        delta_rel, omega, Eqp, Efd, Pm, Pvalve, Ppv, Qpv = sys._unpack_state(X[b])
        delta = sys._angle_reconstruct(delta_rel)            # (n,)
        V, theta = sys._solve_kcl_newton(delta, Eqp, Ppv, Qpv)  # (n,)

        vref = Vref_vec.to(V.dtype).to(V.device)
        uP = (-cfg.kpf * (omega - 1.0)).clamp(min=-cfg.uP_max, max=cfg.uP_max)   # (n,)
        uQ = (-cfg.kvv * (V - vref)).clamp(min=-cfg.uQ_max, max=cfg.uQ_max)      # (n,)
        u_cmd_list.append(torch.cat([uP, uQ], dim=-1).unsqueeze(0))              # (1,2n)
        V_list.append(V.unsqueeze(0))
        th_list.append(theta.unsqueeze(0))

    u_cmd = torch.cat(u_cmd_list, dim=0)    # (B, 2n)
    V_all = torch.cat(V_list, dim=0)        # (B, n)
    th_all = torch.cat(th_list, dim=0)      # (B, n)
    return u_cmd, V_all, th_all
def get_initial_states_batch(sys: IEEE39ControlAffineDAE, B: int, which: str = "equilibrium", perturbation: float = 0.0):
    """Return a batch (B, D) of initial states."""
    x0_1 = ensure_batch(sys.goal_point.clone())  # (1, D)
    if which == "equilibrium" or perturbation == 0.0:
        return x0_1.expand(B, -1).clone()

    X = x0_1.expand(B, -1).clone()
    for b in range(B):
        delta_rel, omega, Eqp, Efd, Pm, Pvalve, Ppv, Qpv = sys._unpack_state(X[b])
        delta_rel_b = delta_rel + perturbation * torch.randn_like(delta_rel)
        omega_b     = omega + 0.01 * perturbation * torch.randn_like(omega)
        X[b] = sys._pack_state(delta_rel=delta_rel_b, omega=omega_b, Eqp=Eqp, Efd=Efd, Pm=Pm, Pvalve=Pvalve, Ppv=Ppv, Qpv=Qpv)
    return X
@torch.no_grad()
def simulate_flat_run_batched(
    sys: IEEE39ControlAffineDAE,
    T_final: float,
    B: int = 256,
    use_monotone: bool = True,
    droop_cfg: DroopConfig = DroopConfig(),
    which_init: str = "equilibrium",
    perturbation: float = 0.0,
    alter_events: Optional[List[AlterEvent]] = None,
    pv_step: float = 0.0,
    t_step: float = 1.0,
):
    """
    Batched rollout with shared ALTER schedule for all batch items.
    Returns final X (B,D) and last control U (B,2n).
    """
    device = sys.goal_point.device
    dt = float(sys.dt)
    N = int(round(T_final / dt))

    X = get_initial_states_batch(sys, B, which=which_init, perturbation=perturbation).to(device)
    U = sys.u_eq.expand(B, -1).clone().to(device)

    # optional ALTER scheduler (shared across batch)
    alter_mgr = AlterManager(sys) if (alter_events and len(alter_events) > 0) else None
    if alter_mgr is not None:
        for e in alter_events:
            alter_mgr.add(e.bus_idx, e.t, e.amount, e.tau)

    t = 0.0
    for k in range(N):
        t_next = (k + 1) * dt

        # optional PV step at t_step
        if (pv_step != 0.0) and (t < t_step <= t_next):
            # broadcast step to all samples on selected PV channels
            U[:, :sys.n_gen] = U[:, :sys.n_gen] + pv_step

        if use_monotone:
            dU, V_now, th_now = monotone_linear_u_batched(sys, X, droop_cfg)
            U_tot = sys.u_eq.expand(B, -1) + dU
        else:
            U_tot = sys.u_eq.expand(B, -1)

        X = rk4_step(sys, X, U_tot, dt, t_now=t, alter=alter_mgr)
        t = t_next

    if alter_mgr is not None:
        alter_mgr.reset()

    return X, U_tot

# -------------------- RK4 Integrator --------------------
def _safe_affine(f, g, u, tag):
    if not torch.isfinite(f).all(): raise RuntimeError(f"[{tag}] non-finite f")
    if not torch.isfinite(g).all(): raise RuntimeError(f"[{tag}] non-finite g")
    if not torch.isfinite(u).all(): raise RuntimeError(f"[{tag}] non-finite u")
    if f.dim() == 3 and f.size(-1) == 1:
        f = f.squeeze(-1)
    out = torch.bmm(g, u.unsqueeze(-1)).squeeze(-1)
    return f + out

def rk4_step(sys: IEEE39ControlAffineDAE, x: torch.Tensor, u: torch.Tensor, dt: float,
             t_now: Optional[float] = None, alter: Optional[AlterManager] = None) -> torch.Tensor:
    """One RK4 step with algebraic solve inside sys.control_affine_dynamics()."""
    if alter is not None and t_now is not None:
        alter.apply(t_now)


    # Enforce system-level control limits (box) on u = [uP, uQ]
    if getattr(sys, "enforce_control_limits", True):
        u_hi, u_lo = sys.control_limits  # NOTE: system returns (upper, lower)
        u = torch.min(torch.max(u, u_lo.view(1, -1).to(u)), u_hi.view(1, -1).to(u))

        # Optional: smooth projection to a per-bus PQ capability circle
        if getattr(sys, "enforce_pv_circle", False):
            def _soft_project_circle(uP, uQ, Smax, beta=30.0):
                r = torch.sqrt(uP**2 + uQ**2 + 1e-12)
                t = torch.tanh(beta * (r - Smax))  # ~0 below circle, ~1 above
                alpha = (1 - t) + t * (Smax / (r + 1e-12))
                return alpha * uP, alpha * uQ
            n = sys.n_gen
            uP, uQ = u[..., :n], u[..., n:]
            # Build Smax from the system's box as a conservative circle
            uP_max = u_hi[:n]; uQ_max = u_hi[n:]
            Smax = torch.sqrt(uP_max**2 + uQ_max**2).view(1, -1).to(u)
            uP, uQ = _soft_project_circle(uP, uQ, Smax)
            u = torch.cat([uP, uQ], dim=-1)
            if u.dim() == 2 and u.shape[0] == 1 and x.shape[0] > 1:
                u = u.expand(x.shape[0], -1)
    f1, g1 = sys.control_affine_dynamics(x, params=None)
    k1 = _safe_affine(f1, g1, u, f"k1@t={t_now:.6f}")


    if alter is not None and t_now is not None:
        alter.apply(t_now + 0.5 * dt)
    f2, g2 = sys.control_affine_dynamics(x + 0.5 * dt * k1, params=None)
    k2 = _safe_affine(f2, g2, u, f"k2@t={t_now+0.5*dt:.6f}")

    if alter is not None and t_now is not None:
        alter.apply(t_now + 0.5 * dt)
    f3, g3 = sys.control_affine_dynamics(x + 0.5 * dt * k2, params=None)
    k3 = _safe_affine(f3, g3, u, f"k3@t={t_now+0.5*dt:.6f}")

    if alter is not None and t_now is not None:
        alter.apply(t_now + dt)
    f4, g4 = sys.control_affine_dynamics(x + dt * k3, params=None)
    k4 = _safe_affine(f4, g4, u, f"k4@t={t_now+dt:.6f}")

    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    if getattr(sys, "enforce_state_limits", False):
        x_hi, x_lo = sys.state_limits
        x_next = torch.min(torch.max(x_next, x_lo.view(1, -1).to(x_next)),
                                    x_hi.view(1, -1).to(x_next))
    return x_next

# -------------------- Audits & Simulation --------------------

def audit_now(sys: IEEE39ControlAffineDAE, x: torch.Tensor) -> Tuple[float, float, float, float]:
    """
    Compute ||xdot|| and ||KCL|| at the current state x (batch size 1).
    Also return ||dEfd/dt|| and ||dω/dt|| for quick diagnostics.
    """
    f, g = sys.control_affine_dynamics(x, params=None)
    xdot = f + torch.bmm(g, sys.u_eq.unsqueeze(-1)).squeeze(-1)

    # unpack and solve algebraic network for KCL residual
    n = sys.n_gen
    delta_rel, omega, Eqp, Efd, Pm, Pvalve, Ppv, Qpv = sys._unpack_state(x[0])
    delta = sys._angle_reconstruct(delta_rel)

    V = getattr(sys, "_last_V")
    theta = getattr(sys, "_last_theta")
    F = sys._kcl_residual(torch.cat([theta[1:], V], dim=0),
                        sys._angle_reconstruct(sys._unpack_state(x[0])[0]),
                        sys._unpack_state(x[0])[2],
                        sys._unpack_state(x[0])[6],
                        sys._unpack_state(x[0])[7])
    # slice indices (match sys packing)
    idx_omega = slice(n - 1, 2 * n - 1)
    idx_Efd = slice(3 * n - 1, 4 * n - 1)

    return float(xdot.norm()), float(F.norm()), float(xdot[0, idx_Efd].norm()), float(xdot[0, idx_omega].norm())
def simulate_flat_run(
    sys: IEEE39ControlAffineDAE,
    T_final: float,
    which_init: str = "equilibrium",
    perturbation: float = 0.0,
    pv_step: float = 0.0,
    t_step: float = 1.0,
    pv_buses: Optional[List[int]] = None,
    log_every: float = 0.005,
    print_every: float = 0.1,
    seed: int = 0,
    save_npz: str = "",
    plot: bool = True,
    alter_events: Optional[List[AlterEvent]] = None,
    ctrl: str = "monotone",
    droop_cfg: DroopConfig = DroopConfig(),
    clbf: CLFController = None,
    return_history: bool = False,
):
    """
    Simulate from equilibrium for T_final seconds with RK4.
    'ctrl':
        - "monotone": u = u_eq + u_droop
        - "clbf"    : u = u_CLF-QP (baseline u_ref = u_eq)
        - "droop_clf": u = u_QP with u_ref = u_eq + u_droop  (additive residual)
        - "const"   : u = u_eq
    """
    torch.manual_seed(seed)
    dt = float(sys.dt)
    if T_final <= 0 or dt <= 0:
        raise ValueError("T_final and dt must be positive.")
    N = int(round(T_final / dt))

    # initial state and input
    x = get_initial_state(sys, which=which_init, perturbation=perturbation)
    u = torch.zeros_like(sys.u_eq)

    pv_buses = list(pv_buses or range(sys.n_gen))

    # Shared ALTER schedule (if any)
    alter_mgr = AlterManager(sys) if (alter_events and len(alter_events) > 0) else None
    if alter_mgr is not None:
        for e in alter_events:
            alter_mgr.add(e.bus_idx, e.t, e.amount, e.tau)

    # Initial PF audit
    delta_rel0, _, Eqp0, _, _, _, Ppv0, Qpv0 = sys._unpack_state(x[0])
    delta0 = sys._angle_reconstruct(delta_rel0)
    V0, th0 = sys._solve_kcl_newton(delta0, Eqp0, Ppv0, Qpv0)
    rep0 = sys.audit_pf_at(V0, th0, delta0, Eqp0, Ppv0, Qpv0)
    print(f"t=0.000  ||k||={rep0['k_norm']:.3e}  max|ΔP|={rep0['dS_P_max']:.3e}  max|ΔQ|={rep0['dS_Q_max']:.3e}")

    # Log containers (include t=0)
    xn, Fn, dEfdn, domegan = audit_now(sys, x)
    times = [0.0]
    xdot_norm = [xn]
    kcl_norm = [Fn]
    dEfd_norm = [dEfdn]
    domega_norm = [domegan]
    states_history = [x.clone().detach().cpu().numpy()]
    V_hist = [V0.detach().cpu().numpy()]

    last_log_t = 0.0
    last_print_t = 0.0

    t = 0.0
    for k in range(N):
        t_next = (k + 1) * dt

        # optional PV step at time t_step
        if (pv_step != 0.0) and (t < t_step <= t_next):
            for i in pv_buses:
                u[0, i] += pv_step
            print(f"[{t_step:.3f}s] Applied PV P step of {pv_step:.4e} pu to buses {pv_buses}.")

        # ----- choose control -----
        if ctrl == "monotone":
            u_dev, V_now, th_now = monotone_linear_u(sys, x, droop_cfg)
            u_tot = sys.u_eq + u_dev

        elif ctrl == "clbf":
            if clbf is None:
                raise RuntimeError("ctrl='clbf' requires a CLF controller instance.")
            # Pure CLF-QP around u_ref = u_eq
            u_tot, _ = clbf.solve_CLF_QP(
                x, u_ref=sys.u_eq.expand(x.shape[0], -1), requires_grad=False
            )
            # For plotting/audit consistency
            V_now, th_now = sys._solve_kcl_newton(
                sys._angle_reconstruct(sys._unpack_state(x[0])[0]),
                sys._unpack_state(x[0])[2],
                sys._unpack_state(x[0])[6],
                sys._unpack_state(x[0])[7],
            )

        elif ctrl == "droop_clf":
            if clbf is None:
                raise RuntimeError("ctrl='droop_clf' requires a CLF controller instance.")
            # 1) droop
            u_droop, V_now, th_now = monotone_linear_u(sys, x, droop_cfg)  # (1, 2n)
            u_ref = sys.u_eq + u_droop
            # 2) CLF-QP uses u_ref = u_eq + u_droop  (=> additive residual)
            u_qp, _ = clbf.solve_CLF_QP(x, u_ref=u_ref, requires_grad=False)
            u_tot = u_qp  # this is u_eq + u_droop + u_CLF (possibly clipped by QP constraints)

        else:  # "const"
            u_tot = sys.u_eq.clone()
            V_now, th_now = sys._solve_kcl_newton(
                sys._angle_reconstruct(sys._unpack_state(x[0])[0]),
                sys._unpack_state(x[0])[2],
                sys._unpack_state(x[0])[6],
                sys._unpack_state(x[0])[7],
            )

        # one RK4 step
        x = rk4_step(sys, x, u_tot, dt, t_now=t, alter=alter_mgr)
        t = t_next

        # logging
        if (t - last_log_t) >= log_every - 1e-12:
            xn, Fn, dEfdn, domegan = audit_now(sys, x)
            times.append(t)
            xdot_norm.append(xn)
            kcl_norm.append(Fn)
            dEfd_norm.append(dEfdn)
            domega_norm.append(domegan)
            states_history.append(x.clone().detach().cpu().numpy())
            V_hist.append(V_now.detach().cpu().numpy())
            last_log_t = t

        if (t - last_print_t) >= print_every - 1e-12:
            if t > 0.0:
                print(f"t={t:6.3f}  ||xdot||={xn:.3e}  ||KCL||={Fn:.3e}  ||dEfd||={dEfdn:.3e}  ||dω||={domegan:.3e}")
                last_print_t = t

    if alter_mgr is not None:
        alter_mgr.reset()

    # final PF audit
    delta_rel, _, Eqp, _, _, _, Ppv, Qpv = sys._unpack_state(x[0])
    delta = sys._angle_reconstruct(delta_rel)
    V, th = sys._solve_kcl_newton(delta, Eqp, Ppv, Qpv)
    repF = sys.audit_pf_at(V, th, delta, Eqp, Ppv, Qpv)
    print(f"FINAL  ||k||={repF['k_norm']:.3e}  max|ΔP|={repF['dS_P_max']:.3e}  max|ΔQ|={repF['dS_Q_max']:.3e}")

    # arrays
    times = np.array(times)
    states_history = np.array(states_history)[:, 0, :]
    V_hist = np.array(V_hist)

    if plot:
        plot_results(sys, times, states_history, xdot_norm, kcl_norm, dEfd_norm, domega_norm,
                     pv_step, t_step, V_hist)

    if isinstance(save_npz, str) and len(save_npz) > 0:
        np.savez(save_npz,
                 t=times,
                 xdot_norm=np.array(xdot_norm),
                 kcl_norm=np.array(kcl_norm),
                 dEfd_norm=np.array(dEfd_norm),
                 domega_norm=np.array(domega_norm),
                 states=states_history,
                 V=V_hist)
        print(f"Saved time series to {save_npz}")

    if return_history:
        hist = dict(
            t=times,
            states=states_history,
            V=V_hist,
            xdot_norm=np.array(xdot_norm),
            kcl_norm=np.array(kcl_norm),
            dEfd_norm=np.array(dEfd_norm),
            domega_norm=np.array(domega_norm),
        )
        return x, u, hist

    return x, u

# -------------------- Plotting --------------------
def plot_compare(sys, H_a, H_b, label_a="Droop", label_b="Droop+CLF", t_step=None):
    """Overlay a few key metrics for two controllers."""
    import matplotlib.pyplot as plt
    t = H_a["t"]
    n = sys.n_gen

    states_a, states_b = H_a["states"], H_b["states"]
    omega_a = states_a[:, (n - 1):(2 * n - 1)]
    omega_b = states_b[:, (n - 1):(2 * n - 1)]

    # 1) Average frequency (Hz)
    plt.figure(figsize=(12, 4))
    fa = 60.0 * omega_a.mean(axis=1)
    fb = 60.0 * omega_b.mean(axis=1)
    plt.plot(t, fa, label=f"{label_a} (avg Hz)")
    plt.plot(t, fb, label=f"{label_b} (avg Hz)", linestyle="--")
    if t_step is not None:
        plt.axvline(t_step, linestyle=":", alpha=0.6)
    plt.xlabel("Time [s]"); plt.ylabel("Frequency [Hz]")
    plt.title("System Frequency Response")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.show()

    # 2) Residual norms
    plt.figure(figsize=(12, 4))
    plt.semilogy(t, H_a["xdot_norm"], label=f"{label_a} ||xdot||")
    plt.semilogy(t, H_b["xdot_norm"], label=f"{label_b} ||xdot||", linestyle="--")
    plt.semilogy(t, H_a["kcl_norm"], label=f"{label_a} ||KCL||")
    plt.semilogy(t, H_b["kcl_norm"], label=f"{label_b} ||KCL||", linestyle="--")
    if t_step is not None:
        plt.axvline(t_step, linestyle=":", alpha=0.6)
    plt.xlabel("Time [s]"); plt.ylabel("Norm")
    plt.title("Residual Norms")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.show()

def plot_results(sys, times, states, xdot_norm, kcl_norm, dEfd_norm, domega_norm,
                 pv_step=0.0, t_step=1.0, V_hist=None):
    """Plot simulation results in multiple subplots."""
    n = sys.n_gen

    # Unpack states per timestep
    delta_rel = states[:, :n-1]
    omega = states[:, n-1:2*n-1]
    Eqp = states[:, 2*n-1:3*n-1]
    Efd = states[:, 3*n-1:4*n-1]
    Pm = states[:, 4*n-1:5*n-1]
    Pvalve = states[:, 5*n-1:6*n-1]
    Ppv = states[:, 6*n-1:7*n-1]
    Qpv = states[:, 7*n-1:8*n-1]

    fig = plt.figure(figsize=(15, 10))

    # 1. Rotor angles (relative to gen 1)
    ax1 = plt.subplot(3, 3, 1)
    for i in range(min(5, n-1)):
        ax1.plot(times, delta_rel[:, i], label=f'Gen {i+2}')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Relative Angle (rad)')
    ax1.set_title('Rotor Angles (relative to Gen 1)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Rotor speeds
    ax2 = plt.subplot(3, 3, 2)
    for i in range(min(5, n)):
        ax2.plot(times, 60*omega[:, i], label=f'Gen {i+1}')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('frequency (Hz)')
    ax2.set_title('Rotor Speeds')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. E'q
    ax3 = plt.subplot(3, 3, 3)
    for i in range(min(5, n)):
        ax3.plot(times, Eqp[:, i], label=f'Gen {i+1}')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel("E'q (pu)")
    ax3.set_title('Transient Voltages')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Efd
    ax4 = plt.subplot(3, 3, 4)
    for i in range(min(5, n)):
        ax4.plot(times, Efd[:, i], label=f'Gen {i+1}')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Efd (pu)')
    ax4.set_title('Field Voltages')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Pm
    ax5 = plt.subplot(3, 3, 5)
    for i in range(min(5, n)):
        ax5.plot(times, Pm[:, i], label=f'Gen {i+1}')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Pm (pu)')
    ax5.set_title('Mechanical Power')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Residual norms
    ax6 = plt.subplot(3, 3, 6)
    ax6.semilogy(times, xdot_norm, 'b-', label='||xdot||')
    ax6.semilogy(times, kcl_norm, 'r--', label='||KCL||')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Norm')
    ax6.set_title('System Residuals')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. PV Active Power
    ax7 = plt.subplot(3, 3, 7)
    if np.any(Ppv != 0):
        for i in range(n):
            if np.any(Ppv[:, i] != 0):
                ax7.plot(times, Ppv[:, i], label=f'PV {i+1}')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('P_PV (pu)')
        ax7.set_title('PV Active Power')
        if pv_step != 0:
            ax7.axvline(t_step, color='k', linestyle=':', alpha=0.5, label=f'Step at t={t_step}')
        ax7.legend(fontsize=8)
    else:
        ax7.text(0.5, 0.5, 'No PV generation', ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('PV Active Power')
    ax7.grid(True, alpha=0.3)

    # 8. PV Reactive Power
    ax8 = plt.subplot(3, 3, 8)
    if np.any(Qpv != 0):
        for i in range(n):
            if np.any(Qpv[:, i] != 0):
                ax8.plot(times, Qpv[:, i], label=f'PV {i+1}')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Q_PV (pu)')
        ax8.set_title('PV Reactive Power')
        ax8.legend(fontsize=8)
    else:
        ax8.text(0.5, 0.5, 'No PV generation', ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title('PV Reactive Power')
    ax8.grid(True, alpha=0.3)

    # 9. dEfd and domega norms
    ax9 = plt.subplot(3, 3, 9)
    ax9.semilogy(times, dEfd_norm, 'g-', label='||dEfd/dt||')
    ax9.semilogy(times, domega_norm, 'm--', label='||dω/dt||')
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Norm')
    ax9.set_title('State Derivative Norms')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    plt.suptitle('IEEE 39-Bus Control-Affine Simulation Results', fontsize=14)
    plt.tight_layout()
    plt.show()

    # 10. Bus voltage magnitudes from algebraic solve (V_hist)
    if V_hist is not None:
        ax10 = plt.figure(figsize=(15, 3)).gca()
        for i in range(sys.n_gen):
            ax10.plot(times, V_hist[:, i], label=f'Bus {i+1}')
        ax10.set_xlabel('Time (s)')
        ax10.set_ylabel('|V| (pu)')
        ax10.set_title('Bus Voltages (from algebraic solve)')
        ax10.grid(True, alpha=0.3)
        ax10.legend(fontsize=7, ncol=5)
    plt.show()


def grad_check_controls(sys, T: float = 0.3):
    """
    Backprop through the rollout w.r.t. a *learnable control sequence* U[k].
    Verifies grads flow through RK4 + algebraic Newton solve.
    """

    dt = float(sys.dt)
    steps = int(round(T/dt))
    x = ensure_batch(sys.goal_point.clone())                # [1, n]
    n_ctrl = sys.n_controls
    device, dtype = x.device, x.dtype

    # Learnable control sequence: U[k] shape [steps, 1, n_ctrl]
    U = nn.Parameter(torch.zeros(steps, 1, n_ctrl, dtype=dtype, device=device))

    loss = torch.zeros((), dtype=dtype, device=device)
    t = 0.0

    for k in range(steps):
        u = sys.u_eq.clone() + U[k]                         # [1, n_ctrl]
        # IMPORTANT: do NOT detach x; keep the graph
        x = rk4_step(sys, x, u, dt, t_now=t, alter=None)
        # simple running loss: penalize rotor-speed deviations from 1 pu
        delta_rel, omega, *_ = sys._unpack_state(x[0])
        loss = loss + (omega - 1.0).pow(2).mean() * dt
        t += dt

    with torch.autograd.set_detect_anomaly(False):
        loss.backward()
    
    for tag, it, z_node in getattr(sys, "_dbg_z_nodes", []):
        if z_node.grad is not None and not torch.isfinite(z_node.grad).all():
            print(f"[KCL-GRAD] non-finite grad at Newton iter {it}")
    print(f"[GRAD u] loss={float(loss):.6e}  ||dL/dU||={float(U.grad.norm()):.6e}")
    return float(loss), U.grad.detach().cpu()

def grad_check_alter(sys, bus_reduced: int = 9, T: float = 0.3, t0: float = 0.2, tau: float = 0.05):
    """
    Backprop through the rollout w.r.t. a *smooth ALTER amount* on one reduced bus.
    Uses a torch-parameterized gate so grads flow w.r.t. 'amount' (and can flow w.r.t. t0, tau if desired).
    """
    dt = float(sys.dt)
    steps = int(round(T/dt))
    x = ensure_batch(sys.goal_point.clone())
    device, dtype = x.device, x.dtype

    # Base sP/sQ (keep copies to restore after test)
    sP0 = sys.sP.detach().clone()
    sQ0 = sys.sQ.detach().clone()

    # Learnable amount parameter (start near +5%)
    amount = nn.Parameter(torch.tensor(1.05, dtype=sP0.dtype, device=sP0.device))
    t0_t  = torch.tensor(float(t0), dtype=dtype, device=device)    # event time (fixed)
    tau_t = torch.tensor(float(tau), dtype=dtype, device=device)   # smooth width (fixed > 0)

    loss = torch.zeros((), dtype=dtype, device=device)
    t = 0.0
    for k in range(steps):
        # smooth gate s(t) = sigmoid((t - t0)/tau)
        tt = torch.tensor(float(t), dtype=dtype, device=device)
        s  = torch.sigmoid((tt - t0_t) / (tau_t + 1e-9))          # scalar in (0,1)

        # scale only the selected reduced bus
        scale = torch.ones_like(sP0)
        scale[bus_reduced] = 1.0 + s * (amount - 1.0)

        # write time-varying sP, sQ to the system
        sys.sP = sP0 * scale
        sys.sQ = sQ0 * scale   # PD+QD scaling; for PD-only use: sys.sQ = sQ0

        # integrate one step at u = u_eq (we only care about alter grads here)
        u = sys.u_eq.clone()
        x = rk4_step(sys, x, u, dt, t_now=t, alter=None)

        # running loss: penalize rotor-speed deviations
        _, omega, *_ = sys._unpack_state(x[0])
        loss = loss + (omega - 1.0).pow(2).mean() * dt
        t += dt
    with torch.autograd.set_detect_anomaly(False):
        loss.backward()
    print(f"[GRAD alter] loss={float(loss):.6e}  dL/d(amount)={float(amount.grad):.6e}")
    # restore base loads
    sys.sP = sP0; sys.sQ = sQ0
    return float(loss), float(amount.grad)

def grad_check_x0(sys, T: float = 0.3):
    """
    Backprop through the rollout w.r.t. the initial state x0.
    """
    dt = float(sys.dt)
    steps = int(round(T/dt))
    x0 = ensure_batch(sys.goal_point.clone()).requires_grad_(True)
    x = x0
    loss = torch.zeros((), dtype=x.dtype, device=x.device)
    t = 0.0
    for k in range(steps):
        u = sys.u_eq.clone()
        x = rk4_step(sys, x, u, dt, t_now=t, alter=None)
        # terminal loss: pull to goal
        loss = loss + (x - ensure_batch(sys.goal_point)).pow(2).mean()
        t += dt
    with torch.autograd.set_detect_anomaly(False):
        loss.backward()
    print(f"[GRAD x0] loss={float(loss):.6e}  ||dL/dx0||={float(x0.grad.norm()):.6e}")
    return float(loss), x0.grad.detach().cpu()
# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Flat-run simulator (flat dx/dt) for IEEE39ControlAffineDAE")
    ap.add_argument("--T", type=float, default=0.3, help="Final time [s]")
    ap.add_argument("--dt", type=float, default=0.001, help="Integrator dt [s] (also passed to system)")
    
    # MODIFIED: Changed default and help text for clarity
    ap.add_argument("--log-every", type=float, default=0.005,
                    help="Logging interval for PLOTS [s] (make small for smooth graphs)")
    
    # NEW: Argument for console printing
    ap.add_argument("--print-every", type=float, default=0.1,
                    help="Console print interval [s]")

    ap.add_argument("--train", action="store_true", help="Enable Neural-CLBF training harness")
    ap.add_argument("--train-steps", type=int, default=1, help="(Reserved) steps per epoch for rollout-style training")
    ap.add_argument("--no-plot", action="store_true", help="Disable plotting (recommended for large batches)")

    ap.add_argument("--init", type=str, default="equilibrium", choices=["equilibrium", "perturbed"],
                    help="Initial state: 'equilibrium' (dx/dt=0) or 'perturbed' (small angle deviation)")
    ap.add_argument("--perturbation", type=float, default=0.1,
                    help="Angle perturbation magnitude if init='perturbed' [rad]")
    ap.add_argument("--pv-step", type=float, default=0.0, help="PV active-power step [pu] at t_step")
    ap.add_argument("--t-step", type=float, default=1.0, help="PV step application time [s]")
    ap.add_argument("--pv-buses", type=int, nargs="*", default=None,
                    help="PV channels to step (reduced indices 0..9); default: all")
    ap.add_argument("--seed", type=int, default=0, help="PyTorch random seed")
    ap.add_argument("--save", type=str, default="", help="Optional .npz filename to save time series")
    ap.add_argument("--alter-amount", type=float, default=0.0,
                    help="Scale factor for load, e.g., 1.05 for +5%% (0 or 1.0 => no event).")
    ap.add_argument("--alter-buses", type=int, nargs="*", default=None,
                    help="Reduced-bus indices (0..9) to alter. If omitted and amount!=0, defaults to [9] (orig bus 39).")
    ap.add_argument("--orig-buses", type=int, nargs="*", default=None,
                    help="Original IEEE-39 bus numbers (1..39). Will be mapped to reduced indices via generator-bus list.")
    ap.add_argument("--alter-t", type=float, default=2.0, help="ALTER event time [s].")
    ap.add_argument("--alter-tau", type=float, default=0.0, help="Smooth width [s]; 0 => hard step.")
    ap.add_argument("--grad-check", type=str, default="none",
                    choices=["none", "u", "alter", "x0"],
                    help="Run an autograd check and exit: 'u' (controls), 'alter' (load amount), or 'x0' (initial state).")
    ap.add_argument("--gc-T", type=float, default=0.3, help="Horizon [s] for gradient check (keep small for memory).")
    ap.add_argument("--gc-bus", type=int, default=9, help="Reduced bus index (0..9) for --grad-check alter.")
    ap.add_argument("--gc-t0", type=float, default=0.2, help="Event time for --grad-check alter.")
    ap.add_argument("--gc-tau", type=float, default=0.05, help="Smooth width for --grad-check alter (must be >0 to get dL/dt0).")
    # Controller selection now includes CLF-QP ("clbf")
    ap.add_argument("--kpf", type=float, default=0.5)
    ap.add_argument("--kvv", type=float, default=5.0)
    ap.add_argument("--uP-max", type=float, default=0.2)
    ap.add_argument("--uQ-max", type=float, default=0.2)
    ap.add_argument("--use-vref", action="store_true")
    ap.add_argument(
        "--ctrl",
        type=str,
        default="droop_clf",
        choices=["const", "monotone", "clbf", "droop_clf"],
        help="'monotone' = pure droop; 'droop_clf' = droop + CLF-QP residual"
    )
    ap.add_argument(
        "--compare",
        action="store_true",
        help="Run pure droop and droop+CLF on the same disturbance and plot together"
    )
    ap.add_argument(
        "--load-clbf",
        type=str,
        default="",
        help="Optional path to saved V_net state_dict for the CLF-QP (additive case)"
    )

    # NEW: training flags
    ap.add_argument("--train-mode", type=str, default="none", choices=["none","mixed"],
                    help="Train mode: 'mixed' = state-only CLF pretrain + short rollouts with ALTER")
    ap.add_argument("--train-epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512, help="Stage-A (state-only) batch size")
    ap.add_argument("--horizon", type=float, default=0.8, help="Rollout horizon [s] for Stage-B")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--events-per-epoch", type=int, default=3)
    ap.add_argument("--alter-amt-range", type=float, nargs=2, default=[0.8, 1.25])

    args = ap.parse_args()
    cfg = DroopConfig(kpf=args.kpf, kvv=args.kvv, uP_max=args.uP_max, uQ_max=args.uQ_max,
                      use_vref=bool(args.use_vref))
    use_monotone = (args.ctrl == "monotone")


    sys = IEEE39ControlAffineDAE(
        nominal_params={
            "pv_ratio": [0.9, 0.9, 0.85, 0.9, 0.85, 0.85, 0.9, 0.9, 0.85, 0.3],
            "T_pv": (0.01, 0.01),
        },
        dt=float(args.dt),
        )

    clbf = None
    if (args.ctrl in ("clbf", "droop_clf")) or args.compare:
        clbf = make_neural_clbf(sys)  # your helper that attaches a V_net to CLFController

        # Optionally load a trained V_net
        if args.load_clbf:
            import torch
            sd = torch.load(args.load_clbf, map_location=sys.goal_point.device)
            clbf.V_net.load_state_dict(sd)
            print(f"[CLF] Loaded V_net state_dict from: {args.load_clbf}")
    # --- Switch PV to first‑order tracking and bind u_eq accordingly ---
    sys.enable_pv_first_order = True
    import inspect, hashlib
    if args.train_mode == "mixed":
        train_mixed_clbf(
            sys,
            epochs=int(args.train_epochs),
            batch_state=int(args.batch),
            batch_roll=min(32, int(0.125*args.batch)),   # e.g., 64 if batch=512
            horizon=float(args.horizon),
            lr=float(args.lr),
            events_per_epoch=int(args.events_per_epoch),
            alter_amt_range=tuple(args.alter_amt_range),
            alter_tau=float(args.alter_tau),
        )
        return
    # Make sure u_eq = [Ppv*, Qpv*] so PV is stationary at the goal
    gp = sys.goal_point[0]
    _, _, _, _, _, _, Ppv_eq, Qpv_eq = sys._unpack_state(gp)
    sys._u_eq = torch.cat([Ppv_eq, Qpv_eq], dim=0).unsqueeze(0)

    # Bind AVR/governor references at the runtime Newton point (dEfd=0, dPvalve=0 at t=0)
    if hasattr(sys, "_repair_equilibrium"):
        sys._repair_equilibrium()




    print("\n=== VERIF 0: module + file path ===")
    print("class module:", sys.__class__.__module__)
    print("class file  :", inspect.getfile(sys.__class__))

    print("\n=== VERIF 1: _f fingerprint ===")
    src_f = inspect.getsource(sys._f)
    print("len(_f source):", len(src_f))
    print("md5(_f):", hashlib.md5(src_f.encode()).hexdigest())
    print("contains 'enable_efd_soft_clip' (old path)?", "enable_efd_soft_clip" in src_f)
    print("contains 'enable_vr_soft_clip' (new Vr box)?", "enable_vr_soft_clip" in src_f)
    print("contains 'Vr_cmd' (new command limiting)?", "Vr_cmd" in src_f)

    print("\n=== VERIF 2: AVR mismatch at t=0 ===")
    x0 = (sys.goal_point if sys.goal_point.dim()==2 else sys.goal_point.unsqueeze(0)).clone()
    drel, omega, Eqp, Efd, Pm, Pvalve, Ppv, Qpv = sys._unpack_state(x0[0])
    V, th = sys._solve_kcl_newton(sys._angle_reconstruct(drel), Eqp, Ppv, Qpv)
    Vref = (sys.Vref if hasattr(sys, "Vref") else sys.Vset).to(V.dtype).to(V.device)
    mismatch = sys.Ka.to(V.dtype)*(Vref - V) - Efd
    print("||Ka*(Vref - V) - Efd|| =", float(mismatch.norm()))

    print("\n=== VERIF 3: PV modeling vs u_eq ===")
    # Are PV states integrators or first-order?
    is_pv_first_order = False
    # quick heuristic: check for the symbol in _f text
    if "enable_pv_first_order" in src_f:
        is_pv_first_order = bool(getattr(sys, "enable_pv_first_order", False))
    print("PV first-order enabled? ", is_pv_first_order)

    f, g = sys.control_affine_dynamics(x0, params=None)
    gu = torch.bmm(g, sys.u_eq.unsqueeze(-1)).squeeze(-1)  # [1, D]
    n = sys.n_gen
    idxP = slice(6*n-1, 7*n-1)  # Ppv rows in x
    idxQ = slice(7*n-1, 8*n-1)  # Qpv rows in x
    pv_inj_norm = float(gu[0, idxP].norm() + gu[0, idxQ].norm())
    print("|| (g @ u_eq) on PV rows || =", pv_inj_norm)

    print("\n=== VERIF 4: xdot decomposition at t=0 ===")
    xdot_full = f + gu
    print("||f|| (plant only) =", float(f.norm()), "  ||g@u_eq|| =", float(gu.norm()), "  ||f+g@u_eq|| =", float(xdot_full.norm()))


    print("\n=== VERIF 4b: xdot with current toggles and u_eq ===")
    x0 = (sys.goal_point if sys.goal_point.dim()==2 else sys.goal_point.unsqueeze(0)).clone()
    f, g = sys.control_affine_dynamics(x0, params=None)
    gu    = torch.bmm(g, sys.u_eq.unsqueeze(-1)).squeeze(-1)
    print("||f||=", float(f.norm()),
        "||g@u_eq||=", float(gu.norm()),
        "||f+g@u_eq||=", float((f + gu).norm()))

    # Break down by state blocks to see who is moving
    n = sys.n_gen
    sl = {
        "δ̇,ω̇": slice(0, 2*n-1),
        "E′q̇":  slice(2*n-1, 3*n-1),
        "Efḋ":  slice(3*n-1, 4*n-1),
        "Pṁ":   slice(4*n-1, 5*n-1),
        "Pval̇": slice(5*n-1, 6*n-1),
        "Ppv̇":  slice(6*n-1, 7*n-1),
        "Qpv̇":  slice(7*n-1, 8*n-1),
    }
    for name, s in sl.items():
        print(f"{name}  ||f+g@u_eq||_block = {float((f+gu)[0, s].norm()):.3e}")

    print("\n=== VERIF 5: limiter toggles (should be OFF unless you turned them on) ===")
    for k in ["enable_vr_soft_clip","enable_efd_box","enable_pref_box","enable_pvalve_box",
            "enable_pvalve_rate","enable_pm_box","enable_pm_rate","enable_q_limit_supervision"]:
        print(f"{k} =", bool(getattr(sys, k, False)))

    print("\n=== VERIF 6: Vr box contains Efd* (only relevant if enable_vr_soft_clip=True) ===")
    if getattr(sys, "enable_vr_soft_clip", False):
        # At the goal, to keep dEfd=0 with a Vr box, you need Vr_cmd(Eq) = Efd* inside [Vr_min, Vr_max]
        Efd_eq = Efd
        print("min(Vr_min - Efd_eq) =", float((sys.Vr_min.to(Efd_eq) - Efd_eq).min()))
        print("min(Efd_eq - Vr_max) =", float((Efd_eq - sys.Vr_max.to(Efd_eq)).min()))
    else:
        print("(Vr limiter is OFF; skip)")

    # 1) Choose which limiters to enable
    sys.enable_vr_soft_clip = False       # AVR command box
    sys.enable_efd_box      = False       # Efd ceiling/floor
    sys.enable_pref_box     = False       # droop output envelope
    sys.enable_pvalve_box   = False       # valve travel
    sys.enable_pvalve_rate  = False       # valve slew
    sys.enable_pm_box       = False       # mechanical power box
    sys.enable_pm_rate      = False
    sys.enable_q_limit_supervision = False

    # If you added PV first-order in _f, enable it AND set u_eq to [Ppv_eq, Qpv_eq]
    # sys.enable_pv_first_order = True
    # sys._u_eq = torch.cat([sys.goal_point[0][-2*sys.n_gen:-sys.n_gen],  # Ppv_eq slice
    #                        sys.goal_point[0][-sys.n_gen:]], dim=0).unsqueeze(0)

    # 2) Re-center references so the *limited* commands equal the state at the goal
    #    (Ka*(Vref - V*) == Efd*, Pref0 := Pm*)
    sys._repair_equilibrium()  # binds Vref and Pref0 to the Newton-consistent goal

    # 3) Now assert the goal lies inside the boxes


    if args.grad_check != "none":
        if args.grad_check == "u":
            trace_grad_nan_controls(sys, T=float(args.gc_T))
            grad_check_controls(sys, T=float(args.gc_T))
        elif args.grad_check == "alter":
            grad_check_alter(sys,
                            bus_reduced=int(args.gc_bus),
                            T=float(args.gc_T),
                            t0=float(args.gc_t0),
                            tau=float(args.gc_tau))
        elif args.grad_check == "x0":
            grad_check_x0(sys, T=float(args.gc_T))
        return

    events: List[AlterEvent] = []
    events = [
        AlterEvent(bus_idx=8, t=0.4, amount=1.20, tau=0.01),  
        AlterEvent(bus_idx=4, t=0.5, amount=1.25, tau=0.01),
        AlterEvent(bus_idx=6, t=0.6, amount=0.7, tau=0.01),
        AlterEvent(bus_idx=0, t=0.5, amount=1.15, tau=0.01),
        AlterEvent(bus_idx=9, t=0.5, amount=0.85, tau=0.01),
    ]
    
    if args.alter_amount not in (0.0, 1.0):
        if args.orig_buses:
            reduced_buses = []
            for b in args.orig_buses:
                if b not in GEN_BUS_ORDER:
                    raise ValueError(f"Original bus {b} is not in reduced generator set {GEN_BUS_ORDER}")
                reduced_buses.append(GEN_BUS_ORDER.index(b))
            print(f"[ALTER] original buses {args.orig_buses} -> reduced indices {reduced_buses}")
        else:
            reduced_buses = args.alter_buses if args.alter_buses is not None else [9]
        for rb in reduced_buses:
            events.append(AlterEvent(bus_idx=int(rb), t=float(args.alter_t),
                                    amount=float(args.alter_amount), tau=float(args.alter_tau)))
        print(f"[ALTER] events: {[(e.bus_idx, e.t, e.amount, e.tau) for e in events]}")

    t0 = time.time()


    B = int(args.batch)
    if args.train:
        train_clbf_through_alter(
            sys,
            epochs=int(args.train_epochs),
            batch=int(args.batch),
            lr=float(args.lr),
            horizon=float(args.horizon),
            events_per_epoch=int(args.events_per_epoch),
            alter_amt_range=tuple(args.alter_amt_range),
            alter_tau=float(args.alter_tau),
        )
        return

    if args.train:
        # ---- TRAINING HARNESS ----
        ctrl = make_neural_clbf(sys).to(sys.goal_point.device)
        opt = torch.optim.AdamW(ctrl.V_net.parameters(), lr=lr, weight_decay=1e-4)

        for ep in range(int(args.train_epochs)):
            loss_val, parts = train_neural_clbf_epoch(sys, ctrl, batch_size=B, steps=int(args.train_steps), optimizer=opt)
            print(f"[train] epoch {ep}  loss={loss_val:.4e}  parts={{" + ", ".join(f"{k}:{v:.3e}" for k,v in parts.items()) + "}}")
        return
    else:
        # ---- BATCHED SIM ----
        if B > 1:
            Xf, Uf = simulate_flat_run_batched(
                sys=sys,
                T_final=float(args.T),
                B=B,
                use_monotone=(args.ctrl == "monotone"),
                droop_cfg=cfg,
                which_init=args.init,
                perturbation=float(args.perturbation),
                alter_events=events,
                pv_step=float(args.pv_step),
                t_step=float(args.t_step),
            )
            print(f"[sim-batch] Done. Final X shape: {tuple(Xf.shape)}  U shape: {tuple(Uf.shape)}")
            if not args.no_plot:
                print("(Plotting is disabled for batch mode by default; use single-case mode for plots.)")
            return
        # fallback: your original single-case path
    if args.compare:
        # Run pure droop
        _, _, H_droop = simulate_flat_run(
            sys=sys,
            T_final=float(args.T),
            which_init=args.init,
            perturbation=float(args.perturbation),
            pv_step=float(args.pv_step),
            t_step=float(args.t_step),
            pv_buses=args.pv_buses,
            log_every=float(args.log_every),
            print_every=float(args.print_every),
            seed=int(args.seed),
            save_npz="",           # don't save per-run files in compare mode
            alter_events=events,
            ctrl="monotone",
            droop_cfg=cfg,
            clbf=None,
            plot=False,
            return_history=True,
        )
        # Run droop + CLF (same events)
        _, _, H_mix = simulate_flat_run(
            sys=sys,
            T_final=float(args.T),
            which_init=args.init,
            perturbation=float(args.perturbation),
            pv_step=float(args.pv_step),
            t_step=float(args.t_step),
            pv_buses=args.pv_buses,
            log_every=float(args.log_every),
            print_every=float(args.print_every),
            seed=int(args.seed),
            save_npz="",
            alter_events=events,
            ctrl="droop_clf",
            droop_cfg=cfg,
            clbf=clbf,
            plot=False,
            return_history=True,
        )
        # Overlay
        plot_compare(sys, H_droop, H_mix, label_a="Droop", label_b="Droop+CLF", t_step=float(args.t_step))
        return
    simulate_flat_run(
        sys=sys,
        T_final=float(args.T),
        which_init=args.init,
        perturbation=float(args.perturbation),
        pv_step=float(args.pv_step),
        t_step=float(args.t_step),
        pv_buses=args.pv_buses,
        log_every=float(args.log_every),
        print_every=float(args.print_every),
        seed=int(args.seed),
        save_npz=args.save,
        alter_events=events,
        ctrl=args.ctrl,          # <— now explicit
        droop_cfg=cfg,
        clbf=clbf
    )

    print(f"Done in {time.time() - t0:.2f} s.")
if __name__ == "__main__":
    main()
