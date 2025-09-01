#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flat-run time-domain simulator and CLBF trainer for IEEE39ControlAffineDAE.

This script integrates the control-affine IEEE39 model with RK4, supports
DAE-style ALTER events, and implements a robust, three-phase curriculum learning
strategy for training the Neural CLBF controller.

**Curriculum Learning Strategy:**
- Phase 1: V-function shaping (static properties). Focuses on learning a quadratic
           prior and satisfying safe/unsafe boundary conditions.
- Phase 2: Instantaneous descent. Introduces system dynamics and enforces the
           CLF condition locally at sampled points using the QP.
- Phase 3: Rollout fine-tuning. Uses short, disturbed trajectories to fine-tune
           the controller for temporal stability.

Usage examples:
  # Simulate for 3 seconds with a disturbance
  python -m neural_clbf.systems.ode_flatrun4_optimized --T 3.0 --alter-amount 1.1 --alter-t 0.5 --alter-tau 0.02

  # Train a new CLBF controller using the curriculum strategy
  python -m neural_clbf.systems.ode_flatrun4_optimized --train-mode curriculum --epochs 50 --phase1-epochs 10 --phase2-epochs 25 --lr 2e-4
"""
import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pytorch_lightning as pl
# <--- FIX: Import the 'inspect' module for robust argument handling
import inspect

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
torch.autograd.set_detect_anomaly(False)
import torch.nn.functional as F
from neural_clbf.controllers.clf_controller import CLFController
from neural_clbf.systems.utils import ScenarioList
from neural_clbf.experiments import ExperimentSuite

# Import your system class (control-affine IEEE39 with algebraic KCL solve)
from neural_clbf.systems.IEEE39ControlAffineDAE298 import IEEE39ControlAffineDAE
# Import Neural CLBF v2 controller
from neural_clbf.controllers.neural_clbf_controller import NeuralCLBFController

# ---- Fast CLF-QP Solver ----
def _V_JV_f_g(ctrl, x):
    """Compute V, ∂V/∂x, f, g once."""
    V, JV = ctrl.V_with_jacobian(x)
    f, g  = ctrl.dynamics_model.control_affine_dynamics(x)
    if f.dim() == 3 and f.size(-1) == 1:
        f = f.squeeze(-1)
    return V, JV, f, g

def solve_clf_qp_fast(ctrl, x, u_ref=None, mode="hybrid", eps=1e-7):
    """
    Fast CLF-QP for the standard 1-constraint problem.
    mode:
      - "cf": pure closed-form.
      - "hybrid": closed-form, clamp to model box, fallback to cvxpylayers if needed.
    """
    B = x.shape[0]
    if u_ref is None:
        u_ref = ctrl.dynamics_model.u_eq.expand(B, -1).to(x)

    V, JV, f, g = _V_JV_f_g(ctrl, x)
    LfV = torch.bmm(JV, f.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    LgV = torch.bmm(JV, g).squeeze(1)

    a = LgV
    b = -(LfV + ctrl.clf_lambda * V)
    p = float(ctrl.clf_relaxation_penalty)

    dot   = (a * u_ref).sum(dim=1) - b
    denom = (a * a).sum(dim=1) + (1.0 / max(p, 1e-12))
    mu    = torch.clamp(dot / denom, min=0.0)
    u_cf  = u_ref - mu.unsqueeze(1) * a
    rho   = (mu / p).unsqueeze(1)

    if mode == "cf":
        return u_cf, rho

    u_try = u_cf
    if hasattr(ctrl.dynamics_model, "control_limits"):
        u_hi, u_lo = ctrl.dynamics_model.control_limits
        u_hi = u_hi.view(1, -1).to(u_try)
        u_lo = u_lo.view(1, -1).to(u_try)
        u_try = torch.min(torch.max(u_try, u_lo), u_hi)

    viol = ((a * u_try).sum(dim=1) - b).clamp_min(0.0)
    ok = (viol <= eps)
    if ok.all():
        rho_box = viol.unsqueeze(1)
        return u_try, rho_box

    bad = (~ok).nonzero(as_tuple=False).flatten()
    u_out = u_try.clone()
    r_out = viol.unsqueeze(1)

    u_fb, r_fb = ctrl.solve_CLF_QP(
        x[bad], u_ref=u_ref[bad], requires_grad=False
    )
    u_out[bad] = u_fb
    r_out[bad] = r_fb
    return u_out, r_fb

# ---- New Curriculum-Based Training Function ----
def train_clbf_curriculum(
    sys: IEEE39ControlAffineDAE,
    # Curriculum settings
    epochs: int = 50,
    phase1_epochs: int = 10,
    phase2_epochs: int = 20,
    # Batch and rollout settings
    batch_size: int = 256,
    rollout_batch_size: int = 32,
    rollout_horizon: float = 0.5,
    # Optimizer settings
    learning_rate: float = 2e-4,
    # Loss weights
    w_pretrain: float = 1.0,
    w_boundary: float = 10.0,
    w_descent: float = 0.5,
    w_qp_relax: float = 100.0,
    w_rollout_relax: float = 1.0,
    w_rollout_descent: float = 0.1,
    # CLF & QP parameters
    clf_lambda: float = 0.5,
    clf_relaxation_penalty: float = 100.0,
    # Disturbance settings for rollouts
    events_per_epoch: int = 3,
    alter_amt_range=(0.8, 1.25),
    alter_tau: float = 0.02,
):
    """
    Trains a Neural CLBF controller using a three-phase curriculum learning strategy.
    
    Phase 1: V-function shaping (static properties: quadratic prior, boundary conditions).
    Phase 2: Instantaneous descent (local dynamics via QP at sampled states).
    Phase 3: Rollout fine-tuning (temporal dynamics and disturbance rejection).
    """
    # <--- FIX: Added a validation block to check the curriculum schedule
    print("\n--- Curriculum Schedule ---")
    phase3_start = phase1_epochs + phase2_epochs
    if epochs <= phase1_epochs:
        print(f"[WARNING] Total epochs ({epochs}) <= Phase 1 duration ({phase1_epochs}). Only Phase 1 will run.")
    elif epochs <= phase3_start:
        print(f"[INFO] Phase 1 will run for {phase1_epochs} epochs.")
        print(f"[INFO] Phase 2 will run for {epochs - phase1_epochs} epochs.")
        print(f"[WARNING] Phase 3 will be skipped because total epochs ({epochs}) <= start epoch ({phase3_start}).")
    else:
        print(f"[INFO] Phase 1 will run for {phase1_epochs} epochs.")
        print(f"[INFO] Phase 2 will run for {phase2_epochs} epochs.")
        print(f"[INFO] Phase 3 will run for {epochs - phase3_start} epochs.")
    print("---------------------------\n")

    print("Starting CLBF Training with Curriculum Learning Strategy")

    # 1. Initialization
    torch.autograd.set_detect_anomaly(False)
    
    print("[INFO] Creating controller and computing LQR gains (this may take a moment)...")
    ctrl = make_neural_clbf(sys)
    ctrl.clf_lambda = clf_lambda
    ctrl.clf_relaxation_penalty = clf_relaxation_penalty
    print("[INFO] Controller created successfully.")

    optimizer = torch.optim.AdamW(ctrl.V_net.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    dt = float(sys.dt)
    K = int(round(rollout_horizon / dt))

    # Helper for sampling states
    def sample_state_batch(B: int):
        device = sys.goal_point.device
        up, lo = sys.state_limits
        span = up - lo
        
        m_goal = max(1, int(0.2 * B))
        m_safe = max(1, int(0.5 * B))
        m_bdry = B - m_goal - m_safe

        Xg = sys.goal_point.expand(m_goal, -1) + 0.05 * span * torch.randn(m_goal, sys.n_dims).type_as(sys.goal_point)
        Xs = sys.sample_safe(m_safe).type_as(sys.goal_point)
        Xb = sys.sample_boundary(max(1, m_bdry)).type_as(sys.goal_point)
        
        X = torch.cat([Xg, Xs, Xb], dim=0)[:B]
        X = torch.max(torch.min(X, up.type_as(X)), lo.type_as(X))
        return X.to(device)

    # 2. Training Loop
    for ep in range(epochs):
        phase = 1
        if ep >= phase1_epochs: phase = 2
        if ep >= phase1_epochs + phase2_epochs: phase = 3
        
        # <--- FIX: Added flush=True to ensure this prints immediately to the log file
        print(f"\n[Epoch {ep+1:03d}/{epochs}] --- PHASE {phase} --- LR: {optimizer.param_groups[0]['lr']:.3e}", flush=True)

        if phase < 3:
            X = sample_state_batch(batch_size)
            losses = []

            # --- V-Function Shaping Losses (Phase 1+) ---
            V = ctrl.V(X)
            P = sys.P.type_as(X)
            P_b = P.unsqueeze(0).expand(X.size(0), -1, -1)
            x0 = sys.goal_point.type_as(X)
            Vp = 0.5 * torch.bmm((X - x0).unsqueeze(1), torch.bmm(P_b, (X - x0).unsqueeze(2))).squeeze()
            losses.append(("pretrain_mse", w_pretrain * F.mse_loss(V, Vp)))

            c = 1.0
            goal_term = 10.0 * ctrl.V(sys.goal_point.type_as(X)).mean()
            safe_mask = sys.safe_mask(X)
            unsafe_mask = sys.unsafe_mask(X)
            
            safe_violation = F.relu(1e-2 + V[safe_mask] - c) if safe_mask.any() else V.new_tensor(0.0)
            unsafe_violation = F.relu(1e-2 + c - V[unsafe_mask]) if unsafe_mask.any() else V.new_tensor(0.0)
            
            losses.append(("goal", goal_term))
            losses.append(("safe_boundary", w_boundary * (safe_violation.mean() if safe_mask.any() else V.new_tensor(0.0))))
            losses.append(("unsafe_boundary", w_boundary * (unsafe_violation.mean() if unsafe_mask.any() else V.new_tensor(0.0))))

            if phase == 2:
                u_qp, r_qp = solve_clf_qp_fast(ctrl, X, u_ref=sys.u_eq.expand(X.size(0), -1), mode="hybrid")
                losses.append(("qp_relax", w_qp_relax * r_qp.mean()))
                
                Lf_V, Lg_V = ctrl.V_lie_derivatives(X)
                V_dot_lin = Lf_V[:, 0, :].squeeze(-1) + torch.bmm(Lg_V[:, 0, :].unsqueeze(1), u_qp.unsqueeze(2)).squeeze()
                descent_violation = torch.relu(V_dot_lin + ctrl.clf_lambda * V)
                losses.append(("descent_lin", w_descent * descent_violation.mean()))

            loss_total = sum(v for _, v in losses)
            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(ctrl.V_net.parameters(), max_norm=1.0)
            optimizer.step()

            loss_str = " | ".join([f"{name}={val:.3e}" for name, val in losses])
            # <--- FIX: Added flush=True to ensure this prints immediately to the log file
            print(f"  StateLoss: {loss_total:.3e} | {loss_str}", flush=True)
            scheduler.step(loss_total)

        if phase == 3:
            alter = AlterManager(sys)
            import random
            for _ in range(events_per_epoch):
                bus = random.randrange(sys.n_gen)
                t0 = random.uniform(0.1 * rollout_horizon, 0.9 * rollout_horizon)
                amt = random.uniform(alter_amt_range[0], alter_amt_range[1])
                alter.add(bus_idx=bus, t=t0, amount=amt, tau=alter_tau)

            x = sample_state_batch(rollout_batch_size)
            t = 0.0
            
            total_rollout_relax = x.new_tensor(0.0)
            total_rollout_descent_lin = x.new_tensor(0.0)

            for k in range(K):
                alter.apply(t)
                u, r = solve_clf_qp_fast(ctrl, x, u_ref=sys.u_eq.expand(x.size(0), -1), mode="hybrid")
                
                with torch.no_grad():
                    x_next_no_grad = rk4_step(sys, x, u, dt, t_now=t, alter=alter)
                
                x_next = rk4_step(sys, x, u, dt, t_now=t, alter=alter)
                
                Vx = ctrl.V(x)
                Lf_V, Lg_V = ctrl.V_lie_derivatives(x)
                Vdot_lin = Lf_V[:, 0, :].squeeze(-1) + torch.bmm(Lg_V[:, 0, :].unsqueeze(1), u.unsqueeze(2)).squeeze()

                total_rollout_relax += r.mean()
                total_rollout_descent_lin += torch.relu(Vdot_lin + ctrl.clf_lambda * Vx).mean()
                
                x = x_next_no_grad.detach()
                t += dt

            loss_rollout = (w_rollout_relax * total_rollout_relax + w_rollout_descent * total_rollout_descent_lin) / K
            
            optimizer.zero_grad()
            loss_rollout.backward()
            torch.nn.utils.clip_grad_norm_(ctrl.V_net.parameters(), max_norm=0.5)
            optimizer.step()

            alter.reset()
            # <--- FIX: Added flush=True to ensure this prints immediately to the log file
            print(f"  RolloutLoss: {loss_rollout:.3e} | Relax: {(total_rollout_relax/K):.3e} | Descent: {(total_rollout_descent_lin/K):.3e}", flush=True)

    print("Training finished.", flush=True)
    torch.save(ctrl.V_net.state_dict(), "trained_clbf_curriculum.pt")
    return ctrl


def make_neural_clbf(sys):
    # This now includes the heavy computation, so we expect a delay here.
    sys.compute_linearized_controller([sys.nominal_params])
    
    ctrl = CLFController(
        dynamics_model=sys,
        scenarios=[sys.nominal_params],
        experiment_suite=ExperimentSuite([]),
        clf_lambda=1.0,
        clf_relaxation_penalty=100.0,
        controller_period=float(sys.dt),
        disable_gurobi=True,
    )
    
    n = sys.n_dims
    ctrl.V_net = nn.Sequential(
        nn.Linear(n, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 1),
    ).to(sys.goal_point.device)
    
    def V_with_jacobian_nn(x: torch.Tensor):
        x_g = x.requires_grad_(True)
        v_nn = ctrl.V_net(x_g).squeeze(-1)
        v_nn = 0.5 * v_nn.pow(2)
        
        P = sys.P.type_as(x)
        x0 = sys.goal_point.type_as(x)
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        P_b = P.unsqueeze(0).expand(x.shape[0], -1, -1)
        v_quad = 0.5 * torch.bmm((x - x0).unsqueeze(1), torch.bmm(P_b, (x - x0).unsqueeze(2))).squeeze()
        
        V = v_nn + v_quad
        JV = torch.autograd.grad(V.sum(), x_g, create_graph=True)[0]
        return V, JV.reshape(x.shape[0], 1, n)
    
    ctrl.V_with_jacobian = V_with_jacobian_nn
    ctrl.V = lambda x: V_with_jacobian_nn(x)[0]
    
    return ctrl

# -------------------- Helpers & Simulation Logic (largely unchanged) --------------------

def ensure_batch(t: torch.Tensor) -> torch.Tensor:
    return t if t.dim() == 2 else t.unsqueeze(0)

@dataclass
class DroopConfig:
    kpf: float = 0.3
    kvv: float = 3.0
    uP_max: float = 0.1
    uQ_max: float = 0.1
    use_vref: bool = False

def monotone_linear_u(sys: IEEE39ControlAffineDAE, x: torch.Tensor, cfg: DroopConfig):
    delta_rel, omega, Eqp, Efd, Pm, Pvalve, Ppv, Qpv = sys._unpack_state(x[0])
    delta = sys._angle_reconstruct(delta_rel)
    V, theta = sys._solve_kcl_newton(delta, Eqp, Ppv, Qpv)
    vref = (sys.Vref if (cfg.use_vref and hasattr(sys, "Vref")) else sys.Vset).to(V.dtype).to(V.device)
    uP = (-cfg.kpf * (omega - 1.0)).clamp(min=-cfg.kpf, max=cfg.uP_max)
    uQ = (-cfg.kvv * (V - vref)).clamp(min=-cfg.uQ_max, max=cfg.uQ_max)
    u_cmd = torch.cat([uP, uQ], dim=-1).unsqueeze(0)
    return u_cmd, V, theta

def get_initial_state(sys: IEEE39ControlAffineDAE, which: str = "equilibrium", perturbation: float = 0.0) -> torch.Tensor:
    x0 = ensure_batch(sys.goal_point.clone())
    if which == "perturbed" and perturbation != 0.0:
        delta_rel, omega, Eqp, Efd, Pm, Pvalve, Ppv, Qpv = sys._unpack_state(x0[0])
        delta_rel_perturbed = delta_rel + perturbation * torch.randn_like(delta_rel)
        omega_perturbed = omega + 0.01 * perturbation * torch.randn_like(omega)
        x0 = sys._pack_state(
            delta_rel=delta_rel_perturbed, omega=omega_perturbed, Eqp=Eqp.clone(),
            Efd=Efd.clone(), Pm=Pm.clone(), Pvalve=Pvalve.clone(), Ppv=Ppv.clone(), Qpv=Qpv.clone(),
        ).unsqueeze(0)
    return x0

# -------------------- ALTER (per-bus load change) --------------------

GEN_BUS_ORDER = [31, 30, 32, 33, 34, 35, 36, 37, 38, 39]

@dataclass
class AlterEvent:
    bus_idx: int
    t: float
    amount: float
    tau: float

class AlterManager:
    def __init__(self, sys: IEEE39ControlAffineDAE):
        self.sys = sys
        self.sP_base = sys.sP.detach().clone()
        self.sQ_base = sys.sQ.detach().clone()
        self.events: List[AlterEvent] = []

    def add(self, bus_idx: int, t: float, amount: float, tau: float = 0.0):
        self.events.append(AlterEvent(bus_idx=int(bus_idx), t=float(t), amount=float(amount), tau=float(tau)))

    @staticmethod
    def _gate_scalar(t_now: float, t0: float, tau: float) -> float:
        if tau <= 0.0:
            return 1.0 if (t_now >= t0) else 0.0
        import math
        return 1.0 / (1.0 + math.exp(-(t_now - t0) / max(tau, 1e-9)))

    def apply(self, t_now: float):
        scale = torch.ones_like(self.sP_base)
        for e in self.events:
            g = self._gate_scalar(t_now, e.t, e.tau)
            inc = 1.0 + g * (e.amount - 1.0)
            scale[e.bus_idx] *= inc
        self.sys.sP = self.sP_base * scale
        self.sys.sQ = self.sQ_base * scale

    def reset(self):
        self.sys.sP = self.sP_base
        self.sys.sQ = self.sQ_base

# -------------------- RK4 Integrator --------------------
def _safe_affine(f, g, u, tag):
    if f.dim() == 3 and f.size(-1) == 1:
        f = f.squeeze(-1)
    return f + torch.bmm(g, u.unsqueeze(-1)).squeeze(-1)

def rk4_step(sys: IEEE39ControlAffineDAE, x: torch.Tensor, u: torch.Tensor, dt: float,
             t_now: Optional[float] = None, alter: Optional[AlterManager] = None) -> torch.Tensor:
    if alter is not None and t_now is not None:
        alter.apply(t_now)
    
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

    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# -------------------- Audits & Simulation --------------------

def audit_now(sys: IEEE39ControlAffineDAE, x: torch.Tensor) -> Tuple[float, float]:
    f, g = sys.control_affine_dynamics(x, params=None)
    xdot = f + torch.bmm(g, sys.u_eq.unsqueeze(-1)).squeeze(-1)

    delta_rel, _, Eqp, _, _, _, Ppv, Qpv = sys._unpack_state(x[0])
    delta = sys._angle_reconstruct(delta_rel)
    V, theta = sys._solve_kcl_newton(delta, Eqp, Ppv, Qpv)
    F = sys._kcl_residual(torch.cat([theta[1:], V], dim=0), delta, Eqp, Ppv, Qpv)
    
    return float(xdot.norm()), float(F.norm())

def simulate_flat_run(
    sys: IEEE39ControlAffineDAE, T_final: float, which_init: str = "equilibrium",
    perturbation: float = 0.0, log_every: float = 0.005, print_every: float = 0.1,
    seed: int = 0, save_npz: str = "", plot: bool = True, alter_events: Optional[List[AlterEvent]] = None,
    ctrl: str = "monotone", droop_cfg: DroopConfig = DroopConfig(), clbf: CLFController = None,
    return_history: bool = False,
):
    torch.manual_seed(seed)
    dt = float(sys.dt)
    N = int(round(T_final / dt))

    x = get_initial_state(sys, which=which_init, perturbation=perturbation)
    
    alter_mgr = AlterManager(sys) if alter_events else None
    if alter_mgr:
        for e in alter_events:
            alter_mgr.add(e.bus_idx, e.t, e.amount, e.tau)

    times, xdot_norm, kcl_norm, states_history, V_hist = [], [], [], [], []
    t = 0.0
    for k in range(N):
        # Use modulo arithmetic for logging to simplify logic
        if k % int(log_every / dt) == 0:
            xn, Fn = audit_now(sys, x)
            delta_rel, _, Eqp, _, _, _, Ppv, Qpv = sys._unpack_state(x[0])
            delta = sys._angle_reconstruct(delta_rel)
            V_now, _ = sys._solve_kcl_newton(delta, Eqp, Ppv, Qpv)
            times.append(t)
            xdot_norm.append(xn)
            kcl_norm.append(Fn)
            states_history.append(x.clone().detach().cpu().numpy())
            V_hist.append(V_now.detach().cpu().numpy())
        
        if k % int(print_every / dt) == 0:
             print(f"t={t:6.3f}  ||xdot||={xdot_norm[-1]:.3e}  ||KCL||={kcl_norm[-1]:.3e}", flush=True)

        u_tot = sys.u_eq.expand(x.shape[0], -1)
        if ctrl == "monotone":
            u_dev, _, _ = monotone_linear_u(sys, x, droop_cfg)
            u_tot += u_dev
        elif ctrl == "clbf" and clbf:
            u_tot, _ = clbf.solve_CLF_QP(x, requires_grad=False)
        elif ctrl == "droop_clf" and clbf:
            u_droop, _, _ = monotone_linear_u(sys, x, droop_cfg)
            u_ref = sys.u_eq + u_droop
            u_tot, _ = clbf.solve_CLF_QP(x, u_ref=u_ref, requires_grad=False)
        
        x = rk4_step(sys, x, u_tot, dt, t_now=t, alter=alter_mgr)
        t += dt

    if alter_mgr: alter_mgr.reset()

    times = np.array(times)
    states_history = np.array(states_history)[:, 0, :]
    V_hist = np.array(V_hist)

    if plot:
        plot_results(sys, times, states_history, xdot_norm, kcl_norm, V_hist)
    
    if save_npz:
        np.savez(save_npz, t=times, xdot_norm=np.array(xdot_norm), kcl_norm=np.array(kcl_norm), states=states_history, V=V_hist)
        print(f"Saved time series to {save_npz}")

    if return_history:
        return {"t": times, "states": states_history, "V": V_hist, "xdot_norm": xdot_norm, "kcl_norm": kcl_norm}
    
    return x, u_tot

# -------------------- Plotting --------------------
def plot_results(sys, times, states, xdot_norm, kcl_norm, V_hist=None):
    n = sys.n_gen
    omega = states[:, n-1:2*n-1]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.plot(times, 60*omega)
    ax1.set_title('Rotor Speeds'); ax1.set_ylabel('Frequency (Hz)'); ax1.grid(True)
    
    ax2.semilogy(times, xdot_norm, label='||xdot||')
    ax2.semilogy(times, kcl_norm, label='||KCL||')
    ax2.set_title('System Residuals'); ax2.set_ylabel('Norm'); ax2.grid(True); ax2.legend()
    
    if V_hist is not None:
        ax3.plot(times, V_hist)
        ax3.set_title('Bus Voltages'); ax3.set_ylabel('|V| (pu)'); ax3.grid(True)
    
    delta_rel = states[:, :n-1]
    ax4.plot(times, delta_rel)
    ax4.set_title('Rotor Angles (rel to Gen 1)'); ax4.set_ylabel('Angle (rad)'); ax4.grid(True)

    plt.tight_layout()
    plt.show()

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Simulator and CLBF trainer for IEEE39 DAE")
    # Sim args
    ap.add_argument("--T", type=float, default=3.0, help="Final time [s]")
    ap.add_argument("--dt", type=float, default=0.001, help="Integrator dt [s]")
    ap.add_argument("--log-every", type=float, default=0.005, help="Logging interval for plots [s]")
    ap.add_argument("--print-every", type=float, default=0.1, help="Console print interval [s]")
    ap.add_argument("--no-plot", action="store_true", help="Disable plotting")
    ap.add_argument("--init", type=str, default="perturbed", choices=["equilibrium", "perturbed"])
    ap.add_argument("--perturbation", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="", help="Optional .npz filename to save time series")
    # Disturbance args
    ap.add_argument("--alter-amount", type=float, default=0.0)
    ap.add_argument("--alter-buses", type=int, nargs="*", default=None)
    ap.add_argument("--alter-t", type=float, default=0.5)
    ap.add_argument("--alter-tau", type=float, default=0.02)
    # Controller args
    ap.add_argument("--ctrl", type=str, default="droop_clf", choices=["const", "monotone", "clbf", "droop_clf"])
    ap.add_argument("--load-clbf", type=str, default="", help="Path to saved V_net state_dict")
    # Training args
    ap.add_argument("--train-mode", type=str, default="none", choices=["none", "curriculum"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--phase1-epochs", type=int, default=10)
    ap.add_argument("--phase2-epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--rollout-batch-size", type=int, default=32)
    ap.add_argument("--rollout-horizon", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--w-pretrain", type=float, default=1.0)
    ap.add_argument("--w-boundary", type=float, default=10.0)
    ap.add_argument("--w-descent", type=float, default=0.5)
    ap.add_argument("--w-qp-relax", type=float, default=100.0)
    ap.add_argument("--w-rollout-relax", type=float, default=1.0)
    ap.add_argument("--w-rollout-descent", type=float, default=0.1)
    ap.add_argument("--clf-lambda", type=float, default=0.5)
    ap.add_argument("--clf-relax-penalty", type=float, default=100.0)
    ap.add_argument("--events-per-epoch", type=int, default=3)
    ap.add_argument("--alter-amt-range", type=float, nargs=2, default=[0.8, 1.25])
    
    args = ap.parse_args()
    
    # <--- FIX: Added diagnostic print for system initialization
    print("[INFO] Initializing IEEE 39-bus system model...")
    sys = IEEE39ControlAffineDAE(
        nominal_params={"pv_ratio": [0.9]*9 + [0.3], "T_pv": (0.01, 0.01)},
        dt=float(args.dt)
    )
    sys._repair_equilibrium()
    print("[INFO] System model initialized.")

    if args.train_mode == "curriculum":
        # <--- FIX: This block programmatically filters args for the training function
        # to prevent passing unexpected arguments like 'T'.
        
        # Get the signature of the training function
        sig = inspect.signature(train_clbf_curriculum)
        
        # Create a dictionary of only the arguments that the function expects
        train_args = {}
        for param in sig.parameters.values():
            if hasattr(args, param.name):
                train_args[param.name] = getattr(args, param.name)
        
        # Call the training function with the filtered arguments
        train_clbf_curriculum(sys, **train_args)
        return

    # --- Simulation Logic ---
    events = []
    if args.alter_amount not in (0.0, 1.0):
        buses = args.alter_buses if args.alter_buses is not None else [9]
        for b in buses:
            events.append(AlterEvent(bus_idx=b, t=args.alter_t, amount=args.alter_amount, tau=args.alter_tau))
    
    clbf_controller = None
    if "clf" in args.ctrl:
        clbf_controller = make_neural_clbf(sys)
        if args.load_clbf:
            clbf_controller.V_net.load_state_dict(torch.load(args.load_clbf, map_location=sys.goal_point.device))
            print(f"Loaded CLBF from {args.load_clbf}")

    t0 = time.time()
    simulate_flat_run(
        sys=sys, T_final=args.T, which_init=args.init, perturbation=args.perturbation,
        log_every=args.log_every, print_every=args.print_every, seed=args.seed,
        save_npz=args.save, plot=not args.no_plot, alter_events=events,
        ctrl=args.ctrl, droop_cfg=DroopConfig(), clbf=clbf_controller
    )
    print(f"Done in {time.time() - t0:.2f} s.")

if __name__ == "__main__":
    main()