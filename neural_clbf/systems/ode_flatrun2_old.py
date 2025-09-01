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

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
torch.autograd.set_detect_anomaly(False)  # must be ON before building the forward graph

# Import your system class (control-affine IEEE39 with algebraic KCL solve)
from neural_clbf.systems.IEEE39ControlAffineDAE_old import IEEE39ControlAffineDAE


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
    tau: float = 0.0 # 0 => hard step; >0 => smooth logistic with width tau

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


# -------------------- RK4 Integrator --------------------
def _safe_affine(f, g, u, tag):
    if not torch.isfinite(f).all(): raise RuntimeError(f"[{tag}] non-finite f")
    if not torch.isfinite(g).all(): raise RuntimeError(f"[{tag}] non-finite g")
    if not torch.isfinite(u).all(): raise RuntimeError(f"[{tag}] non-finite u")
    out = torch.bmm(g, u.unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(out).all(): raise RuntimeError(f"[{tag}] non-finite g@u")
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
    k4 = f4 + torch.bmm(g4, u.unsqueeze(-1)).squeeze(-1)

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
    V, theta = sys._solve_kcl_newton(delta, Eqp, Ppv, Qpv)
    z = torch.cat([theta[1:], V], dim=0)
    F = sys._kcl_residual(z, delta, Eqp, Ppv, Qpv)

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
    print_every: float = 0.5,
    seed: int = 0,
    save_npz: bool = True,
    plot: bool = True,
    alter_events: Optional[List[AlterEvent]] = None,
    use_monotone: bool = True,
    droop_cfg: DroopConfig = DroopConfig(),
):
    """
    Simulate from equilibrium for T_final seconds with RK4.
    """
    torch.manual_seed(seed)
    dt = float(sys.dt)
    if T_final <= 0 or dt <= 0:
        raise ValueError("T_final and dt must be positive.")
    N = int(round(T_final / dt))

    x = get_initial_state(sys, which=which_init, perturbation=perturbation)
    u = torch.zeros_like(sys.u_eq)

    pv_buses = list(pv_buses or range(sys.n_gen))

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

    # FIXED BLOCK: All lists are now correctly initialized with their t=0 values.
    xn, Fn, dEfdn, domegan = audit_now(sys, x)
    times = [0.0]
    xdot_norm = [xn]
    kcl_norm = [Fn]
    dEfd_norm = [dEfdn]
    domega_norm = [domegan]
    states_history = [x.clone().detach().cpu().numpy()]
    V_hist = [V0.detach().cpu().numpy()]

    last_log_t = 0.0 # Can be 0.0 now since t=0 is already logged
    last_print_t = 0.0
    
    # Main loop
    t = 0.0
    for k in range(N):
        t_next = (k + 1) * dt

        if (pv_step != 0.0) and (t < t_step <= t_next):
            for i in pv_buses:
                u[0, i] += pv_step
            print(f"[{t_step:.3f}s] Applied PV P step of {pv_step:.4e} pu to buses {pv_buses}.")
        if use_monotone:
            u_dev, V_now, th_now = monotone_linear_u(sys, x, droop_cfg)  # Δu
            u_tot = sys.u_eq + u_dev                                     # bias by u_eq
        else:
            u_tot = sys.u_eq.clone()                                     # hold equilibrium input
            V_now, th_now = sys._solve_kcl_newton(
                sys._angle_reconstruct(sys._unpack_state(x[0])[0]),
                sys._unpack_state(x[0])[2],
                sys._unpack_state(x[0])[6],
                sys._unpack_state(x[0])[7],
            )
        x = rk4_step(sys, x, u_tot, dt, t_now=t, alter=alter_mgr)

        t = t_next
        
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
            if t > 0.0: # Avoid re-printing the t=0 line
                print(f"t={t:6.3f}  ||xdot||={xn:.3e}  ||KCL||={Fn:.3e}  ||dEfd||={dEfdn:.3e}  ||dω||={domegan:.3e}")
                last_print_t = t

    if alter_mgr is not None:
        alter_mgr.reset()

    delta_rel, _, Eqp, _, _, _, Ppv, Qpv = sys._unpack_state(x[0])
    delta = sys._angle_reconstruct(delta_rel)
    V, th = sys._solve_kcl_newton(delta, Eqp, Ppv, Qpv)
    repF = sys.audit_pf_at(V, th, delta, Eqp, Ppv, Qpv)
    print(f"FINAL  ||k||={repF['k_norm']:.3e}  max|ΔP|={repF['dS_P_max']:.3e}  max|ΔQ|={repF['dS_Q_max']:.3e}")

    times = np.array(times)
    states_history = np.array(states_history)[:, 0, :]
    V_hist = np.array(V_hist)

    if plot:
        plot_results(sys, times, states_history, xdot_norm, kcl_norm, dEfd_norm, domega_norm,
                     pv_step, t_step, V_hist)
    if save_npz:
        np.savez(save_npz, t=times, xdot_norm=np.array(xdot_norm), kcl_norm=np.array(kcl_norm),
                 dEfd_norm=np.array(dEfd_norm), domega_norm=np.array(domega_norm),
                 states=states_history, V=V_hist)
        print(f"Saved time series to {save_npz}")

    return x, u

# -------------------- Plotting --------------------

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
    ap.add_argument("--print-every", type=float, default=0.5,
                    help="Console print interval [s]")

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
    ap.add_argument("--ctrl", type=str, default="monotone", choices=["const","monotone"])
    ap.add_argument("--kpf", type=float, default=0.5)
    ap.add_argument("--kvv", type=float, default=5.0)
    ap.add_argument("--uP-max", type=float, default=0.2)
    ap.add_argument("--uQ-max", type=float, default=0.2)
    ap.add_argument("--use-vref", action="store_true")

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

    # --- Switch PV to first‑order tracking and bind u_eq accordingly ---
    sys.enable_pv_first_order = True
    import inspect, hashlib, torch

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
    simulate_flat_run(
        sys=sys,
        T_final=float(args.T),
        which_init=args.init,
        perturbation=float(args.perturbation),
        pv_step=float(args.pv_step),
        t_step=float(args.t_step),
        pv_buses=args.pv_buses,
        log_every=float(args.log_every),
        print_every=float(args.print_every), # NEW: Pass the new argument
        seed=int(args.seed),
        save_npz=args.save,
        alter_events=events,
        use_monotone=use_monotone, 
        droop_cfg=cfg
    )
    
    print(f"Done in {time.time() - t0:.2f} s.")
if __name__ == "__main__":
    main()
