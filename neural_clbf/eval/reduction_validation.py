"""
Reduction‑validation utilities
==============================

Usage
-----

    from neural_clbf.systems import SwingEquationSystem
    from neural_clbf.dimension_reduction.manager import select_reducer
    from neural_clbf.eval.reduction_validation import validate_reducer

    sys = SwingEquationSystem(n_nodes=10)
    data = sys.collect_random_trajectories(N_traj=8000, return_derivative=True)
    red  = select_reducer(sys, data["X"], data["dXdt"], d_max=10)

    metrics = validate_reducer(sys, red,
                               n_rollouts=50,
                               horizon=5.0,
                               dt=0.01,
                               input_mode="zero")   # or "random"
    print(metrics)
"""

from __future__ import annotations
import torch
from typing import Dict, Literal


# ------------------------------------------------------------------ #
# 1) helper — integrate latent ODE with explicit Euler
# ------------------------------------------------------------------ #
def rollout_rom(
    reducer,                       # any BaseReducer
    sys,                           # original SwingEquationSystem
    x0: torch.Tensor,              # (batch,n)
    horizon: float,
    dt: float,
    input_mode: Literal["zero", "random"] = "zero",
):
    """
    Integrate the reduced‑order model for `horizon` seconds using
    explicit Euler in the latent space (sufficient for validation).

    Returns
    -------
    x_rec  : reconstructed trajectory  (batch, T+1, n) including initial state
    """
    B, n = x0.shape
    T = int(horizon / dt)
    z = reducer.forward(x0)                           # initial latent
    traj = [reducer.inverse(z)]                       # store full‑state

    for _ in range(T):
        # ----- choose control input ------------------------------------ #
        if input_mode == "zero":
            u = torch.zeros(B, sys.n_controls, device=x0.device)
        elif input_mode == "random":
            u_min, u_max = sys.control_limits
            u = torch.rand(B, sys.n_controls, device=x0.device) \
                * (u_max - u_min) + u_min
        else:
            raise ValueError("input_mode must be 'zero' or 'random'")

        # ----- latent dynamics ----------------------------------------- #
        if hasattr(reducer, "dyn"):                   # OpInf etc.
            z_dot = reducer.dyn.forward(z, u)
        elif hasattr(sys, "_f"):                      # projected f
            x_full = reducer.inverse(z)
            
            # Get f(x) - should be (B, n_dims, 1)
            f_x = sys._f(x_full, params=sys.nominal_params)
            
            # Get Jacobian - should be (B, d, n)
            J = reducer.jacobian(x_full)
            
            # Compute z_dot = J @ f(x)
            # J is (B, d, n), f_x is (B, n, 1)
            # Use batch matrix multiply
            z_dot = torch.bmm(J, f_x).squeeze(-1)  # (B, d)
        else:
            raise RuntimeError("Reducer lacks latent dynamics")

        z = z + dt * z_dot
        traj.append(reducer.inverse(z))

    return torch.stack(traj, dim=1)   # (B, T+1, n)


# ------------------------------------------------------------------ #
# 2) public validation routine
# ------------------------------------------------------------------ #
@torch.no_grad()
def validate_reducer(
    sys,
    reducer,
    n_rollouts: int = 50,
    horizon: float = 5.0,
    dt: float = 0.01,
    input_mode: str = "zero",
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Simulate `n_rollouts` random initial conditions both with the full
    model and with the reduced model, then compute error metrics.

    Returns a dict with:
        mean_error     mean trajectory error
        max_error      maximum trajectory error
        relative_error relative error (mean/mean_norm)
        energy_error   energy conservation error
        latent_dim     reducer.latent_dim
        gamma          reducer.gamma
    """
    sys = sys.to(device)
    reducer = reducer.to(device)

    # sample initial states
    up, lo = sys.state_limits
    x0 = torch.rand(n_rollouts, sys.n_dims, device=device) * (up - lo) + lo

    # --- ground truth trajectory -------------------------------------- #
    # Fix: simulate returns tensor, not dict
    full_traj = sys.simulate(
        x0, 
        int(horizon / dt), 
        controller=None,  # Use None or sys.u_nominal
        controller_period=dt,
        params=sys.nominal_params,
    )  # Should be (B,T+1,n)

    # --- ROM trajectory ------------------------------------------------ #
    rom_traj = rollout_rom(
        reducer, sys, x0, horizon, dt, input_mode=input_mode
    ).to(device)  # (B,T+1,n)

    # --- metrics ------------------------------------------------------- #
    # Trajectory errors
    errors = (full_traj - rom_traj).norm(dim=-1)  # (B, T+1)
    mean_error = errors.mean()
    max_error = errors.max()
    
    # Relative error
    full_norm = full_traj.norm(dim=-1).mean()
    relative_error = mean_error / (full_norm + 1e-12)
    
    # Energy conservation
    H_full = sys.energy_function(full_traj.reshape(-1, sys.n_dims))
    H_rom = sys.energy_function(rom_traj.reshape(-1, sys.n_dims))
    energy_error = (H_full - H_rom).abs().mean()

    return {
        "mean_error": mean_error,
        "max_error": max_error,
        "relative_error": relative_error,
        "energy_error": energy_error,
        "latent_dim": reducer.latent_dim,
        "gamma": getattr(reducer, "gamma", 0.0),
    }