"""
Reduction validation utilities with stability fixes
==================================================

Fixed version addressing tensor dimension mismatches and instability issues.
"""

from __future__ import annotations
import torch
import logging
from typing import Dict, Literal, Optional, Callable

# FIX: Add logger definition
logger = logging.getLogger(__name__)


@torch.no_grad()
def rollout_rom(
    sys,
    reducer,
    x0: torch.Tensor,
    T: int,
    controller=None,
    dt: Optional[float] = None,
    method: str = "euler"
) -> torch.Tensor:
    """
    Rollout reduced-order model using reducer's own dynamics when available.
    
    Args:
        sys: Full-order system
        reducer: Dimension reduction object
        x0: Initial condition(s) - shape (batch, n_full)
        T: Number of timesteps
        controller: Control policy (optional)
        dt: Integration timestep (defaults to sys.dt)
        method: Integration method ('euler' or 'rk4')
    
    Returns:
        trajectory: shape (batch, T+1, n_full)
    """
    device = x0.device
    batch_size = x0.shape[0] if x0.dim() > 1 else 1
    
    # Use system's dt if not specified
    if dt is None:
        dt = getattr(sys, 'dt', 0.01)
    
    # Initialize trajectory storage
    trajectory = torch.zeros(batch_size, T + 1, sys.n_dims, device=device)
    trajectory[:, 0] = x0.squeeze() if x0.dim() > 1 else x0
    
    # Project initial condition
    z = reducer.forward(x0)
    
    # Check if reducer has its own dynamics
    has_own_dynamics = hasattr(reducer, 'f_red') and reducer.f_red is not None
    
    if has_own_dynamics:
        logger.debug("Using reducer's intrinsic dynamics (f_red)")
    else:
        logger.debug("Using projection of full-order dynamics")
    
    # Integration loop
    for t in range(T):
        # Get control input
        if controller is not None:
            x_curr = reducer.inverse(z)
            u = controller(x_curr)
        else:
            u = None
        
        # Compute dynamics
        if has_own_dynamics:
            # Use reducer's own dynamics
            z_dot = reducer.f_red(z, u)
        else:
            # Project full dynamics (fallback)
            x_curr = reducer.inverse(z)
            
            if hasattr(reducer, 'dyn') and reducer.dyn is not None:
                # Use learned OpInf dynamics
                z_dot = reducer.dyn.forward(z, u)
            else:
                # Project full-order dynamics
                if u is not None:
                    f = sys._f(x_curr, sys.nominal_params)
                    g = sys._g(x_curr, sys.nominal_params)
                    x_dot = f.squeeze(-1) + (g @ u.unsqueeze(-1)).squeeze(-1)
                else:
                    x_dot = sys._f(x_curr, sys.nominal_params).squeeze(-1)
                
                # Project to reduced space
                J = reducer.jacobian(x_curr)
                z_dot = torch.bmm(J, x_dot.unsqueeze(-1)).squeeze(-1)
        
        # Ensure z_dot has correct shape
        if z_dot.dim() == 1 and z.dim() == 2:
            z_dot = z_dot.unsqueeze(0)
        
        # Integration step
        if method == "euler":
            z = z + dt * z_dot
        elif method == "rk4":
            k1 = z_dot
            
            # k2
            z_mid = z + 0.5 * dt * k1
            if has_own_dynamics:
                k2 = reducer.f_red(z_mid, u)
            else:
                x_mid = reducer.inverse(z_mid)
                x_dot_mid = sys._f(x_mid, sys.nominal_params).squeeze(-1)
                J_mid = reducer.jacobian(x_mid)
                k2 = torch.bmm(J_mid, x_dot_mid.unsqueeze(-1)).squeeze(-1)
            
            # k3
            z_mid = z + 0.5 * dt * k2
            if has_own_dynamics:
                k3 = reducer.f_red(z_mid, u)
            else:
                x_mid = reducer.inverse(z_mid)
                x_dot_mid = sys._f(x_mid, sys.nominal_params).squeeze(-1)
                J_mid = reducer.jacobian(x_mid)
                k3 = torch.bmm(J_mid, x_dot_mid.unsqueeze(-1)).squeeze(-1)
            
            # k4
            z_end = z + dt * k3
            if has_own_dynamics:
                k4 = reducer.f_red(z_end, u)
            else:
                x_end = reducer.inverse(z_end)
                x_dot_end = sys._f(x_end, sys.nominal_params).squeeze(-1)
                J_end = reducer.jacobian(x_end)
                k4 = torch.bmm(J_end, x_dot_end.unsqueeze(-1)).squeeze(-1)
            
            z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # Store reconstructed state
        x_next = reducer.inverse(z)
        trajectory[:, t + 1] = x_next.squeeze() if x_next.dim() > 1 else x_next
    
    return trajectory


@torch.no_grad()
def validate_reducer(
    sys,
    reducer,
    n_rollouts: int = 50,
    t_sim: float = 2.0,
    dt: Optional[float] = None,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Fixed validation with proper dimension handling.
    """
    torch.manual_seed(seed)
    device = torch.device(device)
    
    # Use system's dt if not specified
    if dt is None:
        dt = getattr(sys, 'dt', 0.01)
    
    T = int(t_sim / dt)
    
    # Move system and reducer to device
    sys = sys.to(device)
    reducer = reducer.to(device)
    
    # Sample initial states near equilibrium
    if hasattr(sys, 'goal_point'):
        x_eq = sys.goal_point.to(device).squeeze()
        if x_eq.dim() > 1: 
            x_eq = x_eq[0]
        
        # Add small perturbations
        perturbation = 0.1 * torch.randn(n_rollouts, sys.n_dims, device=device)
        x0 = x_eq + perturbation
        
        # Ensure within bounds
        if hasattr(sys, 'state_limits'):
            upper, lower = sys.state_limits
            
            # Ensure limits are tensors on device
            if not torch.is_tensor(upper):
                upper = torch.tensor(upper, device=device, dtype=x0.dtype)
            if not torch.is_tensor(lower):
                lower = torch.tensor(lower, device=device, dtype=x0.dtype)
            
            upper = upper.to(device).squeeze()
            lower = lower.to(device).squeeze()
            
            # Check dimensions match
            if upper.shape == x0.shape[1:] and lower.shape == x0.shape[1:]:
                # Robust clamping
                if torch.all(lower < upper):
                    x0 = torch.clamp(x0, min=lower, max=upper)
    else:
        # Random initial conditions
        x0 = torch.randn(n_rollouts, sys.n_dims, device=device) * 0.1
    
    # Simulate full system
    full_traj = torch.zeros(n_rollouts, T + 1, sys.n_dims, device=device)
    full_traj[:, 0] = x0
    
    for i in range(n_rollouts):
        x = x0[i:i+1]
        for t in range(T):
            x_dot = sys._f(x, sys.nominal_params).squeeze(-1)
            x = x + dt * x_dot
            full_traj[i, t + 1] = x.squeeze()
    
    # Simulate ROM
    rom_traj = rollout_rom(sys, reducer, x0, T, dt=dt)
    
    # Ensure same shape
    min_T = min(full_traj.shape[1], rom_traj.shape[1])
    full_traj = full_traj[:, :min_T, :]
    rom_traj = rom_traj[:, :min_T, :]
    
    # Compute metrics
    errors = (full_traj - rom_traj).norm(dim=-1)  # (B, T)
    
    # FIX: Ensure errors is 2D for all subsequent operations
    if errors.dim() == 1:
        errors = errors.unsqueeze(0)
    
    # Mean and max errors
    mean_error = errors.mean()
    max_error = errors.max()
    
    # Energy conservation
    if hasattr(sys, 'energy_function'):
        # Reshape for energy function if needed
        full_traj_reshaped = full_traj.reshape(-1, sys.n_dims)
        rom_traj_reshaped = rom_traj.reshape(-1, sys.n_dims)
        
        energy_full = sys.energy_function(full_traj_reshaped).reshape(n_rollouts, -1)
        energy_rom = sys.energy_function(rom_traj_reshaped).reshape(n_rollouts, -1)
        
        # Ensure 1D output
        if energy_full.dim() > 2:
            energy_full = energy_full.squeeze()
        if energy_rom.dim() > 2:
            energy_rom = energy_rom.squeeze()
        
        # Relative energy error
        E0 = energy_full[:, 0].abs().max().item()
        eps_E = max(1e-8, 1e-6 * E0)  # 1 ppm or 1e-8
        
        energy_diff = (energy_full - energy_rom).abs()
        energy_error = energy_diff.mean()
    else:
        energy_error = torch.tensor(0.0, device=device)
        eps_E = 1e-8
    
    # Success rate calculation with dimension safety - FIX: handle 1D case
    if errors.shape[0] > 0 and errors.shape[1] > 1:
        final_errors = errors[:, -1]  # (B,)
        
        # Initial error for threshold - FIX: handle short trajectories
        T_initial = min(10, errors.shape[1])
        if errors.dim() == 1:
            initial_errors = errors[:T_initial].mean().unsqueeze(0)
        else:
            initial_errors = errors[:, :T_initial].mean(dim=1)  # (B,)
        
        # Success threshold
        avg_initial_error = initial_errors.mean().item()
        success_threshold = max(10.0 * avg_initial_error, 1e-3)
        
        # Success mask
        success_mask = (final_errors < success_threshold) & torch.isfinite(final_errors)
        
        # Energy success
        if hasattr(sys, 'energy_function'):
            final_energy_error = energy_diff[:, -1] if energy_diff.dim() > 1 else energy_diff
            energy_success = final_energy_error < eps_E
            success_mask = success_mask & energy_success
        
        success_rate = success_mask.float().mean()
    else:
        success_rate = torch.tensor(0.0, device=device)
    
    return {
        "mean_error": mean_error,
        "max_error": max_error,
        "energy_error": energy_error,
        "success_rate": success_rate,
        "errors": errors,
    }