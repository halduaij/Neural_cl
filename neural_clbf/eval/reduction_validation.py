"""
reduction_validation.py
=======================

Fixed validation utilities with proper dimension handling and stability.
Replace your existing neural_clbf/eval/reduction_validation.py with this file.
"""

from __future__ import annotations
import torch
from typing import Dict, Literal, Optional, Callable
import logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def rollout_rom(
    reducer,
    sys,
    x0: torch.Tensor,
    T: int,
    dt: Optional[float] = None,
    controller: Optional[Callable] = None,
    method: str = "euler"
) -> torch.Tensor:
    """
    Fixed rollout of reduced-order model with proper dimension handling.
    
    Args:
        reducer: Dimension reduction object
        sys: Full-order system
        x0: Initial condition(s) - shape (batch, n_full) or (n_full,)
        T: Number of timesteps
        dt: Integration timestep (defaults to sys.dt)
        controller: Optional control policy
        method: Integration method ('euler' or 'rk4')
    
    Returns:
        trajectory: shape (batch, T+1, n_full) or (T+1, n_full)
    """
    device = x0.device
    
    # Handle input dimensions
    if x0.dim() == 1:
        x0 = x0.unsqueeze(0)
        single_trajectory = True
    else:
        single_trajectory = False
        
    batch_size = x0.shape[0]
    
    # Use system's dt if not specified
    if dt is None:
        dt = getattr(sys, 'dt', 0.01)
    
    # Initialize trajectory storage
    trajectory = torch.zeros(batch_size, T + 1, sys.n_dims, device=device)
    trajectory[:, 0] = x0
    
    # Project initial condition
    z = reducer.forward(x0)
    
    # Check if reducer has its own dynamics
    has_own_dynamics = hasattr(reducer, 'f_red') and reducer.f_red is not None
    has_learned_dynamics = hasattr(reducer, 'dyn') and reducer.dyn is not None
    
    if has_own_dynamics:
        logger.debug("Using reducer's intrinsic dynamics (f_red)")
    elif has_learned_dynamics:
        logger.debug("Using learned OpInf dynamics")
    else:
        logger.debug("Using projection of full-order dynamics")
    
    # Integration loop
    for t in range(T):
        # Get control input
        if controller is not None:
            x_curr = reducer.inverse(z)
            u = controller(x_curr)
            # Ensure proper shape
            if u.dim() == 1:
                u = u.unsqueeze(0)
            if u.shape[0] == 1 and batch_size > 1:
                u = u.expand(batch_size, -1)
        else:
            u = None
        
        # Compute dynamics
        if has_own_dynamics:
            # Use reducer's own dynamics
            z_dot = reducer.f_red(z, u)
        elif has_learned_dynamics:
            # Use learned OpInf dynamics
            if u is not None and reducer.n_controls > 0:
                # Ensure u has correct dimensions
                if u.shape[1] != reducer.n_controls:
                    u_correct = torch.zeros(batch_size, reducer.n_controls, device=device)
                    n_copy = min(u.shape[1], reducer.n_controls)
                    u_correct[:, :n_copy] = u[:, :n_copy]
                    u = u_correct
            z_dot = reducer.dyn.forward(z, u)
        else:
            # Project full-order dynamics
            x_curr = reducer.inverse(z)
            
            if u is not None:
                f = sys._f(x_curr, sys.nominal_params)
                g = sys._g(x_curr, sys.nominal_params)
                # Handle dimension mismatches
                if u.shape[1] != g.shape[2]:
                    u_correct = torch.zeros(batch_size, g.shape[2], device=device)
                    n_copy = min(u.shape[1], g.shape[2])
                    u_correct[:, :n_copy] = u[:, :n_copy]
                    u = u_correct
                x_dot = f.squeeze(-1) + (g @ u.unsqueeze(-1)).squeeze(-1)
            else:
                x_dot = sys._f(x_curr, sys.nominal_params).squeeze(-1)
            
            # Project to reduced space
            J = reducer.jacobian(x_curr)
            if J.dim() == 2:
                J = J.unsqueeze(0)
            if J.shape[0] == 1 and batch_size > 1:
                J = J.expand(batch_size, -1, -1)
            z_dot = torch.bmm(J, x_dot.unsqueeze(-1)).squeeze(-1)
        
        # Ensure z_dot has correct shape
        if z_dot.dim() == 1 and z.dim() == 2:
            z_dot = z_dot.unsqueeze(0)
        elif z_dot.dim() == 2 and z.dim() == 1:
            z_dot = z_dot.squeeze(0)
        
        # Integration step
        if method == "euler":
            z = z + dt * z_dot
        elif method == "rk4":
            # RK4 implementation
            k1 = z_dot
            
            # k2
            z_mid = z + 0.5 * dt * k1
            if has_own_dynamics:
                k2 = reducer.f_red(z_mid, u)
            elif has_learned_dynamics:
                k2 = reducer.dyn.forward(z_mid, u)
            else:
                x_mid = reducer.inverse(z_mid)
                x_dot_mid = sys._f(x_mid, sys.nominal_params).squeeze(-1)
                J_mid = reducer.jacobian(x_mid)
                if J_mid.dim() == 2:
                    J_mid = J_mid.unsqueeze(0).expand(batch_size, -1, -1)
                k2 = torch.bmm(J_mid, x_dot_mid.unsqueeze(-1)).squeeze(-1)
            
            # k3
            z_mid = z + 0.5 * dt * k2
            if has_own_dynamics:
                k3 = reducer.f_red(z_mid, u)
            elif has_learned_dynamics:
                k3 = reducer.dyn.forward(z_mid, u)
            else:
                x_mid = reducer.inverse(z_mid)
                x_dot_mid = sys._f(x_mid, sys.nominal_params).squeeze(-1)
                J_mid = reducer.jacobian(x_mid)
                if J_mid.dim() == 2:
                    J_mid = J_mid.unsqueeze(0).expand(batch_size, -1, -1)
                k3 = torch.bmm(J_mid, x_dot_mid.unsqueeze(-1)).squeeze(-1)
            
            # k4
            z_end = z + dt * k3
            if has_own_dynamics:
                k4 = reducer.f_red(z_end, u)
            elif has_learned_dynamics:
                k4 = reducer.dyn.forward(z_end, u)
            else:
                x_end = reducer.inverse(z_end)
                x_dot_end = sys._f(x_end, sys.nominal_params).squeeze(-1)
                J_end = reducer.jacobian(x_end)
                if J_end.dim() == 2:
                    J_end = J_end.unsqueeze(0).expand(batch_size, -1, -1)
                k4 = torch.bmm(J_end, x_dot_end.unsqueeze(-1)).squeeze(-1)
            
            # Ensure all k's have same shape
            for k in [k1, k2, k3, k4]:
                if k.dim() == 1 and z.dim() == 2:
                    k = k.unsqueeze(0)
            
            z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # Store reconstructed state
        x_next = reducer.inverse(z)
        trajectory[:, t + 1] = x_next
    
    # Return with correct shape
    if single_trajectory:
        return trajectory.squeeze(0)
    else:
        return trajectory


@torch.no_grad()
def validate_reducer(
    sys,
    reducer,
    n_rollouts: int = 50,
    horizon: float = 2.0,
    dt: Optional[float] = None,
    device: str = "cpu",
    seed: int = 42,
    input_mode: str = "zero"
) -> Dict[str, torch.Tensor]:
    """
    Fixed validation with proper dimension handling.
    
    Args:
        sys: System object
        reducer: Reducer to validate
        n_rollouts: Number of test trajectories
        horizon: Simulation time in seconds
        dt: Integration timestep
        device: Computation device
        seed: Random seed
        input_mode: Control mode ('zero', 'random', 'nominal')
    
    Returns:
        Dictionary of validation metrics
    """
    torch.manual_seed(seed)
    device = torch.device(device)
    
    # Use system's dt if not specified
    if dt is None:
        dt = getattr(sys, 'dt', 0.01)
    
    T = int(horizon / dt)
    
    # Move system and reducer to device
    sys = sys.to(device)
    reducer = reducer.to(device)
    
    # Sample initial states near equilibrium
    if hasattr(sys, 'goal_point'):
        x_eq = sys.goal_point.to(device).squeeze()
        if x_eq.dim() > 1: 
            x_eq = x_eq[0]
        
        # Add small perturbations
        perturbation = 0.4 * torch.randn(n_rollouts, sys.n_dims, device=device)
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
    
    # Define controller based on input mode
    if input_mode == "zero":
        controller = None
    elif input_mode == "random":
        def controller(x):
            return 0.01 * torch.randn(x.shape[0], sys.n_controls, device=x.device)
    elif input_mode == "nominal":
        if hasattr(sys, 'u_nominal'):
            controller = lambda x: sys.u_nominal(x, sys.nominal_params)
        else:
            controller = None
    else:
        controller = None
    
    # Simulate full system
    full_traj = torch.zeros(n_rollouts, T + 1, sys.n_dims, device=device)
    full_traj[:, 0] = x0
    
    for i in range(n_rollouts):
        x = x0[i:i+1]
        for t in range(T):
            # Get control
            if controller is not None:
                u = controller(x)
            else:
                u = torch.zeros(1, sys.n_controls, device=device)
            
            # Full dynamics
            f = sys._f(x, sys.nominal_params)
            g = sys._g(x, sys.nominal_params)
            x_dot = f.squeeze(-1) + (g @ u.unsqueeze(-1)).squeeze(-1)
            
            x = x + dt * x_dot
            full_traj[i, t + 1] = x.squeeze()
    
    # Simulate ROM
    rom_traj = rollout_rom(reducer, sys, x0, T, dt=dt, controller=controller, method='rk4')
    
    # Ensure same shape
    min_T = min(full_traj.shape[1], rom_traj.shape[1])
    full_traj = full_traj[:, :min_T, :]
    rom_traj = rom_traj[:, :min_T, :]
    
    # Compute metrics
    errors = (full_traj - rom_traj).norm(dim=-1)  # (B, T)
    
    # Ensure errors is 2D
    if errors.dim() == 1:
        errors = errors.unsqueeze(0)
    
    # Mean and max errors
    mean_error = errors.mean()
    max_error = errors.max()
    
    # Relative error
    full_norm = full_traj.norm(dim=-1)
    rel_errors = errors / (full_norm + 1e-8)
    relative_error = rel_errors.mean()
    
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
    
    # Success rate calculation with dimension safety
    if errors.shape[0] > 0 and errors.shape[1] > 1:
        final_errors = errors[:, -1]  # (B,)
        
        # Initial error for threshold
        T_initial = min(10, errors.shape[1])
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
        "relative_error": relative_error,
        "energy_error": energy_error,
        "success_rate": success_rate,
        "errors": errors,
    }