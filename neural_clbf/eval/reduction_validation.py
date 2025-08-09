"""
reduction_validation.py
========================

Fixed rollout with proper OpInf handling - direct replacement
Main fixes:
1. Proper control dimension handling for OpInf
2. Correct dynamics selection logic
3. Better error handling
"""

from __future__ import annotations
import torch
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

def rollout_rom(
    reducer,
    sys,
    x0: torch.Tensor,
    T: int,
    controller=None,
    dt: Optional[float] = None,
    method: str = "rk4"
) -> torch.Tensor:
    """
    Rollout reduced-order model with FIXED OpInf handling.
    
    Critical fixes:
    1. Proper control dimension for OpInf
    2. Correct dynamics selection
    3. Consistent integration methods
    """
    
    device = x0.device
    batch_size = x0.shape[0] if x0.dim() > 1 else 1
    if x0.dim() == 1: 
        x0 = x0.unsqueeze(0)
    
    if dt is None: 
        dt = sys.dt
    
    # Initialize trajectory storage
    trajectory = torch.zeros(batch_size, T + 1, sys.n_dims, device=device)
    trajectory[:, 0] = x0
    
    # Project initial condition to reduced space
    z = reducer.forward(x0)
    
    # Check what type of dynamics we have
    has_opinf_dyn = hasattr(reducer, 'dyn') and reducer.dyn is not None
    has_f_red = hasattr(reducer, 'f_red') and reducer.f_red is not None
    
    # Special case: Symplectic integrator for symplectic systems
    if has_f_red and method == "symplectic" and hasattr(reducer, 'J_symplectic'):
        logger.debug("Using symplectic integrator")
        
        def stack_qp(q, p):
            z_out = torch.zeros_like(z)
            z_out[..., ::2], z_out[..., 1::2] = q, p
            return z_out
        
        for t in range(T):
            u = controller(reducer.inverse(z)) if controller else None
            q, p = z[..., ::2], z[..., 1::2]
            
            def get_force(current_q, current_p):
                with torch.enable_grad():
                    current_z = stack_qp(current_q, current_p).detach().requires_grad_(True)
                    z_dot = reducer.f_red(current_z, u)
                return z_dot[..., 1::2]
            
            # Symplectic Euler
            p_half = p + 0.5 * dt * get_force(q, p)
            q_next = q + dt * p_half
            p_next = p_half + 0.5 * dt * get_force(q_next, p_half)
            
            z = stack_qp(q_next, p_next)
            trajectory[:, t + 1] = reducer.inverse(z)
    
    # Standard integration (RK4 or Euler)
    else:
        for t in range(T):
            # Get control input
            if controller:
                x_curr = reducer.inverse(z)
                u = controller(x_curr)
            else:
                u = None
            
            # Define dynamics function based on reducer type
            def get_z_dot(current_z):
                # CRITICAL FIX: OpInf dynamics handling
                if has_opinf_dyn:
                    # OpInf uses its learned dynamics
                    # FIX: Ensure control has correct dimension
                    if u is not None:
                        # Get expected control dimension from dynamics
                        expected_m = reducer.dyn.m
                        actual_m = u.shape[-1] if u.dim() > 0 else 1
                        
                        if actual_m != expected_m:
                            # Create properly sized control
                            u_sized = torch.zeros(current_z.shape[0], expected_m, 
                                                 device=current_z.device, dtype=current_z.dtype)
                            if actual_m < expected_m:
                                u_sized[:, :actual_m] = u
                            else:
                                u_sized = u[:, :expected_m]
                            return reducer.dyn.forward(current_z, u_sized)
                        else:
                            return reducer.dyn.forward(current_z, u)
                    else:
                        # No control
                        return reducer.dyn.forward(current_z, None)
                
                # Symplectic or other reducers with f_red
                elif has_f_red:
                    return reducer.f_red(current_z, u)
                
                # Fallback: project full dynamics (e.g., for LCR)
                else:
                    x_curr = reducer.inverse(current_z)
                    
                    # Compute full dynamics
                    if u is not None:
                        f = sys._f(x_curr, sys.nominal_params)
                        g = sys._g(x_curr, sys.nominal_params)
                        x_dot = f.squeeze(-1) + (g @ u.unsqueeze(-1)).squeeze(-1)
                    else:
                        x_dot = sys._f(x_curr, sys.nominal_params).squeeze(-1)
                    
                    # Project to reduced space
                    J = reducer.jacobian(x_curr)
                    return (J @ x_dot.unsqueeze(-1)).squeeze(-1)
            
            # Integrate using selected method
            if method == "rk4":
                # 4th order Runge-Kutta
                k1 = get_z_dot(z)
                k2 = get_z_dot(z + 0.5 * dt * k1)
                k3 = get_z_dot(z + 0.5 * dt * k2)
                k4 = get_z_dot(z + dt * k3)
                z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            else:  # Euler
                z_dot = get_z_dot(z)
                z = z + dt * z_dot
            
            # Reconstruct full state
            trajectory[:, t + 1] = reducer.inverse(z)
    
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
    input_mode: str = "zero",
) -> Dict[str, float]:
    """
    Validate reducer performance with proper error metrics.
    """
    torch.manual_seed(seed)
    device = torch.device(device)
    if dt is None: 
        dt = sys.dt
    T = int(horizon / dt)
    
    # Move to device
    sys = sys.to(device)
    reducer = reducer.to(device)
    
    # Sample initial conditions
    x0 = sys.goal_point.to(device).repeat(n_rollouts, 1)
    x0 = x0 + 0.1 * torch.randn(n_rollouts, sys.n_dims, device=device)
    
    # Define controller
    if input_mode == "zero":
        controller = lambda x: torch.zeros(x.shape[0], sys.n_controls, device=device)
    else:
        controller = sys.u_nominal

    # Simulate full system
    full_traj = sys.simulate(x0, T + 1, controller=controller, method='rk4')

    # Simulate ROM
    # Determine integration method based on reducer type
    if hasattr(reducer, 'J_symplectic'):
        rom_method = "symplectic"
    else:
        rom_method = "rk4"
    
    rom_traj = rollout_rom(
        reducer, sys, x0, T, 
        controller=controller, 
        dt=dt, 
        method=rom_method
    )
    
    # Ensure same length
    min_len = min(full_traj.shape[1], rom_traj.shape[1])
    full_traj = full_traj[:, :min_len, :]
    rom_traj = rom_traj[:, :min_len, :]
    
    # Compute errors
    errors = torch.norm(full_traj - rom_traj, p=2, dim=-1)
    final_errors = errors[:, -1]
    
    # Success metric
    success_threshold = 0.1
    success_rate = (final_errors < success_threshold).float().mean().item()

    # Energy error if available
    energy_error = 0.0
    if hasattr(sys, 'energy_function'):
        energy_full = sys.energy_function(full_traj.reshape(-1, sys.n_dims))
        energy_rom = sys.energy_function(rom_traj.reshape(-1, sys.n_dims))
        energy_full = energy_full.reshape(n_rollouts, -1)
        energy_rom = energy_rom.reshape(n_rollouts, -1)
        energy_error = (energy_full - energy_rom).abs().mean().item()

    return {
        "mean_error": errors.mean().item(),
        "max_error": errors.max().item(),
        "relative_error": errors.mean().item() / (torch.norm(full_traj, dim=-1).mean().item() + 1e-8),
        "energy_error": energy_error,
        "success_rate": success_rate,
    }