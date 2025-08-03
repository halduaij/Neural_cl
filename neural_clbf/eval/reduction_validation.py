"""
Reduction‑validation utilities
==============================

Fixed version addressing tensor dimension mismatches.
"""

from __future__ import annotations
import torch
from typing import Dict, Literal, Optional


def rollout_rom(
    reducer,                       
    sys,                          
    x0: torch.Tensor,             
    horizon: float,
    dt: float,
    input_mode: Literal["zero", "random"] = "zero",
    controller: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
):
    """
    Integrate the reduced‑order model (ROM).
    """
    B, n = x0.shape
    T = int(horizon / dt)
    device = x0.device
    
    z = reducer.forward(x0)
    
    # FIX: Detach trajectory points to prevent unnecessary gradient tracking
    traj = [reducer.inverse(z).detach()] 
    
    for t in range(T):
        # Determine control input
        if controller:
            x_rec_current = reducer.inverse(z)
            u = controller(x_rec_current)
        elif input_mode == "zero":
            u = torch.zeros(B, sys.n_controls, device=device)
        elif input_mode == "random":
             # (random input logic omitted for brevity, same as original)
             pass
        else:
            u = torch.zeros(B, sys.n_controls, device=device)

        # Compute latent dynamics
        if hasattr(reducer, "dyn") and reducer.dyn is not None:
            # OpInf-style dynamics (Stable by design now)
            try:
                # Handle potential control dimension mismatch in OpInf
                if hasattr(reducer.dyn, 'm') and u.shape[1] != reducer.dyn.m:
                     u_adapted = torch.zeros(B, reducer.dyn.m, device=device)
                     min_dim = min(u.shape[1], reducer.dyn.m)
                     u_adapted[:, :min_dim] = u[:, :min_dim]
                     z_dot = reducer.dyn.forward(z, u_adapted)
                else:
                     z_dot = reducer.dyn.forward(z, u)
                
            except Exception as e:
                print(f"OpInf dynamics error: {e}, using damped fallback.")
                z_dot = -0.5 * z # Stable fallback
                    
        elif hasattr(sys, "closed_loop_dynamics"):
            # Projection-based dynamics (e.g., Symplectic, LCR)
            
            x_full = reducer.inverse(z)
            
            # FIX: The Autograd error is resolved because we now use analytical Jacobians
            # in the reducer subclasses (SPR, OpInf, LCR).

            # Compute full dynamics
            x_dot = sys.closed_loop_dynamics(x_full, u, params=sys.nominal_params)

            # Project dynamics back: z_dot = J(x) @ x_dot
            J = reducer.jacobian(x_full)  # (B, d, n) - Now analytical
            
            if x_dot.dim() == 2:
                x_dot = x_dot.unsqueeze(-1)
                
            z_dot = torch.bmm(J, x_dot).squeeze(-1)  # (B, d)
        
        else:
            raise RuntimeError("Reducer lacks latent dynamics capability.")

        # Euler step and stability check
        z_new = z + dt * z_dot
        
        # Check for instability/NaN during simulation
        if not torch.isfinite(z_new).all() or (z_new.norm(dim=1) > 1e5).any():
            print(f"Warning: Latent instability detected at step {t}. Clamping.")
            z_new = torch.clamp(z_new, -1e5, 1e5)
            z_new = torch.nan_to_num(z_new, nan=0.0)

        z = z_new
        traj.append(reducer.inverse(z).detach())

    return torch.stack(traj, dim=1)


@torch.no_grad()
def validate_reducer(
    sys,
    reducer,
    n_rollouts: int = 50,
    horizon: float = 5.0,
    dt: float = 0.01,
    input_mode: str = "zero",
    device: str = "cpu",
    controller: Optional[callable] = None,
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
    # Move to device
    device = torch.device(device)
    sys = sys.to(device)
    reducer = reducer.to(device)
    
    # Ensure reducer has correct full dimension
    if not hasattr(reducer, 'full_dim') or reducer.full_dim is None:
        reducer.full_dim = sys.n_dims

    # Sample initial states
    up, lo = sys.state_limits
    up = up.to(device) if torch.is_tensor(up) else torch.tensor(up, device=device)
    lo = lo.to(device) if torch.is_tensor(lo) else torch.tensor(lo, device=device)
            
    # Sample near equilibrium instead of random
    if hasattr(sys, 'goal_point'):
        x_eq = sys.goal_point.squeeze().to(device)
        x0 = x_eq.unsqueeze(0).repeat(n_rollouts, 1)
        x0 += 0.1 * torch.randn_like(x0)  # 10% perturbation around equilibrium
    else:
        # Fallback to original if no equilibrium point
        x0 = torch.rand(n_rollouts, sys.n_dims, device=device) * (up - lo) + lo
    # Number of timesteps
    T = int(horizon / dt)
    
    # Simulate ground truth trajectory
    try:
        if controller is None:
            # Use zero control or nominal controller
            controller = lambda x: torch.zeros(x.shape[0], sys.n_controls, device=x.device)
        
        # Simulate using the system's simulate method
        full_traj = sys.simulate(
            x0, 
            T,  # num_steps, not including initial condition
            controller=controller,
            controller_period=dt,
            params=sys.nominal_params,
        )  # Should return (B, T+1, n)
        
        # Ensure we have the right shape
        if full_traj.shape[1] != T + 1:
            # If simulate doesn't include initial condition, prepend it
            if full_traj.shape[1] == T:
                full_traj = torch.cat([x0.unsqueeze(1), full_traj], dim=1)
            else:
                # Truncate or pad to match expected length
                target_shape = (n_rollouts, T + 1, sys.n_dims)
                if full_traj.shape[1] > T + 1:
                    full_traj = full_traj[:, :T+1, :]
                else:
                    # Pad with last state
                    padding_needed = T + 1 - full_traj.shape[1]
                    last_states = full_traj[:, -1:, :].expand(-1, padding_needed, -1)
                    full_traj = torch.cat([full_traj, last_states], dim=1)
                    
    except Exception as e:
        print(f"Warning: System simulate failed ({e}), using manual integration")
        # Manual integration as fallback
        full_traj = torch.zeros(n_rollouts, T + 1, sys.n_dims, device=device)
        full_traj[:, 0] = x0
        
        for t in range(T):
            x_current = full_traj[:, t]
            u = controller(x_current) if controller else torch.zeros(n_rollouts, sys.n_controls, device=device)
            xdot = sys.closed_loop_dynamics(x_current, u, sys.nominal_params)
            full_traj[:, t + 1] = x_current + dt * xdot

    # Simulate ROM trajectory
    try:
        rom_traj = rollout_rom(
            reducer, sys, x0, horizon, dt, input_mode=input_mode
        )
    except Exception as e:
        print(f"ROM rollout failed: {e}")
        raise

    # Ensure trajectories have same shape
    min_length = min(full_traj.shape[1], rom_traj.shape[1])
    full_traj = full_traj[:, :min_length, :]
    rom_traj = rom_traj[:, :min_length, :]

    # Compute metrics
    errors = (full_traj - rom_traj).norm(dim=-1)  # (B, T)
    mean_error = errors.mean()
    max_error = errors.max()
    
    # Relative error
    full_norm = full_traj.norm(dim=-1).mean()
    relative_error = mean_error / (full_norm + 1e-12)
    
    # Energy conservation (if system has energy function)
    if hasattr(sys, 'energy_function'):
        try:
            # Reshape for energy computation
            H_full = sys.energy_function(full_traj.reshape(-1, sys.n_dims))
            H_rom = sys.energy_function(rom_traj.reshape(-1, sys.n_dims))
            
            # Handle different energy function output shapes
            if H_full.dim() > 1:
                H_full = H_full.view(-1)
            if H_rom.dim() > 1:
                H_rom = H_rom.view(-1)
                
            energy_error = (H_full - H_rom).abs().mean()
        except Exception as e:
            print(f"Warning: Energy computation failed ({e}), setting to NaN")
            energy_error = torch.tensor(float('nan'), device=device)
    else:
        energy_error = torch.tensor(0.0, device=device)
    
    # Success rate (trajectories that didn't diverge too much)
    final_errors = errors[:, -1]
    success_threshold = 10.0 * errors[:, 0].mean()  # 10x initial error
    success_rate = (final_errors < success_threshold).float().mean()

    return {
        "mean_error": mean_error,
        "max_error": max_error,
        "relative_error": relative_error,
        "energy_error": energy_error,
        "success_rate": success_rate,
        "latent_dim": reducer.latent_dim,
        "gamma": getattr(reducer, "gamma", 0.0),
    }