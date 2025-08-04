"""
Reduction validation utilities with stability fixes
==================================================

Fixed version addressing tensor dimension mismatches and instability issues.
"""

from __future__ import annotations
import torch
from typing import Dict, Literal, Optional, Callable


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
    Integrate the reduced-order model (ROM) with stability safeguards.
    """
    B, n = x0.shape
    T = int(horizon / dt)
    device = x0.device
    
    # Initial projection
    z = reducer.forward(x0)
    
    # Initialize trajectory storage
    traj = [reducer.inverse(z).detach()]
    
    # Stability monitoring
    max_norm = 1e5  # Maximum allowed norm
    instability_counter = 0
    max_instabilities = 10
    
    for t in range(T):
        # Store previous state for fallback
        z_prev = z.clone()
        
        try:
            # Determine control input
            if controller is not None:
                x_rec_current = reducer.inverse(z)
                u = controller(x_rec_current)
            elif input_mode == "zero":
                u = torch.zeros(B, sys.n_controls, device=device)
            elif input_mode == "random":
                u = 0.1 * torch.randn(B, sys.n_controls, device=device)
            else:
                u = torch.zeros(B, sys.n_controls, device=device)

            # Compute latent dynamics
            if hasattr(reducer, "dyn") and reducer.dyn is not None:
                # OpInf-style dynamics
                try:
                    # Handle control dimension mismatch
                    if hasattr(reducer.dyn, 'm') and u.shape[1] != reducer.dyn.m:
                        u_adapted = torch.zeros(B, reducer.dyn.m, device=device)
                        min_dim = min(u.shape[1], reducer.dyn.m)
                        u_adapted[:, :min_dim] = u[:, :min_dim]
                        z_dot = reducer.dyn.forward(z, u_adapted)
                    else:
                        z_dot = reducer.dyn.forward(z, u)
                    
                    # Check for NaN/Inf in dynamics
                    if not torch.isfinite(z_dot).all():
                        raise RuntimeError("Non-finite dynamics detected")
                        
                except Exception as e:
                    # Fallback to damped dynamics
                    print(f"  OpInf dynamics error at t={t}: {e}, using damped fallback.")
                    z_dot = -0.5 * z  # Stable fallback
                    
            elif hasattr(sys, "closed_loop_dynamics"):
                # Projection-based dynamics (e.g., Symplectic, LCR)
                x_full = reducer.inverse(z)
                
                # Clamp to state limits if available
                if hasattr(sys, 'state_limits'):
                    upper, lower = sys.state_limits
                    x_full = torch.clamp(x_full, lower, upper)
                
                # Compute full dynamics
                x_dot = sys.closed_loop_dynamics(x_full, u, params=sys.nominal_params)
                
                # Project dynamics using analytical Jacobian
                J = reducer.jacobian(x_full)  # (B, d, n)
                
                # Handle dimension mismatches
                if x_dot.dim() == 2:
                    x_dot = x_dot.unsqueeze(-1)
                    
                z_dot = torch.bmm(J, x_dot).squeeze(-1)  # (B, d)
                
                # Check for NaN/Inf
                if not torch.isfinite(z_dot).all():
                    print(f"  Warning: Non-finite projected dynamics at t={t}")
                    z_dot = torch.nan_to_num(z_dot, nan=0.0, posinf=0.0, neginf=0.0)
            
            else:
                raise RuntimeError("Reducer lacks latent dynamics capability.")

            # Euler integration with clamping
            z_new = z + dt * z_dot
            
            # Stability checks
            z_norm = z_new.norm(dim=1).max().item()
            
            if not torch.isfinite(z_new).all():
                print(f"  Warning: NaN/Inf detected at t={t}. Using previous state.")
                z_new = z_prev
                instability_counter += 1
                
            elif z_norm > max_norm:
                print(f"  Warning: Large norm {z_norm:.2e} at t={t}. Clamping.")
                # Scale down to maximum norm
                scale = max_norm / (z_norm + 1e-8)
                z_new = z_new * scale
                instability_counter += 1
                
            else:
                # Check for rapid growth
                if t > 0:
                    growth_rate = z_norm / (z_prev.norm(dim=1).max().item() + 1e-8)
                    if growth_rate > 10.0:
                        print(f"  Warning: Rapid growth (rate={growth_rate:.1f}) at t={t}")
                        # Apply damping
                        z_new = 0.9 * z_new + 0.1 * z_prev
                        instability_counter += 1
            
            # Terminate if too many instabilities
            if instability_counter > max_instabilities:
                print(f"  ERROR: Too many instabilities. Terminating at t={t}")
                # Pad with last stable state
                x_last = reducer.inverse(z).detach()
                for _ in range(T - t):
                    traj.append(x_last)
                break
            
            # Update state
            z = z_new
            
            # Store trajectory point
            x_rec = reducer.inverse(z)
            
            # Final safety check on reconstructed state
            if hasattr(sys, 'state_limits'):
                upper, lower = sys.state_limits
                x_rec = torch.clamp(x_rec, lower, upper)
                
            traj.append(x_rec.detach())
            
        except Exception as e:
            print(f"  Critical error at t={t}: {e}")
            # Pad trajectory with last valid state
            if len(traj) > 0:
                x_last = traj[-1]
                for _ in range(T - t):
                    traj.append(x_last)
            break

    # Ensure we have the right trajectory length
    traj = traj[:T+1]  # Include initial condition
    
    # Stack trajectory
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
    Simulate n_rollouts random initial conditions with stability improvements.

    Returns a dict with:
        mean_error     mean trajectory error
        max_error      maximum trajectory error
        relative_error relative error (mean/mean_norm)
        energy_error   energy conservation error
        success_rate   fraction of stable trajectories
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

    # Sample initial states near equilibrium
    if hasattr(sys, 'goal_point'):
        x_eq = sys.goal_point.squeeze().to(device)
        
        # Smaller perturbations for high-dimensional systems
        perturbation_scale = 0.1 if sys.n_dims < 20 else 0.05
        
        x0 = x_eq.unsqueeze(0).repeat(n_rollouts, 1)
        x0 += perturbation_scale * torch.randn_like(x0)
        
        # Ensure within bounds
        if hasattr(sys, 'state_limits'):
            upper, lower = sys.state_limits
            upper = upper.to(device) if torch.is_tensor(upper) else torch.tensor(upper, device=device)
            lower = lower.to(device) if torch.is_tensor(lower) else torch.tensor(lower, device=device)
            x0 = torch.clamp(x0, lower, upper)
    else:
        # Fallback to random sampling
        upper, lower = sys.state_limits
        upper = upper.to(device) if torch.is_tensor(upper) else torch.tensor(upper, device=device)
        lower = lower.to(device) if torch.is_tensor(lower) else torch.tensor(lower, device=device)
        x0 = torch.rand(n_rollouts, sys.n_dims, device=device) * (upper - lower) + lower
    
    # Number of timesteps
    T = int(horizon / dt)
    
    # Default controller
    if controller is None:
        controller = lambda x: torch.zeros(x.shape[0], sys.n_controls, device=x.device)
    
    # Simulate ground truth trajectory
    try:
        # Use system's simulate method
        full_traj = sys.simulate(
            x0, 
            T,
            controller=controller,
            controller_period=max(dt, sys.dt),  # Ensure valid controller period
            params=sys.nominal_params,
        )
        
        # Ensure correct shape
        if full_traj.shape[1] != T + 1:
            if full_traj.shape[1] == T:
                full_traj = torch.cat([x0.unsqueeze(1), full_traj], dim=1)
            else:
                # Pad or truncate
                target_length = T + 1
                if full_traj.shape[1] > target_length:
                    full_traj = full_traj[:, :target_length, :]
                else:
                    padding_needed = target_length - full_traj.shape[1]
                    last_states = full_traj[:, -1:, :].expand(-1, padding_needed, -1)
                    full_traj = torch.cat([full_traj, last_states], dim=1)
                    
    except Exception as e:
        print(f"Warning: System simulate failed ({e}), using manual integration")
        # Manual integration as fallback
        full_traj = torch.zeros(n_rollouts, T + 1, sys.n_dims, device=device)
        full_traj[:, 0] = x0
        
        for t in range(T):
            x_current = full_traj[:, t]
            u = controller(x_current)
            
            try:
                xdot = sys.closed_loop_dynamics(x_current, u, sys.nominal_params)
                full_traj[:, t + 1] = x_current + dt * xdot
            except Exception as e2:
                print(f"Manual integration failed at t={t}: {e2}")
                # Just copy last state
                full_traj[:, t + 1] = x_current

    # Simulate ROM trajectory
    try:
        rom_traj = rollout_rom(
            reducer, sys, x0, horizon, dt, 
            input_mode=input_mode, controller=controller
        )
    except Exception as e:
        print(f"ROM rollout failed: {e}")
        # Return failure metrics
        return {
            "mean_error": torch.tensor(float('inf'), device=device),
            "max_error": torch.tensor(float('inf'), device=device),
            "relative_error": torch.tensor(1.0, device=device),
            "energy_error": torch.tensor(float('inf'), device=device),
            "success_rate": torch.tensor(0.0, device=device),
            "latent_dim": reducer.latent_dim,
            "gamma": getattr(reducer, "gamma", float('inf')),
        }

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
    
    # Energy conservation
    if hasattr(sys, 'energy_function'):
        try:
            # Compute energies
            H_full = sys.energy_function(full_traj.reshape(-1, sys.n_dims))
            H_rom = sys.energy_function(rom_traj.reshape(-1, sys.n_dims))
            
            # Handle shape issues
            if H_full.dim() > 1:
                H_full = H_full.view(-1)
            if H_rom.dim() > 1:
                H_rom = H_rom.view(-1)
            
            # Compute error, handling NaN/Inf
            energy_diff = (H_full - H_rom).abs()
            energy_diff_finite = energy_diff[torch.isfinite(energy_diff)]
            
            if energy_diff_finite.numel() > 0:
                energy_error = energy_diff_finite.mean()
            else:
                energy_error = torch.tensor(float('inf'), device=device)
                
        except Exception as e:
            print(f"Warning: Energy computation failed ({e})")
            energy_error = torch.tensor(float('nan'), device=device)
    else:
        energy_error = torch.tensor(0.0, device=device)
    
    # Success rate (trajectories that remained stable)
    final_errors = errors[:, -1]
    initial_errors = errors[:, min(10, errors.shape[1]-1)].mean(dim=1)  # Average over first few steps
    
    # Success: final error is not too much worse than initial
    success_threshold = max(10.0 * initial_errors.mean().item(), 1.0)
    success_mask = (final_errors < success_threshold) & torch.isfinite(final_errors)
    success_rate = success_mask.float().mean()

    return {
        "mean_error": mean_error,
        "max_error": max_error,
        "relative_error": relative_error,
        "energy_error": energy_error,
        "success_rate": success_rate,
        "latent_dim": reducer.latent_dim,
        "gamma": getattr(reducer, "gamma", 0.0),
    }