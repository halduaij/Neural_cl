"""
Complete Swing Equation Dimension Reduction Benchmark with Stability Metrics
=============================================================================

This script provides a full pipeline for benchmarking dimension reduction with
proper stability classification and validation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass
import os
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules
from neural_clbf.systems import SwingEquationSystem
from neural_clbf.dimension_reduction.symplectic_projection import SymplecticProjectionReducer as SPR
from neural_clbf.dimension_reduction.opinf import OpInfReducer
from neural_clbf.dimension_reduction.lyap_coherency import LyapCoherencyReducer as LCR
from neural_clbf.eval.reduction_validation import rollout_rom
# Note: Not importing select_reducer from manager to avoid SPR constructor issues


# ============================================================================
# PART 1: STABILITY METRICS
# ============================================================================

@dataclass
class StabilityMetrics:
    """Container for multiple stability indicators"""
    energy_based: Dict[str, float]
    frequency_based: Dict[str, float]
    trajectory_based: Dict[str, float]
    spectral_based: Dict[str, float]
    combined_score: float
    is_stable: bool


class SwingStabilityAnalyzer:
    """
    Stability analysis for swing equation systems using control-theoretic definitions.
    Stable = bounded trajectories (frequencies don't grow unboundedly)
    Unstable = unbounded trajectories (frequencies diverge)
    """
    
    def __init__(self, sys, 
                 frequency_bound: float = 10.0,  # rad/s - if |omega| exceeds this, it's unstable
                 divergence_rate: float = 0.1,   # If frequencies grow faster than this, unstable
                 energy_growth_threshold: float = 0.05):  # Max allowed energy growth rate
        self.sys = sys
        self.frequency_bound = frequency_bound
        self.divergence_rate = divergence_rate
        self.energy_growth_threshold = energy_growth_threshold
        
        # Compute equilibrium for reference
        x_eq = sys.goal_point.squeeze()
        self.E_eq = sys.energy_function(x_eq.unsqueeze(0)).item()
        
    def compute_frequency_stability(self, trajectory: torch.Tensor) -> Dict[str, float]:
        """Check if frequencies remain bounded (control system stability)."""
        omega = trajectory[:, self.sys.N_NODES-1:]  # Extract frequencies
        
        # Check if frequencies are bounded
        max_freq = omega.abs().max().item()
        is_bounded = max_freq < self.frequency_bound
        
        # Check divergence rate using linear regression on log scale
        omega_norm = omega.norm(dim=-1)
        if len(omega_norm) > 10:
            t = torch.arange(len(omega_norm), dtype=torch.float32) * self.sys.dt
            
            # Avoid log of zero
            omega_norm_safe = omega_norm + 1e-10
            
            # Linear fit to log(omega) to get exponential growth rate
            log_omega = torch.log(omega_norm_safe)
            # Remove any NaN or Inf values
            valid_mask = torch.isfinite(log_omega)
            if valid_mask.sum() > 2:
                t_valid = t[valid_mask]
                log_omega_valid = log_omega[valid_mask]
                
                # Simple linear regression
                n = len(t_valid)
                if n > 2:
                    t_mean = t_valid.mean()
                    log_omega_mean = log_omega_valid.mean()
                    
                    numerator = ((t_valid - t_mean) * (log_omega_valid - log_omega_mean)).sum()
                    denominator = ((t_valid - t_mean) ** 2).sum()
                    
                    if denominator > 1e-10:
                        growth_rate = (numerator / denominator).item()
                    else:
                        growth_rate = 0.0
                else:
                    growth_rate = 0.0
            else:
                growth_rate = 0.0
        else:
            growth_rate = 0.0
        
        # Check oscillation amplitude (bounded oscillations are stable)
        omega_std = omega.std(dim=0).mean().item()
        
        return {
            "max_frequency": max_freq,
            "is_bounded": is_bounded,
            "growth_rate": growth_rate,
            "oscillation_amplitude": omega_std,
            "final_frequency_norm": omega_norm[-1].item() if len(omega_norm) > 0 else 0.0,
        }
    
    def compute_energy_metric(self, trajectory: torch.Tensor) -> Dict[str, float]:
        """Energy-based stability metric (simplified)."""
        E_traj = torch.tensor([
            self.sys.energy_function(trajectory[i:i+1]).item() 
            for i in range(len(trajectory))
        ])
        
        E_max = E_traj.max().item()
        E_final = E_traj[-1].item()
        E_mean = E_traj.mean().item()
        
        # Energy growth rate (exponential fit)
        if len(E_traj) > 1:
            t = torch.arange(len(E_traj), dtype=torch.float32) * self.sys.dt
            log_E = torch.log(E_traj.abs() + 1e-10)
            
            valid_mask = torch.isfinite(log_E)
            if valid_mask.sum() > 2:
                t_valid = t[valid_mask]
                log_E_valid = log_E[valid_mask]
                n = len(t_valid)
                
                # Linear regression for growth rate
                t_mean = t_valid.mean()
                log_E_mean = log_E_valid.mean()
                
                numerator = ((t_valid - t_mean) * (log_E_valid - log_E_mean)).sum()
                denominator = ((t_valid - t_mean) ** 2).sum()
                
                if denominator > 1e-10:
                    energy_growth_rate = (numerator / denominator).item()
                else:
                    energy_growth_rate = 0.0
            else:
                energy_growth_rate = 0.0
        else:
            energy_growth_rate = 0.0
        
        return {
            "max_energy": E_max,
            "final_energy": E_final,
            "mean_energy": E_mean,
            "energy_growth_rate": energy_growth_rate,
        }
    
    def compute_convergence_metric(self, trajectory: torch.Tensor) -> Dict[str, float]:
        """Check if trajectory converges to equilibrium or a limit cycle."""
        x_eq = self.sys.goal_point.squeeze()
        
        distances = torch.tensor([
            (trajectory[i] - x_eq).norm().item() 
            for i in range(len(trajectory))
        ])
        
        # Check if converging, diverging, or oscillating
        if len(distances) > 10:
            # Check last 25% of trajectory
            last_quarter = len(distances) // 4
            recent_distances = distances[-last_quarter:]
            
            # Trend in recent distances
            is_converging = recent_distances[-1] < recent_distances[0]
            is_bounded = distances.max() < 100.0  # Some reasonable bound
            
            # Check for limit cycle (bounded but not converging)
            variance = recent_distances.std().item()
            mean_dist = recent_distances.mean().item()
            is_oscillating = variance > 0.01 * mean_dist and is_bounded
        else:
            is_converging = False
            is_bounded = True
            is_oscillating = False
        
        return {
            "final_distance": distances[-1].item(),
            "max_distance": distances.max().item(),
            "is_converging": is_converging,
            "is_bounded": is_bounded,
            "is_oscillating": is_oscillating,
        }
    
    def compute_combined_stability(self, trajectory: torch.Tensor) -> StabilityMetrics:
        """
        Compute stability based on control-theoretic definitions:
        - Stable if frequencies remain bounded
        - Unstable if frequencies grow unboundedly
        """
        freq_metrics = self.compute_frequency_stability(trajectory)
        energy_metrics = self.compute_energy_metric(trajectory)
        convergence_metrics = self.compute_convergence_metric(trajectory)
        
        # Simplified spectral metrics (placeholder)
        spectral_metrics = {
            "max_eigenvalue_real": 0.0,
            "spectral_radius": 1.0,
            "stability_margin": 0.0,
            "discrete_stable": True
        }
        
        # Control-theoretic stability: bounded trajectories
        is_stable = (
            freq_metrics["is_bounded"] and  # Frequencies don't blow up
            freq_metrics["growth_rate"] < self.divergence_rate and  # Not growing exponentially
            energy_metrics["energy_growth_rate"] < self.energy_growth_threshold  # Energy not exploding
        )
        
        # Combined score (higher is better, max 1.0)
        score_components = []
        
        # Frequency boundedness (most important)
        if freq_metrics["is_bounded"]:
            freq_score = 1.0 / (1.0 + freq_metrics["growth_rate"])
        else:
            freq_score = 0.0
        score_components.append(0.5 * freq_score)  # 50% weight
        
        # Energy boundedness
        energy_score = 1.0 / (1.0 + abs(energy_metrics["energy_growth_rate"]))
        score_components.append(0.3 * energy_score)  # 30% weight
        
        # Convergence (nice to have but not required for stability)
        if convergence_metrics["is_bounded"]:
            conv_score = 1.0 / (1.0 + convergence_metrics["final_distance"])
        else:
            conv_score = 0.0
        score_components.append(0.2 * conv_score)  # 20% weight
        
        combined_score = sum(score_components)
        
        return StabilityMetrics(
            energy_based=energy_metrics,
            frequency_based=freq_metrics,
            trajectory_based=convergence_metrics,
            spectral_based=spectral_metrics,
            combined_score=combined_score,
            is_stable=is_stable
        )
    
    # Keep the old methods for backwards compatibility but simplified
    def compute_rocof_metric(self, trajectory: torch.Tensor) -> Dict[str, float]:
        """Rate of Change of Frequency metric (simplified)."""
        rocof_values = []
        
        for i in range(len(trajectory) - 1):
            omega_curr = trajectory[i, self.sys.N_NODES-1:]
            omega_next = trajectory[i+1, self.sys.N_NODES-1:]
            
            rocof = (omega_next - omega_curr) / self.sys.dt
            rocof_hz = rocof * 60.0 / (2.0 * np.pi)
            rocof_values.append(rocof_hz)
        
        if rocof_values:
            rocof_tensor = torch.stack(rocof_values)
            return {
                "max_rocof": rocof_tensor.abs().max().item(),
                "mean_rocof": rocof_tensor.abs().mean().item(),
                "rocof_violations": 0.0,  # Not used in new stability definition
                "final_rocof": rocof_tensor[-1].abs().max().item(),
            }
        else:
            return {"max_rocof": 0.0, "mean_rocof": 0.0, "rocof_violations": 0.0, "final_rocof": 0.0}


# ============================================================================
# PART 2: BENCHMARK RESULTS STRUCTURE
# ============================================================================

@dataclass
class BenchmarkResults:
    """Store benchmark results for analysis"""
    method: str
    latent_dim: int
    stable_errors: Dict[str, torch.Tensor]
    unstable_errors: Dict[str, torch.Tensor]
    stable_trajectories_full: List[torch.Tensor]
    stable_trajectories_rom: List[torch.Tensor]
    unstable_trajectories_full: List[torch.Tensor]
    unstable_trajectories_rom: List[torch.Tensor]
    computation_time_full: float
    computation_time_rom: float
    worst_case_states: torch.Tensor
    worst_case_errors: torch.Tensor
    stability_preservation: Dict[str, float]


# ============================================================================
# PART 3: SYSTEM CREATION AND CONTROL
# ============================================================================

def create_10_generator_system() -> SwingEquationSystem:
    """Create the 10-generator system with your existing parameters."""
    
    M = torch.tensor([1.4, 1.01, 1.1766666, 0.95333344, 0.8666667,
                      1.16, 0.88, 0.81, 1.1500001, 16.666668], dtype=torch.float32)
    
    D = torch.tensor([0.19666669, 0.28833333, 0.28833333, 0.28833333, 0.28833333,
                      0.28833333, 0.28833333, 0.28833333, 0.30366668, 0.30366668], 
                     dtype=torch.float32)
    
    P = torch.tensor([-0.19983394, -0.25653884, -0.25191885, -0.10242008, -0.34510365,
                       0.23206371, 0.4404325, 0.5896664, 0.26257738, -0.36892462], 
                     dtype=torch.float32)
    
    K = torch.tensor([[18.217514, 1.1387165, 1.360604, 1.2791332, 0.572532,
                       1.2913872, 1.051677, 3.1750703, 1.7979614, 5.1949754],
                      [1.1387165, 13.809675, 2.2804017, 0.7847816, 0.3512633,
                       0.7922997, 0.6452313, 0.6032209, 0.5339053, 3.1851373],
                      [1.360604, 2.2804017, 14.325089, 1.0379374, 0.46457428,
                       1.0478808, 0.85337085, 0.7284146, 0.66612864, 3.2071328],
                      [1.2791332, 0.7847816, 1.0379374, 14.242994, 2.7465186,
                       2.0251245, 1.6492164, 0.7610195, 0.90347135, 1.6209806],
                      [0.572532, 0.3512633, 0.46457428, 2.7465186, 10.82826,
                       0.90643305, 0.7381789, 0.34062758, 0.4043881, 0.72554076],
                      [1.2913872, 0.7922997, 1.0478808, 2.0251245, 0.90643305,
                       16.046877, 4.1242394, 0.76831, 0.9121265, 1.6365094],
                      [1.051677, 0.6452313, 0.85337085, 1.6492164, 0.7381789,
                       4.1242394, 12.7116995, 0.62569463, 0.74281555, 1.3327368],
                      [3.1750703, 0.6032209, 0.7284146, 0.7610195, 0.34062758,
                       0.76831, 0.62569463, 11.512734, 1.4515272, 2.5534966],
                      [1.7979614, 0.5339053, 0.66612864, 0.90347135, 0.4043881,
                       0.9121265, 0.74281555, 1.4515272, 10.306445, 1.6531545],
                      [5.1949754, 3.1851373, 3.2071328, 1.6209806, 0.72554076,
                       1.6365094, 1.3327368, 2.5534966, 1.6531545, 25.04985]], 
                     dtype=torch.float32)
    
    params = dict(M=M, D=D, P=P, K=K)
    sys = SwingEquationSystem(params, dt=0.01)
    sys.delta_star = sys.solve_equilibrium_robust(verbose=False)
    
    return sys


def linear_damping_controller(sys: SwingEquationSystem, K_d: float = 0.5) -> callable:
    """Create a simple linear damping controller u = -K_d * omega"""
    
    def controller(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        omega = x[:, sys.N_NODES-1:]
        u = -K_d * omega
        return u
    
    return controller


# ============================================================================
# PART 4: INITIAL CONDITION CLASSIFICATION
# ============================================================================

def classify_initial_conditions(sys, analyzer: SwingStabilityAnalyzer,
                               n_samples: int = 100,
                               controller: callable = None,
                               T: float = 3.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate and classify initial conditions. REQUIRES 40-60% stable/unstable split or exits.
    
    Returns:
        stable_ics: Initial conditions classified as stable
        unstable_ics: Initial conditions classified as unstable
        stability_scores: Continuous stability scores
    """
    if controller is None:
        controller = linear_damping_controller(sys)
    
    print("\n" + "="*80)
    print("GENERATING AND CLASSIFYING INITIAL CONDITIONS")
    print("="*80)
    print("Requirement: 40-60% stable/unstable split for valid benchmark")
    
    x_eq = sys.goal_point.squeeze()
    
    # We'll generate half with small perturbations (likely stable)
    # and half with large perturbations (likely unstable)
    n_small = n_samples // 2
    n_large = n_samples - n_small
    
    initial_conditions = []
    
    # Small perturbations (likely to be stable)
    print(f"Generating {n_small} small perturbations (0.001 to 0.05)...")
    for i in range(n_small):
        scale = np.random.uniform(0.001, 0.05)
        x = x_eq + scale * torch.randn_like(x_eq)
        initial_conditions.append(x)
    
    # Large perturbations (likely to be unstable)
    print(f"Generating {n_large} large perturbations (0.2 to 1.0)...")
    for i in range(n_large):
        scale = np.random.uniform(8, 16.0)  # Much larger perturbations
        x = x_eq + scale * torch.randn_like(x_eq)
        initial_conditions.append(x)
    
    # Shuffle so they're not in order
    indices = torch.randperm(n_samples)
    initial_conditions = torch.stack(initial_conditions)[indices]
    
    # Classify each IC
    stable_mask = torch.zeros(n_samples, dtype=torch.bool)
    stability_scores = torch.zeros(n_samples)
    
    print(f"\nClassifying {n_samples} initial conditions...")
    
    n_stable_total = 0
    n_unstable_total = 0
    
    for i in range(n_samples):
        x = initial_conditions[i:i+1]
        trajectory = [x.squeeze()]
        
        n_steps = int(T / sys.dt)
        diverged = False
        
        for t in range(n_steps):
            u = controller(x)
            f = sys._f(x, sys.nominal_params)
            g = sys._g(x, sys.nominal_params)
            xdot = f.squeeze(-1) + (g @ u.unsqueeze(-1)).squeeze(-1)
            x = x + sys.dt * xdot
            
            # Check if trajectory is diverging badly
            if x.norm() > 1000 or not torch.isfinite(x).all():
                diverged = True
                # Fill rest with last value
                for _ in range(t, n_steps):
                    trajectory.append(x.squeeze())
                break
            
            trajectory.append(x.squeeze())
        
        trajectory = torch.stack(trajectory)
        
        # If diverged, it's definitely unstable
        if diverged:
            stable_mask[i] = False
            stability_scores[i] = 0.0
            n_unstable_total += 1
        else:
            # Compute stability metrics
            metrics = analyzer.compute_combined_stability(trajectory)
            stable_mask[i] = metrics.is_stable
            stability_scores[i] = metrics.combined_score
            
            if metrics.is_stable:
                n_stable_total += 1
            else:
                n_unstable_total += 1
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{n_samples} ICs")
            print(f"    Running total: {n_stable_total} stable, {n_unstable_total} unstable")
    
    # Split into stable and unstable
    stable_ics = initial_conditions[stable_mask]
    unstable_ics = initial_conditions[~stable_mask]
    
    # Calculate percentage
    stable_percentage = 100 * len(stable_ics) / n_samples
    unstable_percentage = 100 * len(unstable_ics) / n_samples
    
    print(f"\nFinal Classification Results:")
    print(f"  Stable: {len(stable_ics)} conditions ({stable_percentage:.1f}%)")
    print(f"  Unstable: {len(unstable_ics)} conditions ({unstable_percentage:.1f}%)")
    print(f"  Score range: [{stability_scores.min():.3f}, {stability_scores.max():.3f}]")
    
    # HARD REQUIREMENT: Must have 40-60% split
    if stable_percentage < 40 or stable_percentage > 60:
        print("\n" + "="*80)
        print("ERROR: INVALID SPLIT FOR BENCHMARKING")
        print("="*80)
        print(f"Got {stable_percentage:.1f}% stable, {unstable_percentage:.1f}% unstable")
        print("Required: 40-60% stable for valid benchmark")
        print("\nSuggestions to fix:")
        
        if stable_percentage > 65:
            print("  Too many stable trajectories. Try:")
            print("  1. Increase large perturbation range (e.g., 0.5 to 2.0)")
            print("  2. Decrease controller damping")
            print("  3. Increase simulation time T to catch late instabilities")
        else:
            print("  Too many unstable trajectories. Try:")
            print("  1. Decrease large perturbation range (e.g., 0.1 to 0.5)")
            print("  2. Increase controller damping")
            print("  3. Adjust stability analyzer thresholds")
        
        print("\nExiting - fix parameters and try again.")
        print("="*80)
        
        # Exit the program
        import sys as sys_module
        sys_module.exit(1)
    
    print("\n✓ Valid split achieved! Proceeding with benchmark...")
    
    return stable_ics, unstable_ics, stability_scores


# ============================================================================
# PART 5: SIMULATION FUNCTIONS
# ============================================================================

def simulate_trajectories(sys: SwingEquationSystem,
                         initial_conditions: torch.Tensor,
                         controller: callable,
                         T: float = 5.0,
                         method: str = "rk4") -> Tuple[List[torch.Tensor], float]:
    """Simulate trajectories for given initial conditions."""
    
    n_steps = int(T / sys.dt)
    n_ics = len(initial_conditions)
    trajectories = []
    
    start_time = time.time()
    
    for i in range(n_ics):
        x = initial_conditions[i:i+1]
        traj = [x.squeeze()]
        
        # Print progress for slow simulations
        if i > 0 and i % 5 == 0:
            elapsed = time.time() - start_time
            if elapsed > 2:
                print(f"    Full system progress: {i}/{n_ics} trajectories...")
        
        for t in range(n_steps):
            u = controller(x)
            
            if method == "euler":
                f = sys._f(x, sys.nominal_params)
                g = sys._g(x, sys.nominal_params)
                xdot = f.squeeze(-1) + (g @ u.unsqueeze(-1)).squeeze(-1)
                x = x + sys.dt * xdot
            else:  # rk4
                def dynamics(x_curr):
                    u_curr = controller(x_curr)
                    f = sys._f(x_curr, sys.nominal_params)
                    g = sys._g(x_curr, sys.nominal_params)
                    return f.squeeze(-1) + (g @ u_curr.unsqueeze(-1)).squeeze(-1)
                
                k1 = dynamics(x)
                k2 = dynamics(x + 0.5 * sys.dt * k1)
                k3 = dynamics(x + 0.5 * sys.dt * k2)
                k4 = dynamics(x + sys.dt * k3)
                x = x + (sys.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            traj.append(x.squeeze())
        
        trajectories.append(torch.stack(traj))
    
    computation_time = time.time() - start_time
    return trajectories, computation_time


def simulate_rom_trajectories(sys: SwingEquationSystem,
                             reducer,
                             initial_conditions: torch.Tensor,
                             controller: callable,
                             T: float = 5.0) -> Tuple[List[torch.Tensor], float]:
    """Simulate reduced order model trajectories."""
    
    n_steps = int(T / sys.dt)
    n_ics = len(initial_conditions)
    trajectories = []
    
    print(f"    Starting ROM simulation for {n_ics} ICs, {n_steps} steps each...")
    
    start_time = time.time()
    
    # Determine integration method based on reducer type
    if hasattr(reducer, 'J_symplectic'):
        # Use symplectic integration for SPR
        method = "symplectic"
        print(f"    Using symplectic integration for SPR")
    else:
        # Use RK4 for other reducers
        method = "rk4"
    
    for i in range(n_ics):
        x0 = initial_conditions[i:i+1]
        
        # Print progress for longer simulations
        if i > 0 and i % 5 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / i) * (n_ics - i)
            print(f"    ROM IC {i+1}/{n_ics} (elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s)...")
        
        try:
            # Use the appropriate method for each reducer type
            traj = rollout_rom(reducer, sys, x0, n_steps, controller, dt=sys.dt, method=method)
            trajectories.append(traj.squeeze(0))
                
        except Exception as e:
            print(f"    Warning: ROM simulation failed for IC {i+1}: {e}")
            # Use a simple constant trajectory as fallback
            fallback_traj = x0.squeeze(0).unsqueeze(0).repeat(n_steps+1, 1)
            trajectories.append(fallback_traj)
    
    computation_time = time.time() - start_time
    print(f"    ROM simulation completed in {computation_time:.1f}s")
    
    return trajectories, computation_time


# ============================================================================
# PART 6: ERROR COMPUTATION AND ANALYSIS
# ============================================================================

def compute_trajectory_errors(trajs_full: List[torch.Tensor], 
                            trajs_rom: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Compute various error metrics between trajectories."""
    
    n_traj = len(trajs_full)
    pointwise_errors = []
    max_errors = []
    mean_errors = []
    final_errors = []
    
    for i in range(n_traj):
        error = (trajs_full[i] - trajs_rom[i]).norm(dim=-1)
        pointwise_errors.append(error)
        max_errors.append(error.max().item())
        mean_errors.append(error.mean().item())
        final_errors.append(error[-1].item())
    
    return {
        "pointwise": pointwise_errors,
        "max": torch.tensor(max_errors),
        "mean": torch.tensor(mean_errors),
        "final": torch.tensor(final_errors),
    }


def analyze_stability_preservation(analyzer: SwingStabilityAnalyzer,
                                  trajs_full: List[torch.Tensor],
                                  trajs_rom: List[torch.Tensor]) -> Dict[str, float]:
    """Analyze how well stability is preserved in reduction."""
    
    preservation_scores = []
    classification_matches = []
    
    for traj_full, traj_rom in zip(trajs_full, trajs_rom):
        metrics_full = analyzer.compute_combined_stability(traj_full)
        metrics_rom = analyzer.compute_combined_stability(traj_rom)
        
        # Score preservation
        score_diff = abs(metrics_full.combined_score - metrics_rom.combined_score)
        preservation_scores.append(np.exp(-score_diff))
        
        # Classification match
        classification_matches.append(float(metrics_full.is_stable == metrics_rom.is_stable))
    
    return {
        "mean_preservation": np.mean(preservation_scores),
        "classification_accuracy": np.mean(classification_matches),
        "worst_preservation": np.min(preservation_scores),
    }


# ============================================================================
# PART 7: SINGLE REDUCER BENCHMARK
# ============================================================================

def benchmark_reducer(sys: SwingEquationSystem,
                      reducer,
                      method_name: str,
                      stable_ics: torch.Tensor,
                      unstable_ics: torch.Tensor,
                      controller: callable,
                      analyzer: SwingStabilityAnalyzer,
                      T: float = 5.0) -> BenchmarkResults:
    """Benchmark a single reducer with stability metrics."""
    
    print(f"\n" + "="*60)
    print(f"Benchmarking {method_name} (d={reducer.latent_dim})")
    print("="*60)
    
    # Use all ICs if in full mode, or limit for quick mode
    if T < 5.0:  # Quick mode indicator
        max_ics_per_type = 5
        stable_ics = stable_ics[:max_ics_per_type]
        unstable_ics = unstable_ics[:max_ics_per_type]
        print(f"  Quick mode: Testing with {len(stable_ics)} stable and {len(unstable_ics)} unstable ICs")
    else:
        print(f"  Full mode: Testing with {len(stable_ics)} stable and {len(unstable_ics)} unstable ICs")
    
    print(f"  Simulation time: {T} seconds")
    
    # Simulate full system
    print("  Simulating full system...")
    stable_traj_full, time_full_stable = simulate_trajectories(
        sys, stable_ics, controller, T, method="rk4"
    )
    unstable_traj_full, time_full_unstable = simulate_trajectories(
        sys, unstable_ics, controller, T, method="rk4"
    )
    
    # Simulate ROM
    print("  Simulating ROM...")
    stable_traj_rom, time_rom_stable = simulate_rom_trajectories(
        sys, reducer, stable_ics, controller, T
    )
    unstable_traj_rom, time_rom_unstable = simulate_rom_trajectories(
        sys, reducer, unstable_ics, controller, T
    )
    
    # Compute errors
    stable_errors = compute_trajectory_errors(stable_traj_full, stable_traj_rom)
    unstable_errors = compute_trajectory_errors(unstable_traj_full, unstable_traj_rom)
    
    # Analyze stability preservation
    print("  Analyzing stability preservation...")
    stable_preservation = analyze_stability_preservation(
        analyzer, stable_traj_full, stable_traj_rom
    )
    unstable_preservation = analyze_stability_preservation(
        analyzer, unstable_traj_full, unstable_traj_rom
    )
    
    # Overall preservation
    overall_preservation = {
        "stable": stable_preservation,
        "unstable": unstable_preservation,
        "overall_accuracy": (stable_preservation["classification_accuracy"] + 
                           unstable_preservation["classification_accuracy"]) / 2
    }
    
    # Identify worst cases
    all_mean_errors = torch.cat([stable_errors["mean"], unstable_errors["mean"]])
    worst_indices = torch.topk(all_mean_errors, k=min(10, len(all_mean_errors)))[1]
    
    all_ics = torch.cat([stable_ics, unstable_ics], dim=0)
    worst_ics = all_ics[worst_indices]
    worst_errors = all_mean_errors[worst_indices]
    
    # Print summary
    print(f"\n  Results for {method_name}:")
    print(f"    Stable trajectories:")
    print(f"      Mean error: {stable_errors['mean'].mean():.6f}")
    print(f"      Max error: {stable_errors['max'].max():.6f}")
    print(f"      Stability preserved: {stable_preservation['classification_accuracy']:.1%}")
    print(f"    Unstable trajectories:")
    print(f"      Mean error: {unstable_errors['mean'].mean():.6f}")
    print(f"      Max error: {unstable_errors['max'].max():.6f}")
    print(f"      Stability preserved: {unstable_preservation['classification_accuracy']:.1%}")
    print(f"    Computation time:")
    print(f"      Full: {time_full_stable + time_full_unstable:.2f}s")
    print(f"      ROM: {time_rom_stable + time_rom_unstable:.2f}s")
    if (time_rom_stable + time_rom_unstable) > 0:
        print(f"      Speedup: {(time_full_stable + time_full_unstable) / (time_rom_stable + time_rom_unstable):.2f}x")
    
    return BenchmarkResults(
        method=method_name,
        latent_dim=reducer.latent_dim,
        stable_errors=stable_errors,
        unstable_errors=unstable_errors,
        stable_trajectories_full=stable_traj_full,
        stable_trajectories_rom=stable_traj_rom,
        unstable_trajectories_full=unstable_traj_full,
        unstable_trajectories_rom=unstable_traj_rom,
        computation_time_full=time_full_stable + time_full_unstable,
        computation_time_rom=time_rom_stable + time_rom_unstable,
        worst_case_states=worst_ics,
        worst_case_errors=worst_errors,
        stability_preservation=overall_preservation,
    )


# ============================================================================
# PART 8: VISUALIZATION
# ============================================================================

def visualize_comprehensive_results(results: List[BenchmarkResults], 
                                   sys: SwingEquationSystem,
                                   analyzer: SwingStabilityAnalyzer):
    """Create comprehensive visualization of benchmark results."""
    
    n_methods = len(results)
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Error comparison with stability preservation
    ax1 = plt.subplot(3, 4, 1)
    methods = [r.method for r in results]
    stable_errors = [r.stable_errors["mean"].mean().item() for r in results]
    unstable_errors = [r.unstable_errors["mean"].mean().item() for r in results]
    
    x = np.arange(len(methods))
    width = 0.35
    ax1.bar(x - width/2, stable_errors, width, label='Stable', color='blue', alpha=0.7)
    ax1.bar(x + width/2, unstable_errors, width, label='Unstable', color='red', alpha=0.7)
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Mean Error')
    ax1.set_title('Trajectory Error Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45)
    ax1.legend()
    ax1.set_yscale('log')
    
    # 2. Stability preservation accuracy
    ax2 = plt.subplot(3, 4, 2)
    preservation = [r.stability_preservation["overall_accuracy"] for r in results]
    colors = ['green' if p > 0.9 else 'orange' if p > 0.7 else 'red' for p in preservation]
    ax2.bar(methods, preservation, color=colors, alpha=0.7)
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Classification Accuracy')
    ax2.set_title('Stability Preservation')
    ax2.axhline(y=0.9, color='k', linestyle='--', alpha=0.3, label='90% threshold')
    ax2.set_ylim([0, 1.1])
    ax2.legend()
    
    # 3. Computational speedup
    ax3 = plt.subplot(3, 4, 3)
    speedups = [r.computation_time_full / r.computation_time_rom for r in results]
    ax3.bar(methods, speedups, color='purple', alpha=0.7)
    ax3.set_xlabel('Method')
    ax3.set_ylabel('Speedup Factor')
    ax3.set_title('Computational Efficiency')
    ax3.axhline(y=1, color='k', linestyle='--', alpha=0.3)
    
    # 4. Dimension vs Error tradeoff
    ax4 = plt.subplot(3, 4, 4)
    dims = [r.latent_dim for r in results]
    total_errors = [(r.stable_errors["mean"].mean() + r.unstable_errors["mean"].mean()).item()/2 
                   for r in results]
    ax4.scatter(dims, total_errors, s=100, c=preservation, cmap='RdYlGn', vmin=0, vmax=1)
    ax4.set_xlabel('Latent Dimension')
    ax4.set_ylabel('Average Error')
    ax4.set_title('Dimension-Error Tradeoff')
    ax4.set_yscale('log')
    cbar = plt.colorbar(ax4.scatter(dims, total_errors, s=100, c=preservation, 
                                    cmap='RdYlGn', vmin=0, vmax=1), ax=ax4)
    cbar.set_label('Stability Preservation')
    
    # 5-8. Sample trajectories for best method
    best_idx = np.argmax([r.stability_preservation["overall_accuracy"] for r in results])
    best_result = results[best_idx]
    
    # Time array
    t = np.arange(len(best_result.stable_trajectories_full[0])) * sys.dt
    
    # 5. Stable trajectory - angles
    ax5 = plt.subplot(3, 4, 5)
    traj_idx = 0
    traj_full = best_result.stable_trajectories_full[traj_idx]
    traj_rom = best_result.stable_trajectories_rom[traj_idx]
    ax5.plot(t, traj_full[:, 0].detach().cpu().numpy(), 'b-', label='Full', linewidth=2)
    ax5.plot(t, traj_rom[:, 0].detach().cpu().numpy(), 'r--', label='ROM', linewidth=2)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('θ₁₂ (rad)')
    ax5.set_title(f'Stable: Angle ({best_result.method})')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Stable trajectory - frequency
    ax6 = plt.subplot(3, 4, 6)
    ax6.plot(t, traj_full[:, sys.N_NODES-1].detach().cpu().numpy(), 'b-', label='Full', linewidth=2)
    ax6.plot(t, traj_rom[:, sys.N_NODES-1].detach().cpu().numpy(), 'r--', label='ROM', linewidth=2)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('ω₁ (rad/s)')
    ax6.set_title(f'Stable: Frequency ({best_result.method})')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Unstable trajectory - angles
    ax7 = plt.subplot(3, 4, 7)
    traj_full = best_result.unstable_trajectories_full[traj_idx]
    traj_rom = best_result.unstable_trajectories_rom[traj_idx]
    ax7.plot(t, traj_full[:, 0].detach().cpu().numpy(), 'b-', label='Full', linewidth=2)
    ax7.plot(t, traj_rom[:, 0].detach().cpu().numpy(), 'r--', label='ROM', linewidth=2)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('θ₁₂ (rad)')
    ax7.set_title(f'Unstable: Angle ({best_result.method})')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Unstable trajectory - frequency
    ax8 = plt.subplot(3, 4, 8)
    ax8.plot(t, traj_full[:, sys.N_NODES-1].detach().cpu().numpy(), 'b-', label='Full', linewidth=2)
    ax8.plot(t, traj_rom[:, sys.N_NODES-1].detach().cpu().numpy(), 'r--', label='ROM', linewidth=2)
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('ω₁ (rad/s)')
    ax8.set_title(f'Unstable: Frequency ({best_result.method})')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Energy evolution comparison
    ax9 = plt.subplot(3, 4, 9)
    E_full = [sys.energy_function(traj_full[i:i+1]).detach().cpu().item() for i in range(len(traj_full))]
    E_rom = [sys.energy_function(traj_rom[i:i+1]).detach().cpu().item() for i in range(len(traj_rom))]
    ax9.plot(t, E_full, 'b-', label='Full', linewidth=2)
    ax9.plot(t, E_rom, 'r--', label='ROM', linewidth=2)
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Energy')
    ax9.set_title(f'Energy Evolution ({best_result.method})')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. Error evolution over time
    ax10 = plt.subplot(3, 4, 10)
    for i, result in enumerate(results):
        stable_error_mean = torch.stack(result.stable_errors["pointwise"]).mean(dim=0).detach().cpu().numpy()
        ax10.plot(t, stable_error_mean, label=result.method, alpha=0.7)
    ax10.set_xlabel('Time (s)')
    ax10.set_ylabel('Mean Error')
    ax10.set_title('Error Growth Over Time')
    ax10.legend()
    ax10.set_yscale('log')
    ax10.grid(True, alpha=0.3)
    
    # 11. Worst case analysis
    ax11 = plt.subplot(3, 4, 11)
    for i, result in enumerate(results):
        ax11.scatter([i] * len(result.worst_case_errors), 
                    result.worst_case_errors.cpu().numpy(),
                    alpha=0.5, s=50)
    ax11.set_xticks(range(len(methods)))
    ax11.set_xticklabels(methods, rotation=45)
    ax11.set_ylabel('Error')
    ax11.set_title('Worst Case Distribution')
    ax11.set_yscale('log')
    ax11.grid(True, alpha=0.3)
    
    # 12. Summary metrics table
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('tight')
    ax12.axis('off')
    
    table_data = []
    headers = ['Method', 'Dim', 'Error', 'Stability', 'Speedup']
    for r in results:
        avg_error = (r.stable_errors["mean"].mean() + r.unstable_errors["mean"].mean()).item() / 2
        table_data.append([
            r.method,
            f'{r.latent_dim}',
            f'{avg_error:.4f}',
            f'{r.stability_preservation["overall_accuracy"]:.1%}',
            f'{r.computation_time_full/r.computation_time_rom:.1f}x'
        ])
    
    table = ax12.table(cellText=table_data, colLabels=headers,
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    plt.suptitle('Comprehensive Dimension Reduction Benchmark Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('comprehensive_benchmark_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


# ============================================================================
# PART 9: REFINEMENT IDENTIFICATION
# ============================================================================

def identify_refinement_regions(results: List[BenchmarkResults],
                               stable_ics: torch.Tensor,
                               unstable_ics: torch.Tensor,
                               sys: SwingEquationSystem,
                               threshold_percentile: float = 90) -> torch.Tensor:
    """Identify regions needing refinement based on poor ROM performance."""
    
    print("\n" + "="*80)
    print("IDENTIFYING REFINEMENT REGIONS")
    print("="*80)
    
    all_ics = torch.cat([stable_ics, unstable_ics], dim=0)
    refinement_candidates = []
    
    for result in results:
        print(f"\n{result.method}:")
        
        # Find poor preservation cases
        poor_stable = []
        for i, (traj_full, traj_rom) in enumerate(zip(result.stable_trajectories_full, 
                                                      result.stable_trajectories_rom)):
            error = (traj_full - traj_rom).norm(dim=-1).mean()
            if error > result.stable_errors["mean"].mean() * 2:  # 2x average error
                poor_stable.append(stable_ics[i])
        
        poor_unstable = []
        for i, (traj_full, traj_rom) in enumerate(zip(result.unstable_trajectories_full,
                                                      result.unstable_trajectories_rom)):
            error = (traj_full - traj_rom).norm(dim=-1).mean()
            if error > result.unstable_errors["mean"].mean() * 2:
                poor_unstable.append(unstable_ics[i])
        
        if poor_stable:
            poor_stable = torch.stack(poor_stable)
            refinement_candidates.append(poor_stable)
            print(f"  Found {len(poor_stable)} poor stable cases")
        
        if poor_unstable:
            poor_unstable = torch.stack(poor_unstable)
            refinement_candidates.append(poor_unstable)
            print(f"  Found {len(poor_unstable)} poor unstable cases")
    
    if refinement_candidates:
        all_poor = torch.cat(refinement_candidates, dim=0)
        
        # Generate additional samples around poor regions
        n_additional = 100
        additional_samples = []
        
        for _ in range(n_additional):
            base_idx = torch.randint(0, len(all_poor), (1,))
            base_state = all_poor[base_idx]
            perturbation = 0.02 * torch.randn_like(base_state)
            new_sample = base_state + perturbation
            additional_samples.append(new_sample)
        
        additional_samples = torch.cat(additional_samples, dim=0)
        
        print(f"\nGenerated {len(additional_samples)} additional training samples")
        print("These samples focus on regions where ROM performance is poor")
        
        return additional_samples
    else:
        print("\nNo significant poor performance regions identified!")
        return torch.zeros(0, sys.n_dims)


# ============================================================================
# PART 10: MAIN BENCHMARK PIPELINE
# ============================================================================

def main(quick_mode=False):
    """Complete benchmark pipeline with stability-based classification.
    
    Args:
        quick_mode: If True, uses smaller datasets and shorter simulations for faster testing
    """
    
    print("="*80)
    print("COMPLETE SWING EQUATION DIMENSION REDUCTION BENCHMARK")
    if quick_mode:
        print("Running in QUICK MODE for faster testing")
    else:
        print("Running in FULL MODE")
    print("="*80)
    
    # 1. Create system
    print("\n1. Creating 10-generator system...")
    sys = create_10_generator_system()
    print(f"   System dimension: {sys.n_dims}")
    print(f"   Number of generators: {sys.N_NODES}")
    
    # 2. Create controller
    print("\n2. Creating linear damping controller...")
    controller = linear_damping_controller(sys, K_d=0.5)
    
    # 3. Create stability analyzer
    print("\n3. Creating stability analyzer...")
    analyzer = SwingStabilityAnalyzer(
        sys,
        frequency_bound=10.0,        # Frequencies above 10 rad/s are unstable
        divergence_rate=0.1,         # Exponential growth rate threshold
        energy_growth_threshold=0.05 # Max allowed energy growth rate
    )
    
    # 4. Generate and classify initial conditions
    if quick_mode:
        n_samples = 100  # Increased from 50 to find better split
        T_classify = 2.0
    else:
        n_samples = 200  # Increased from 100 to find better split
        T_classify = 3.0
        
    stable_ics, unstable_ics, stability_scores = classify_initial_conditions(
        sys, analyzer, n_samples=n_samples, controller=controller, T=T_classify
    )
    
    # 5. Collect training data for reducers
    print("\n5. Collecting training data for reducers...")
    n_train = 200 if quick_mode else 500
    X_train = []
    Xdot_train = []
    
    x_eq = sys.goal_point.squeeze()
    for scale in [0.01, 0.05, 0.1, 0.2]:
        X_batch = x_eq + scale * torch.randn(n_train // 4, sys.n_dims)
        X_train.append(X_batch)
        
        for x in X_batch:
            u = controller(x.unsqueeze(0))
            f = sys._f(x.unsqueeze(0), sys.nominal_params)
            g = sys._g(x.unsqueeze(0), sys.nominal_params)
            xdot = f.squeeze() + (g @ u.unsqueeze(-1)).squeeze()
            Xdot_train.append(xdot)
    
    X_train = torch.cat(X_train, dim=0)
    # Ensure X_train doesn't require gradients to avoid issues in reducers
    X_train = X_train.detach()
    Xdot_train = torch.stack(Xdot_train).detach()
    print(f"   Training samples: {len(X_train)}")
    
    # 6. Create reducers
    print("\n6. Creating dimension reduction models...")
    reducers = []
    
    # Symplectic Projection (following the working pattern from retain_freq.py)
    try:
        print("   Creating SPR...")
        A, J, R = sys.linearise(return_JR=True)
        # Use the correct constructor: sys as first argument, just like in retain_freq.py
        spr = SPR(sys, A, J, R, 12, enhanced=True, X_data=X_train)
        spr.full_dim = sys.n_dims
        reducers.append(("SPR-12", spr))
        print("   SPR-12 created successfully")
    except Exception as e:
        print(f"   SPR failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Operator Inference
    try:
        print("   Creating OpInf...")
        opinf = OpInfReducer(10, sys.n_dims, sys.n_controls)
        opinf.sys = sys
        V_fn = sys.energy_function
        V_min = V_fn(X_train).min().item()
        opinf.fit(X_train, Xdot_train, V_fn, V_min)
        reducers.append(("OpInf-10", opinf))
    except Exception as e:
        print(f"   OpInf failed: {e}")
    
    # Lyapunov Coherency
    try:
        print("   Creating LCR...")
        lcr = LCR(sys, 5, X_train)
        lcr.full_dim = sys.n_dims
        V_min = sys.energy_function(X_train).min().item()
        lcr.gamma = lcr.compute_gamma(V_min)
        # Only add if gamma is reasonable
        if lcr.gamma < 100:
            reducers.append(("LCR-5", lcr))
            print(f"   LCR-5 created with gamma={lcr.gamma:.2f}")
        else:
            print(f"   LCR-5 created but skipped due to high gamma={lcr.gamma:.2f}")
    except Exception as e:
        print(f"   LCR failed: {e}")
    
    # Try smaller OpInf
    try:
        print("   Creating smaller OpInf...")
        opinf_small = OpInfReducer(6, sys.n_dims, sys.n_controls)
        opinf_small.sys = sys
        V_fn = sys.energy_function
        V_min = V_fn(X_train).min().item()
        opinf_small.fit(X_train, Xdot_train, V_fn, V_min)
        reducers.append(("OpInf-6", opinf_small))
    except Exception as e:
        print(f"   Small OpInf failed: {e}")
    
    # Custom auto-selection from created reducers
    if len(reducers) > 0:
        print("   Auto-selecting from successfully created reducers...")
        # Simple selection based on gamma and dimension
        best_score = float('inf')
        best_reducer = None
        best_name = None
        
        for name, reducer in reducers:
            # Score = gamma + 0.02 * (latent_dim / full_dim)
            score = reducer.gamma + 0.02 * (reducer.latent_dim / sys.n_dims)
            print(f"     {name}: gamma={reducer.gamma:.2f}, dim={reducer.latent_dim}, score={score:.3f}")
            if score < best_score:
                best_score = score
                best_reducer = reducer
                best_name = name
        
        if best_reducer is not None:
            print(f"   Selected {best_name} as best reducer")
    
    if not reducers:
        print("ERROR: No reducers could be created!")
        return
    
    # 7. Benchmark each reducer
    print("\n7. Running comprehensive benchmarks...")
    results = []
    
    # Set simulation time based on mode
    T_benchmark = 2.0 if quick_mode else 5.0
    
    for name, reducer in reducers:
        try:
            result = benchmark_reducer(
                sys, reducer, name, stable_ics, unstable_ics, 
                controller, analyzer, T=T_benchmark
            )
            results.append(result)
        except Exception as e:
            print(f"   Benchmark failed for {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 8. Visualize results
    print("\n8. Creating comprehensive visualizations...")
    if results:
        if not quick_mode:
            fig = visualize_comprehensive_results(results, sys, analyzer)
            print("   Saved to 'comprehensive_benchmark_results.png'")
        else:
            print("   Skipping detailed visualizations in quick mode")
    
    # 9. Identify refinement regions
    if not quick_mode:
        additional_training_data = identify_refinement_regions(
            results, stable_ics, unstable_ics, sys
        )
    else:
        print("\n9. Skipping refinement identification (quick mode)")
        additional_training_data = torch.zeros(0, sys.n_dims)
    
    # 10. Summary report
    print("\n" + "="*80)
    print("FINAL BENCHMARK SUMMARY")
    print("="*80)
    
    print("\n%-15s %-6s %-10s %-10s %-12s %-8s" % 
          ("Method", "Dim", "Error", "Stability", "Preservation", "Speedup"))
    print("-"*75)
    
    for result in results:
        avg_error = (result.stable_errors["mean"].mean() + 
                    result.unstable_errors["mean"].mean()).item() / 2
        print("%-15s %-6d %-10.6f %-10.1f%% %-12.3f %-8.2fx" % (
            result.method,
            result.latent_dim,
            avg_error,
            result.stability_preservation["overall_accuracy"] * 100,
            result.stability_preservation["stable"]["mean_preservation"],
            result.computation_time_full / result.computation_time_rom
        ))
    
    # Find best method
    if results:
        # Score based on error, stability preservation, and speedup
        scores = []
        for r in results:
            error_score = 1.0 / (1.0 + (r.stable_errors["mean"].mean() + 
                                        r.unstable_errors["mean"].mean()).item())
            stability_score = r.stability_preservation["overall_accuracy"]
            speedup_score = min(1.0, r.computation_time_full / r.computation_time_rom / 10)
            total_score = 0.4 * error_score + 0.4 * stability_score + 0.2 * speedup_score
            scores.append(total_score)
        
        best_idx = np.argmax(scores)
        best = results[best_idx]
        
        print(f"\nBEST METHOD: {best.method} (d={best.latent_dim})")
        print(f"  Score: {scores[best_idx]:.3f}")
        print(f"  Average error: {(best.stable_errors['mean'].mean() + best.unstable_errors['mean'].mean()).item() / 2:.6f}")
        print(f"  Stability preservation: {best.stability_preservation['overall_accuracy']:.1%}")
        print(f"  Speedup: {best.computation_time_full / best.computation_time_rom:.2f}x")
    
    # 11. Save all results
    print("\n11. Saving results...")
    save_dict = {
        'results': results,
        'stable_ics': stable_ics,
        'unstable_ics': unstable_ics,
        'stability_scores': stability_scores,
        'additional_training_data': additional_training_data,
        'system_params': sys.nominal_params,
        'mode': 'quick' if quick_mode else 'full'
    }
    filename = 'quick_benchmark_results.pt' if quick_mode else 'complete_benchmark_results.pt'
    torch.save(save_dict, filename)
    print(f"   Saved to '{filename}'")
    
    print("\nBenchmark complete!")
    print("="*80)
    
    return results, stable_ics, unstable_ics, additional_training_data


if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Benchmark dimension reduction methods')
    parser.add_argument('--quick', '-q', action='store_true', 
                      help='Run in quick mode with smaller datasets')
    parser.add_argument('--full', '-f', action='store_true',
                      help='Run in full mode (default)')
    args = parser.parse_args()
    
    # Determine mode
    if args.quick:
        quick_mode = True
    else:
        quick_mode = False  # Default to full mode
    
    # Run the benchmark
    results, stable_ics, unstable_ics, additional_training = main(quick_mode=quick_mode)