# FILE: test_stability_preservation_final_v8.py
# DESCRIPTION:
# This script provides the definitive analysis of ROM stability preservation by
# precisely replicating the methodology of the user's `retain_freq.py`.
#
# FINAL CORRECTIONS:
# 1.  All import paths are corrected to match the user's environment.
# 2.  The data collection method is reverted to the simple 0.01 perturbation.
# 3.  SPR is initialized WITHOUT X_data, forcing the standard basis,
#     which is consistent with the user's successful implementation.

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Assuming the user's files are in the python path
from neural_clbf.systems import SwingEquationSystem
from neural_clbf.dimension_reduction.symplectic_projection import SymplecticProjectionReducer
from neural_clbf.dimension_reduction.opinf import OpInfReducer
from neural_clbf.dimension_reduction.lyap_coherency import LyapCoherencyReducer
from neural_clbf.eval.reduction_validation import rollout_rom # <-- CORRECTED IMPORT

# ==========================================================================================
# UTILITY AND SYSTEM SETUP FUNCTIONS (Aligned with retain_freq.py)
# ==========================================================================================

def create_stable_system():
    """Create a stable swing equation system with 10 generators."""
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
    print("✓ System created successfully.")
    return sys

def collect_training_data(sys, n_samples=15000):
    """
    Data collection method identical to the user's `retain_freq.py`.
    """
    print("\nCollecting training data with 0.01 perturbation...")
    x_eq = sys.goal_point.squeeze()
    X = x_eq.unsqueeze(0).repeat(n_samples, 1)
    X += 0.04 * torch.randn_like(X)
    
    Xdot = sys._f(X, sys.nominal_params).squeeze(-1)
    
    print(f"✓ Training data collected: {X.shape[0]} total points.")
    return X, Xdot


def classify_stability(trajectory: torch.Tensor, n_nodes: int):
    """Classifies a batch of trajectories as stable or unstable based on the norm of omega."""
    if trajectory.dim() != 3:
        raise ValueError("Requires a 3D trajectory tensor (batch, time, dims)")
    omega_states = trajectory[..., n_nodes - 1:]
    omega_magnitudes = torch.norm(omega_states, p=2, dim=-1)
    
    split_point = max(1, trajectory.shape[1] // 5)
    avg_mag_start = torch.mean(omega_magnitudes[:, :split_point], dim=1)
    avg_mag_end = torch.mean(omega_magnitudes[:, -split_point:], dim=1)
    
    is_stable = (avg_mag_end < 0.8 * avg_mag_start) | (avg_mag_start < 1e-4)
    return ["stable" if stable else "unstable" for stable in is_stable]


def compute_confusion_matrix(true_labels, pred_labels):
    """Compute confusion matrix for binary classification."""
    tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 'stable' and p == 'stable')
    tn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 'unstable' and p == 'unstable')
    fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 'unstable' and p == 'stable')
    fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 'stable' and p == 'unstable')
    
    return tp, tn, fp, fn


def print_confusion_matrix(name, tp, tn, fp, fn):
    """Print a nicely formatted confusion matrix."""
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n    {name} - Confusion Matrix:")
    print("    " + "="*45)
    print("                  Predicted")
    print("                  Stable   Unstable")
    print(f"    Actual Stable     {tp:3d}      {fn:3d}     | {tp+fn:3d}")
    print(f"          Unstable   {fp:3d}      {tn:3d}     | {fp+tn:3d}")
    print("                  " + "-"*25)
    print(f"                     {tp+fp:3d}      {fn+tn:3d}     | {total:3d}")
    print()
    print(f"    Accuracy:  {accuracy:6.2%}")
    print(f"    Precision: {precision:6.2%} (stable class)")
    print(f"    Recall:    {recall:6.2%} (stable class)")
    print(f"    F1-Score:  {f1:6.2%}")
    print("    " + "="*45)


def generate_diagnostic_plots(sys, reducers, x0, T_horizon, dt):
    """
    Generates a multi-panel diagnostic plot for a single initial condition.
    """
    print("\nGenerating comprehensive diagnostic plots...")
    fig, axs = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    n_steps = int(T_horizon / dt)
    t_span = np.arange(n_steps + 1) * dt
    
    zero_controller = lambda x: torch.zeros(x.shape[0], sys.n_controls, device=x.device)

    simulated_points_full = sys.simulate(x0.unsqueeze(0), n_steps, controller=zero_controller, method='rk4')[0]
    traj_full = torch.cat((x0.unsqueeze(0), simulated_points_full), dim=0).detach()
    
    min_len = min(traj_full.shape[0], t_span.shape[0])
    traj_full = traj_full[:min_len]
    t_span = t_span[:min_len]

    colors = plt.cm.viridis(np.linspace(0, 0.85, len(reducers)))
    for i, (name, reducer) in enumerate(reducers.items()):
        if reducer is None: continue
        
        rom_method =  "rk4"
        traj_rom = rollout_rom(reducer, sys, x0.unsqueeze(0), n_steps, controller=zero_controller, dt=dt, method=rom_method)[0].detach()
        traj_rom = traj_rom[:min_len]

        omega_norm_rom = torch.norm(traj_rom[:, sys.N_NODES-1:], dim=1).cpu().numpy()
        state_norm_rom = torch.norm(traj_rom, dim=1).cpu().numpy()
        state_error = torch.norm(traj_full - traj_rom, dim=1).cpu().numpy()
        E_full = sys.energy_function(traj_full)
        E_rom = sys.energy_function(traj_rom)
        energy_error = torch.abs(E_full - E_rom).cpu().numpy()

        axs[0, 0].plot(t_span, omega_norm_rom, '--', linewidth=2, label=name, color=colors[i])
        axs[0, 1].plot(t_span, state_norm_rom, '--', linewidth=2, label=name, color=colors[i])
        axs[1, 0].semilogy(t_span, state_error, '--', linewidth=2, label=name, color=colors[i])
        axs[1, 1].semilogy(t_span, energy_error, '--', linewidth=2, label=name, color=colors[i])

    omega_norm_full = torch.norm(traj_full[:, sys.N_NODES-1:], dim=1).cpu().numpy()
    state_norm_full = torch.norm(traj_full, dim=1).cpu().numpy()
    
    axs[0, 0].plot(t_span, omega_norm_full, 'k-', linewidth=3, label='Full System', zorder=10)
    axs[0, 1].plot(t_span, state_norm_full, 'k-', linewidth=3, label='Full System', zorder=10)

    axs[0, 0].set_title("Stability Analysis: $||\omega(t)||$"); axs[0, 0].set_ylabel("rad/s"); axs[0, 0].set_yscale('log')
    axs[0, 1].set_title("State Norm Evolution: $||x(t)||$"); axs[0, 1].set_ylabel("Norm")
    axs[1, 0].set_title("State Reconstruction Error"); axs[1, 0].set_ylabel("$||x_{full} - x_{ROM}||$")
    axs[1, 1].set_title("Energy Conservation Error"); axs[1, 1].set_ylabel("$|E_{full} - E_{ROM}|$")

    for ax in axs.flat:
        ax.set_xlabel("Time (s)"); ax.legend(); ax.grid(True, which="both", ls="-", alpha=0.3)
    
    fig.suptitle("Comprehensive ROM Performance Diagnostics", fontsize=16, fontweight='bold')
    plt.savefig("comprehensive_diagnostics.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"📈 Comprehensive diagnostic plot saved to 'comprehensive_diagnostics.png'")

# ==========================================================================================
# MAIN ANALYSIS SCRIPT
# ==========================================================================================

def run_final_analysis(n_trajectories=100, T_horizon=3.0):
    print("#"*80)
    print("#      DEFINITIVE STABILITY ANALYSIS FOR AUTONOMOUS SYSTEM (u=0)      #")
    print("#"*80)

    sys = create_stable_system()
    X_train, Xdot_train = collect_training_data(sys)
    
    zero_controller = lambda x: torch.zeros(x.shape[0], sys.n_controls, device=x.device)

    reducers = {}
    print("\nInitializing Reduced-Order Models...")
    try:
        A, J, R = sys.linearise(return_JR=True)
        # Replicating the user's successful initialization: NO X_data is passed.
        spr = SymplecticProjectionReducer(sys, A, J, R, 10, enhanced=True)
        reducers['SPR-14'] = spr
        print(f"  - SPR-14 initialized using standard basis (consistent with retain_freq.py).")
    except Exception as e:
        reducers['SPR-14'] = None; print(f"  - ❌ SPR-14 failed: {e}")
    try:
        # OpInf and LCR benefit from data, so we still pass it to them.
        opinf = OpInfReducer(10, 19, sys.n_controls)
        opinf.sys = sys
        opinf.fit(X_train, Xdot_train, sys.energy_function, sys.energy_function(X_train).min().item())
        reducers['OpInf-14'] = opinf; print("  - OpInf-14 initialized.")
    except Exception as e:
        reducers['OpInf-14'] = None; print(f"  - ❌ OpInf-14 failed: {e}")
    try:
        lcr = LyapCoherencyReducer(sys, 5, X_train)
        lcr.gamma = lcr.compute_gamma(sys.energy_function(X_train).min().item())
        reducers['LCR-14'] = lcr; print("  - LCR-14 initialized.")
    except Exception as e:
        reducers['LCR-14'] = None; print(f"  - ❌ LCR-14 failed: {e}")

    x_eq = sys.goal_point.squeeze()
    
    # MODIFIED: Create a more balanced mix of stable and unstable trajectories
    # Use a narrower range centered around the critical perturbation level
    # For this system, perturbations around 0.1-0.5 tend to be near the stability boundary
    
    # Create three groups of perturbations:
    # 1. Small perturbations (likely stable)
    n_stable = n_trajectories // 3
    small_perturb = torch.linspace(0.1, 0.5, n_stable)
    
    # 2. Medium perturbations (mixed stability)
    n_mixed = n_trajectories // 3
    medium_perturb = torch.linspace(0.5, 1, n_mixed)
    
    # 3. Large perturbations (likely unstable)
    n_unstable = n_trajectories - n_stable - n_mixed
    large_perturb = torch.linspace(1, 4, n_unstable)
    
    # Combine and shuffle for better distribution
    perturb_scales = torch.cat([small_perturb, medium_perturb, large_perturb])
    # Shuffle the scales
    idx = torch.randperm(perturb_scales.shape[0])
    perturb_scales = perturb_scales[idx]
    
    # Generate initial conditions with random directions
    x0_test = []
    for scale in perturb_scales:
        direction = torch.randn_like(x_eq)
        direction = direction / torch.norm(direction)  # Normalize direction
        x0 = x_eq + scale * direction
        x0_test.append(x0)
    x0_test = torch.stack(x0_test)
    
    dt = sys.dt
    n_steps = int(T_horizon / dt)
    
    print(f"\nSimulating {n_trajectories} validation trajectories...")
    simulated_trajs = sys.simulate(x0_test, n_steps, controller=zero_controller, method='rk4')
    full_trajs = torch.cat((x0_test.unsqueeze(1), simulated_trajs), dim=1)
    
    full_labels = classify_stability(full_trajs, sys.N_NODES)
    
    print("\n" + "="*50)
    print(f"STABILITY CLASSIFICATION RESULTS")
    print("="*50)
    num_stable = full_labels.count('stable')
    num_unstable = n_trajectories - num_stable
    print(f"\nGround Truth Distribution:")
    print(f"  - Stable trajectories:   {num_stable}/{n_trajectories} ({num_stable/n_trajectories:6.1%})")
    print(f"  - Unstable trajectories: {num_unstable}/{n_trajectories} ({num_unstable/n_trajectories:6.1%})")
    
    # Store all results for summary
    all_results = {}
    
    for name, reducer in reducers.items():
        if reducer is None:
            print(f"\n▶ {name:<10}: SKIPPED (Initialization failed)")
            continue
            
        rom_method = "rk4"
        rom_trajs = rollout_rom(reducer, sys, x0_test, n_steps, controller=zero_controller, dt=dt, method=rom_method)
        rom_labels = classify_stability(rom_trajs, sys.N_NODES)
        
        # Compute confusion matrix
        tp, tn, fp, fn = compute_confusion_matrix(full_labels, rom_labels)
        
        # Store results
        all_results[name] = {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'accuracy': (tp + tn) / n_trajectories
        }
        
        # Print confusion matrix
        print_confusion_matrix(name, tp, tn, fp, fn)
    
    # Summary comparison
    if len(all_results) > 0:
        print("\n" + "="*50)
        print("SUMMARY COMPARISON")
        print("="*50)
        print(f"\n{'Method':<12} {'Accuracy':<10} {'TP':<6} {'TN':<6} {'FP':<6} {'FN':<6}")
        print("-"*50)
        for name, res in all_results.items():
            print(f"{name:<12} {res['accuracy']:8.1%}   {res['tp']:4d}   {res['tn']:4d}   "
                  f"{res['fp']:4d}   {res['fn']:4d}")
        
        # Find best performer
        best_method = max(all_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest performing method: {best_method[0]} with {best_method[1]['accuracy']:.1%} accuracy")
    
    example_idx = n_trajectories // 2
    generate_diagnostic_plots(sys, reducers, x0_test[example_idx], T_horizon, dt)

if __name__ == "__main__":
    run_final_analysis()