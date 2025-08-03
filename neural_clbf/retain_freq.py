"""
Full Dimension Test with Actual Trajectory Simulation
=====================================================

Tests all reducers with d=19 (full dimension) by simulating actual trajectories
over time and comparing full vs reduced model behavior.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from neural_clbf.systems import SwingEquationSystem
from neural_clbf.dimension_reduction.symplectic_projection import SymplecticProjectionReducer
from neural_clbf.dimension_reduction.opinf import OpInfReducer
from neural_clbf.dimension_reduction.lyap_coherency import LyapCoherencyReducer
from neural_clbf.eval.reduction_validation import rollout_rom, validate_reducer


def create_stable_system():
    """Create a stable system with reasonable parameters."""
    M = torch.tensor([1.4, 1.01, 1.1766666, 0.95333344, 0.8666667,
                      1.16, 0.88, 0.81, 1.1500001, 16.666668], dtype=torch.float32)
    
    # Use higher damping for stability
    D = torch.tensor([0.19666669, 0.28833333, 0.28833333, 0.28833333, 0.28833333,
                      0.28833333, 0.28833333, 0.28833333, 0.30366668, 0.30366668], 
                     dtype=torch.float32) * 5.0  # 5x damping
    
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
    
    delta_star = torch.tensor([-0.05420687, -0.07780334, -0.07351729, -0.05827823, -0.09359571,
                               -0.02447385, -0.00783582, 0.00259523, -0.0162409, -0.06477749],
                              dtype=torch.float32)
    
    params = dict(M=M, D=D, P=P, K=K)
    
    sys = SwingEquationSystem(params, dt=0.001)
    sys.delta_star = delta_star
    
    return sys, params


def collect_training_data(sys, params, n_samples=1000):
    """Collect training data for reducers."""
    theta_eq = sys.delta_star[0] - sys.delta_star[1:]
    omega_eq = torch.zeros(10)
    x_eq = torch.cat([theta_eq, omega_eq])
    
    # Small perturbations
    X = x_eq.unsqueeze(0).repeat(n_samples, 1)
    X += 0.01 * torch.randn_like(X)
    
    # Compute derivatives
    Xdot = []
    for i in range(n_samples):
        f = sys._f(X[i:i+1], params)
        Xdot.append(f.squeeze())
    
    return X, torch.stack(Xdot)

def simulate_trajectory(sys, x0, T, dt=0.01, controller=None):
    """Simulate a trajectory using the full system dynamics."""
    n_steps = int(T / dt)
    
    if controller is None:
        controller = lambda x: torch.zeros(x.shape[0], sys.n_controls)
    
    # Use system's simulate method
    traj = sys.simulate(
        x0.unsqueeze(0) if x0.dim() == 1 else x0,
        n_steps,
        controller=controller,
        controller_period=dt , # Ensure controller period is at least dt
        params=sys.nominal_params
    )
    
    return traj


def test_trajectory_preservation():
    """Test all reducers with trajectory simulation."""
    print("="*80)
    print("TRAJECTORY PRESERVATION TEST (d=19)")
    print("="*80)
    
    # Create system
    sys, params = create_stable_system()
    X_train, Xdot_train = collect_training_data(sys, params)
    
        
    # After collecting X_train
    print("\nChecking training data independence:")
    X_centered = X_train - X_train.mean(0)
    U, S, Vt = torch.linalg.svd(X_centered.T)

    # Show singular values
    print("Singular values:", S[:10])
    print("Normalized singular values:", S[:10] / S[0])

    # Find effective rank
    tol = 1e-8
    effective_rank = (S > tol * S[0]).sum().item()
    print(f"Effective rank (tol={tol}): {effective_rank}")

    # Check which modes are missing
    if effective_rank < 19:
        print("\nMissing modes (smallest singular values):")
        for i in range(effective_rank, min(19, len(S))):
            mode = Vt[i]
            print(f"  Mode {i}: singular value = {S[i]:.2e}")
            # Check if it's a frequency mode
            freq_component = mode[9:].norm()
            angle_component = mode[:9].norm()
            print(f"    Frequency content: {freq_component:.2%}, Angle content: {angle_component:.2%}")
    # Energy function
    V_fn = sys.energy_function
    V_min = V_fn(X_train).min().item()
    
    # Test parameters
    T_horizon = 2.0  # 5 second simulation
    dt = 0.01
    n_test_trajectories = 5
    
    print(f"\nSystem: 10-generator network")
    print(f"Full dimension: 19")
    print(f"Simulation horizon: {T_horizon}s")
    print(f"Number of test trajectories: {n_test_trajectories}")
    
    # Sample initial conditions
    theta_eq = sys.delta_star[0] - sys.delta_star[1:]
    omega_eq = torch.zeros(10)
    x_eq = torch.cat([theta_eq, omega_eq])
    
    x0_test = []
    for i in range(n_test_trajectories):
        # Different perturbation sizes
        perturbation_scale = 0.05 * (i + 1) / n_test_trajectories
        x0 = x_eq + perturbation_scale * torch.randn_like(x_eq)
        x0_test.append(x0)
    x0_test = torch.stack(x0_test)
    
    results = {}
    
    # Test 1: Symplectic Projection (d=18)
    print("\n1. Testing Symplectic Projection (d=18):")
    try:
        A, J, R = sys.linearise(return_JR=True)
        spr = SymplecticProjectionReducer(A, J, R, 18)
        spr.full_dim = 19
        
        # Simulate trajectories
        errors = []
        energy_errors = []
        
        for i, x0 in enumerate(x0_test):
            # Full system trajectory
            traj_full = simulate_trajectory(sys, x0, T_horizon, dt)
            
            # Reduced system trajectory
            traj_rom = rollout_rom(spr, sys, x0.unsqueeze(0), T_horizon, dt)
            
            # Ensure same length
            min_len = min(traj_full.shape[1], traj_rom.shape[1])
            traj_full = traj_full[:, :min_len]
            traj_rom = traj_rom[:, :min_len]
            
            # Trajectory error
            traj_error = (traj_full - traj_rom).norm(dim=-1).mean().item()
            errors.append(traj_error)
            
            # Energy error
            E_full = V_fn(traj_full.reshape(-1, 19)).reshape(traj_full.shape[0], -1)
            E_rom = V_fn(traj_rom.reshape(-1, 19)).reshape(traj_rom.shape[0], -1)
            energy_error = (E_full - E_rom).abs().mean().item()
            energy_errors.append(energy_error)
            
            print(f"   Trajectory {i+1}: error = {traj_error:.6e}, energy error = {energy_error:.6e}")
        
        results['SPR-18'] = {
            'mean_traj_error': np.mean(errors),
            'max_traj_error': np.max(errors),
            'mean_energy_error': np.mean(energy_errors),
            'gamma': spr.gamma
        }
        
    except Exception as e:
        print(f"   Failed: {e}")
        results['SPR-18'] = {'error': str(e)}
    
    # Test 2: OpInf with d=19
    print("\n2. Testing OpInf (d=19, full dimension):")
    try:
        opinf = OpInfReducer(19, 19, sys.n_controls)
        opinf.sys = sys
        opinf.fit(X_train, Xdot_train, V_fn, V_min)
        opinf.full_dim = 19
        
        errors = []
        energy_errors = []
        
        for i, x0 in enumerate(x0_test):
            # Full system trajectory
            traj_full = simulate_trajectory(sys, x0, T_horizon, dt)
            
            # Reduced system trajectory
            traj_rom = rollout_rom(opinf, sys, x0.unsqueeze(0), T_horizon, dt)
            
            # Ensure same length
            min_len = min(traj_full.shape[1], traj_rom.shape[1])
            traj_full = traj_full[:, :min_len]
            traj_rom = traj_rom[:, :min_len]
            
            # Trajectory error
            traj_error = (traj_full - traj_rom).norm(dim=-1).mean().item()
            errors.append(traj_error)
            
            # Energy error
            E_full = V_fn(traj_full.reshape(-1, 19)).reshape(traj_full.shape[0], -1)
            E_rom = V_fn(traj_rom.reshape(-1, 19)).reshape(traj_rom.shape[0], -1)
            energy_error = (E_full - E_rom).abs().mean().item()
            energy_errors.append(energy_error)
            
            print(f"   Trajectory {i+1}: error = {traj_error:.6e}, energy error = {energy_error:.6e}")
        
        results['OpInf-19'] = {
            'mean_traj_error': np.mean(errors),
            'max_traj_error': np.max(errors),
            'mean_energy_error': np.mean(energy_errors),
            'gamma': opinf.gamma
        }
        
    except Exception as e:
        print(f"   Failed: {e}")
        results['OpInf-19'] = {'error': str(e)}
    
    # Test 3: Lyapunov Coherency (k=9)
    print("\n3. Testing Lyapunov Coherency (k=9 groups → d=18):")
    try:
        lcr = LyapCoherencyReducer(sys, 9, X_train)
        lcr.full_dim = 19
        lcr.gamma = lcr.compute_gamma(V_min)
        
        errors = []
        energy_errors = []
        
        for i, x0 in enumerate(x0_test):
            # Full system trajectory
            traj_full = simulate_trajectory(sys, x0, T_horizon, dt)
            
            # Reduced system trajectory
            traj_rom = rollout_rom(lcr, sys, x0.unsqueeze(0), T_horizon, dt)
            
            # Ensure same length
            min_len = min(traj_full.shape[1], traj_rom.shape[1])
            traj_full = traj_full[:, :min_len]
            traj_rom = traj_rom[:, :min_len]
            
            # Trajectory error
            traj_error = (traj_full - traj_rom).norm(dim=-1).mean().item()
            errors.append(traj_error)
            
            # Energy error
            E_full = V_fn(traj_full.reshape(-1, 19)).reshape(traj_full.shape[0], -1)
            E_rom = V_fn(traj_rom.reshape(-1, 19)).reshape(traj_rom.shape[0], -1)
            energy_error = (E_full - E_rom).abs().mean().item()
            energy_errors.append(energy_error)
            
            print(f"   Trajectory {i+1}: error = {traj_error:.6e}, energy error = {energy_error:.6e}")
        
        results['LCR-18'] = {
            'mean_traj_error': np.mean(errors),
            'max_traj_error': np.max(errors),
            'mean_energy_error': np.mean(energy_errors),
            'gamma': lcr.gamma
        }
        
    except Exception as e:
        print(f"   Failed: {e}")
        results['LCR-18'] = {'error': str(e)}
    
    # Comprehensive validation using validate_reducer
    print("\n" + "="*80)
    print("COMPREHENSIVE VALIDATION (50 trajectories)")
    print("="*80)
    
    validation_results = {}
    
    for name, reducer in [
        ('SPR-18', spr if 'spr' in locals() else None),
        ('OpInf-19', opinf if 'opinf' in locals() else None),
        ('LCR-18', lcr if 'lcr' in locals() else None)
    ]:
        if reducer is None:
            continue
            
        print(f"\nValidating {name}:")
        try:


            def sample_near_equilibrium(n_samples):
                x_eq = sys.goal_point.squeeze()
                x0 = x_eq.unsqueeze(0).repeat(n_samples, 1)
                x0 += 0.1 * torch.randn_like(x0)  # 10% perturbation
                return x0

            # Monkey-patch the sampling
            original_sample = sys.sample_state_space
            sys.sample_state_space = sample_near_equilibrium

            val_metrics = validate_reducer(
                sys, reducer, 
                n_rollouts=50,
                horizon=2.0,
                dt=0.01,
                input_mode="zero"
            )

            # Restore original
            sys.sample_state_space = original_sample
            validation_results[name] = val_metrics
            print(f"  Mean error: {val_metrics['mean_error']:.6e}")
            print(f"  Max error: {val_metrics['max_error']:.6e}")
            print(f"  Relative error: {val_metrics['relative_error']:.3%}")
            print(f"  Energy error: {val_metrics['energy_error']:.6e}")
            print(f"  Success rate: {val_metrics['success_rate']:.1%}")
            
        except Exception as e:
            print(f"  Validation failed: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY - TRAJECTORY PRESERVATION")
    print("="*80)
    print("\nExpected behavior for full/near-full dimension:")
    print("- Trajectory errors should be < 1e-6")
    print("- Energy errors should be < 1e-8") 
    print("- Success rate should be 100%")
    
    print(f"\n{'Method':<10} {'Mean Traj Err':<15} {'Max Traj Err':<15} {'Energy Err':<15} {'Success Rate':<12}")
    print("-"*75)
    
    for method, res in validation_results.items():
        if 'error' not in res:
            mean_err = f"{res['mean_error']:.2e}"
            max_err = f"{res['max_error']:.2e}"
            energy_err = f"{res['energy_error']:.2e}"
            success = f"{res['success_rate']:.1%}"
            print(f"{method:<10} {mean_err:<15} {max_err:<15} {energy_err:<15} {success:<12}")
    
    # Plot some example trajectories
    print("\n" + "="*80)
    print("PLOTTING EXAMPLE TRAJECTORIES")
    print("="*80)
    
    # Pick one initial condition for plotting
    x0_plot = x0_test[2]  # Medium perturbation
    
    plt.figure(figsize=(15, 10))
    
    for idx, (name, reducer) in enumerate([
        ('Full System', None),
        ('SPR-18', spr if 'spr' in locals() else None),
        ('OpInf-19', opinf if 'opinf' in locals() else None),
        ('LCR-18', lcr if 'lcr' in locals() else None)
    ]):
        if name != 'Full System' and reducer is None:
            continue
            
        # Simulate
        if name == 'Full System':
            traj = simulate_trajectory(sys, x0_plot, T_horizon, dt)
        else:
            traj = rollout_rom(reducer, sys, x0_plot.unsqueeze(0), T_horizon, dt)
        
        t = np.arange(traj.shape[1]) * dt
        
        # Plot angles (first 3)
        plt.subplot(3, 2, 1)
        for i in range(3):
            plt.plot(t, traj[0, :, i].cpu().numpy(), 
                    label=f'{name} θ_{i+2}' if idx == 0 else None,
                    linestyle='-' if name == 'Full System' else '--')
        plt.ylabel('Angle (rad)')
        plt.title('Rotor Angles (first 3)')
        if idx == 0:
            plt.legend()
        plt.grid(True)
        
        # Plot frequencies (first 3)
        plt.subplot(3, 2, 2)
        for i in range(3):
            plt.plot(t, traj[0, :, 9+i].cpu().numpy(),
                    label=f'{name} ω_{i+1}' if idx == 0 else None,
                    linestyle='-' if name == 'Full System' else '--')
        plt.ylabel('Frequency (rad/s)')
        plt.title('Rotor Frequencies (first 3)')
        if idx == 0:
            plt.legend()
        plt.grid(True)
        
        # Plot energy
        plt.subplot(3, 2, 3)
        E = V_fn(traj.reshape(-1, 19)).reshape(-1).cpu().numpy()
        plt.plot(t, E, label=name, 
                linestyle='-' if name == 'Full System' else '--')
        plt.ylabel('Energy')
        plt.xlabel('Time (s)')
        plt.title('Total System Energy')
        plt.legend()
        plt.grid(True)
        
        # Plot energy error (if not full system)
        if name != 'Full System':
            plt.subplot(3, 2, 4)
            traj_full = simulate_trajectory(sys, x0_plot, T_horizon, dt)
            min_len = min(traj_full.shape[1], traj.shape[1])
            E_full = V_fn(traj_full[:, :min_len].reshape(-1, 19)).reshape(-1).cpu().numpy()
            E_rom = V_fn(traj[:, :min_len].reshape(-1, 19)).reshape(-1).cpu().numpy()
            plt.semilogy(t[:min_len], np.abs(E_full - E_rom), label=name)
            plt.ylabel('|Energy Error|')
            plt.xlabel('Time (s)')
            plt.title('Energy Conservation Error')
            plt.legend()
            plt.grid(True)
        
        # Plot state error norm
        if name != 'Full System':
            plt.subplot(3, 2, 5)
            traj_full = simulate_trajectory(sys, x0_plot, T_horizon, dt)
            min_len = min(traj_full.shape[1], traj.shape[1])
            error = (traj_full[:, :min_len] - traj[:, :min_len]).norm(dim=-1).squeeze().cpu().numpy()
            plt.semilogy(t[:min_len], error, label=name)
            plt.ylabel('||x_full - x_rom||')
            plt.xlabel('Time (s)')
            plt.title('State Reconstruction Error')
            plt.legend()
            plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('full_dimension_trajectories.png', dpi=150)
    print("\nTrajectory plots saved to 'full_dimension_trajectories.png'")

    # Add this import at the top of retain_freq.py
    from neural_clbf.reducer_diagnostic import run_full_diagnostic

    # Then at the end of test_trajectory_preservation(), add:
    print("\n" + "="*80)
    print("RUNNING DETAILED DIAGNOSTIC")
    print("="*80)

    # Run diagnostic on the reducers we created
    run_full_diagnostic(
        sys, 
        spr=spr if 'spr' in locals() else None,
        opinf=opinf if 'opinf' in locals() else None,
        lcr=lcr if 'lcr' in locals() else None,
        X_train=X_train
    )
    
    return results, validation_results


if __name__ == "__main__":
    results, validation_results = test_trajectory_preservation()
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    all_passed = True
    for method, res in validation_results.items():
        if 'error' not in res:
            # Check if errors are small enough
            passed = (res['mean_error'] < 1e-4 and 
                     res['energy_error'] < 1e-4 and
                     res['success_rate'] > 0.95)
            
            status = "PASS" if passed else "FAIL"
            print(f"{method}: {status}")
            
            if not passed:
                all_passed = False
                print(f"  - Mean error: {res['mean_error']:.6e} (threshold: 1e-4)")
                print(f"  - Energy error: {res['energy_error']:.6e} (threshold: 1e-4)")
                print(f"  - Success rate: {res['success_rate']:.1%} (threshold: 95%)")
    
    if all_passed:
        print("\nAll reducers successfully preserve trajectories at full/near-full dimension!")
    else:
        print("\nSome reducers failed to preserve trajectories adequately.")
        print("This may indicate numerical issues or implementation problems.")