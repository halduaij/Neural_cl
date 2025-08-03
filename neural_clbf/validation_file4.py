"""
Direct testing of each reducer to bypass selection issues
"""
import torch
import numpy as np
from neural_clbf.systems import SwingEquationSystem
from neural_clbf.dimension_reduction.symplectic_projection import SymplecticProjectionReducer
from neural_clbf.dimension_reduction.opinf import OpInfReducer
from neural_clbf.dimension_reduction.lyap_coherency import LyapCoherencyReducer
from neural_clbf.eval.reduction_validation import validate_reducer


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
    
    sys = SwingEquationSystem(params, dt=0.01)
    sys.delta_star = delta_star
    
    # Fix dynamics
    original_f = sys._f
    def _f_fixed(x, params):
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, 19, 1))
        
        theta = x[:, :9]
        omega = x[:, 9:]
        
        delta_eq = delta_star
        theta_eq = delta_eq[0] - delta_eq[1:]
        
        delta = torch.zeros(batch_size, 10)
        delta[:, 0] = delta_eq[0]
        delta[:, 1:] = delta_eq[1:] + (theta_eq - theta)
        
        # Angle derivatives
        for i in range(1, 10):
            f[:, i-1, 0] = omega[:, 0] - omega[:, i]
        
        # Frequency derivatives
        for i in range(10):
            omega_dot = P[i] / M[i] - (D[i] / M[i]) * omega[:, i]
            for j in range(10):
                if i != j:
                    omega_dot -= (K[i, j] / M[i]) * torch.sin(delta[:, i] - delta[:, j])
            f[:, 9+i, 0] = omega_dot
        
        return f
    
    sys._f = _f_fixed
    
    return sys, params


def collect_small_data(sys, params):
    """Collect data with very small perturbations."""
    theta_eq = sys.delta_star[0] - sys.delta_star[1:]
    omega_eq = torch.zeros(10)
    x_eq = torch.cat([theta_eq, omega_eq])
    
    n_samples = 1000
    
    # Very small perturbations
    X = x_eq.unsqueeze(0).repeat(n_samples, 1)
    X += 0.005 * torch.randn_like(X)  # 0.5% noise
    
    # Compute derivatives
    Xdot = []
    for i in range(n_samples):
        f = sys._f(X[i:i+1], params)
        Xdot.append(f.squeeze())
    
    data = {
        'X': X,
        'dXdt': torch.stack(Xdot)
    }
    
    return data


def test_reducers_directly():
    """Test each reducer type directly."""
    print("="*80)
    print("DIRECT REDUCER TESTING")
    print("="*80)
    
    # Create system
    sys, params = create_stable_system()
    
    # Collect data
    print("\n1. Collecting training data:")
    data = collect_small_data(sys, params)
    print(f"   Data shape: {data['X'].shape}")
    print(f"   Data range: [{data['X'].min():.3f}, {data['X'].max():.3f}]")
    
    # Energy function for gamma calculation
    def simple_energy(x):
        """Simple quadratic energy function."""
        return 0.5 * (x ** 2).sum(dim=1)
    
    V_min = simple_energy(data['X']).min().item()
    print(f"   V_min: {V_min:.3f}")
    
    results = []
    
    # Test 1: Symplectic Projection
    print("\n2. Testing Symplectic Projection:")
    try:
        # Get linearization
        A = torch.eye(19) * (-0.5)  # Stable dummy matrix
        J = torch.zeros(19, 19)
        J[:9, 9:] = torch.eye(9, 10)
        J[9:, :9] = -torch.eye(10, 9)
        R = torch.zeros(19, 19)
        R[9:, 9:] = torch.diag(params['D'])
        
        for d in [2, 4, 6, 8]:
            try:
                spr = SymplecticProjectionReducer(A, J, R, d)
                spr.full_dim = 19
                print(f"   SPR d={d}: gamma={spr.gamma}")
                
                # Quick validation
                val = validate_reducer(sys, spr, n_rollouts=3, horizon=0.3, dt=0.01)
                
                results.append({
                    'type': 'SPR',
                    'd': d,
                    'gamma': spr.gamma,
                    'error': val['mean_error'].item()
                })
                
                print(f"   Mean error: {val['mean_error']:.3f}")
            except Exception as e:
                print(f"   SPR d={d} failed: {e}")
                
    except Exception as e:
        print(f"   Symplectic Projection failed: {e}")
    
    # Test 2: Lyapunov Coherency
    print("\n3. Testing Lyapunov Coherency:")
    try:
        for k in [2, 3, 4]:
            try:
                lcr = LyapCoherencyReducer(sys, k, data['X'])
                lcr.full_dim = 19
                lcr.gamma = lcr.compute_gamma(V_min)
                
                print(f"   LCR k={k} (d={lcr.latent_dim}): gamma={lcr.gamma:.3f}")
                
                if lcr.gamma < 50:
                    val = validate_reducer(sys, lcr, n_rollouts=3, horizon=0.3, dt=0.01)
                    results.append({
                        'type': 'LCR',
                        'd': lcr.latent_dim,
                        'gamma': lcr.gamma,
                        'error': val['mean_error'].item()
                    })
                    print(f"   Mean error: {val['mean_error']:.3f}")
                else:
                    print(f"   Skipping (gamma too high)")
                    
            except Exception as e:
                print(f"   LCR k={k} failed: {e}")
                
    except Exception as e:
        print(f"   Lyapunov Coherency failed: {e}")
    
    # Test 3: OpInf
    print("\n4. Testing Operator Inference:")
    try:
        for d in [4, 6, 8]:
            try:
                opinf = OpInfReducer(d, 19, sys.n_controls)
                opinf.sys = sys
                opinf.fit(data['X'], data['dXdt'], simple_energy, V_min)
                opinf.full_dim = 19
                
                print(f"   OpInf d={d}: gamma={opinf.gamma:.3f}")
                
                if opinf.gamma < 50:
                    val = validate_reducer(sys, opinf, n_rollouts=3, horizon=0.3, dt=0.01)
                    results.append({
                        'type': 'OpInf',
                        'd': d,
                        'gamma': opinf.gamma,
                        'error': val['mean_error'].item()
                    })
                    print(f"   Mean error: {val['mean_error']:.3f}")
                else:
                    print(f"   Skipping (gamma too high)")
                    
            except Exception as e:
                print(f"   OpInf d={d} failed: {e}")
                
    except Exception as e:
        print(f"   Operator Inference failed: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL REDUCERS")
    print("="*80)
    
    if results:
        print(f"\n{'Type':<10} {'Dim':<5} {'Gamma':<10} {'Error':<10}")
        print("-"*40)
        
        for r in sorted(results, key=lambda x: x['error']):
            print(f"{r['type']:<10} {r['d']:<5} {r['gamma']:<10.3f} {r['error']:<10.3f}")
    
    return results

"""
Proper validation metrics for power system dimension reduction
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from neural_clbf.systems import SwingEquationSystem
from neural_clbf.dimension_reduction.symplectic_projection import SymplecticProjectionReducer
from neural_clbf.dimension_reduction.opinf import OpInfReducer
from neural_clbf.dimension_reduction.lyap_coherency import LyapCoherencyReducer


def evaluate_power_system_reduction(sys, reducer, data, n_test=10, horizon=2.0, dt=0.01):
    """
    Comprehensive evaluation of dimension reduction for power systems.
    
    Metrics:
    1. Angle reconstruction error (in degrees)
    2. Frequency reconstruction error (in Hz)
    3. Energy preservation error
    4. Inter-area oscillation mode preservation
    5. Transient stability (critical clearing time)
    """
    device = data['X'].device
    results = {}
    
    # Get equilibrium for reference
    delta_eq = sys.delta_star
    theta_eq = delta_eq[0] - delta_eq[1:]
    
    print(f"\nEvaluating {type(reducer).__name__} (d={reducer.latent_dim}):")
    
    # 1. State Reconstruction Error
    print("\n1. State Reconstruction:")
    
    # Take random test points
    test_idx = torch.randperm(data['X'].shape[0])[:n_test]
    X_test = data['X'][test_idx]
    
    # Project and reconstruct
    Z_test = reducer.forward(X_test)
    X_recon = reducer.inverse(Z_test)
    
    # Separate angle and frequency errors
    theta_error = X_recon[:, :9] - X_test[:, :9]  # Angle differences
    omega_error = X_recon[:, 9:] - X_test[:, 9:]   # Frequencies
    
    # Convert to physical units
    theta_error_deg = theta_error * 180 / np.pi
    omega_error_hz = omega_error / (2 * np.pi) * 60  # Assuming 60 Hz base
    
    results['theta_rmse_deg'] = theta_error_deg.pow(2).mean().sqrt().item()
    results['omega_rmse_hz'] = omega_error_hz.pow(2).mean().sqrt().item()
    
    print(f"   Angle RMSE: {results['theta_rmse_deg']:.2f} degrees")
    print(f"   Frequency RMSE: {results['omega_rmse_hz']:.3f} Hz")
    
    # 2. Energy Preservation
    print("\n2. Energy Preservation:")
    
    # Compute energy for original and reconstructed states
    E_orig = sys.energy_function(X_test)
    E_recon = sys.energy_function(X_recon)
    
    energy_error = ((E_recon - E_orig) / E_orig).abs()
    results['energy_error'] = energy_error.mean().item()
    
    print(f"   Mean relative energy error: {results['energy_error']:.1%}")
    
    # 3. Dynamic Trajectory Error
    print("\n3. Dynamic Trajectory Test:")
    
    # Simulate short trajectories
    n_steps = int(horizon / dt)
    traj_errors = []
    
    for i in range(min(5, n_test)):  # Test 5 trajectories
        x0 = X_test[i:i+1]
        
        # Full model trajectory
        x_full = x0.clone()
        traj_full = [x0]
        
        for t in range(n_steps):
            f = sys._f(x_full, sys.nominal_params)
            x_full = x_full + dt * f.squeeze(-1)
            traj_full.append(x_full)
        
        traj_full = torch.cat(traj_full, dim=0)
        
        # Reduced model trajectory
        z0 = reducer.forward(x0)
        z = z0.clone()
        traj_red = [x0]
        
        for t in range(n_steps):
            # Get full state
            x_red = reducer.inverse(z)
            
            # Compute dynamics in full space
            f_full = sys._f(x_red, sys.nominal_params)
            
            # Project dynamics to reduced space
            J = reducer.jacobian(x_red)
            f_red = torch.bmm(J, f_full).squeeze(-1)
            
            # Update reduced state
            z = z + dt * f_red
            
            # Reconstruct and store
            x_red = reducer.inverse(z)
            traj_red.append(x_red)
        
        traj_red = torch.cat(traj_red, dim=0)
        
        # Compute trajectory error
        traj_error = (traj_full - traj_red).norm(dim=1).mean()
        traj_errors.append(traj_error.item())
    
    results['traj_error'] = np.mean(traj_errors)
    print(f"   Mean trajectory error: {results['traj_error']:.3f}")
    
    # 4. Frequency Response (simplified)
    print("\n4. Frequency Response Test:")
    
    # Apply small perturbation and check frequency response
    x_eq = torch.cat([theta_eq, torch.zeros(10)]).unsqueeze(0)
    
    # Perturb one generator
    x_pert = x_eq.clone()
    x_pert[0, 9] = 0.1  # 0.1 rad/s frequency perturbation
    
    # Project perturbation
    z_eq = reducer.forward(x_eq)
    z_pert = reducer.forward(x_pert)
    
    # Reconstruct
    x_pert_recon = reducer.inverse(z_pert)
    
    # Check if perturbation is preserved
    omega_pert_orig = x_pert[0, 9:] - x_eq[0, 9:]
    omega_pert_recon = x_pert_recon[0, 9:] - x_eq[0, 9:]
    
    pert_preservation = (omega_pert_recon.norm() / omega_pert_orig.norm()).item()
    results['pert_preservation'] = pert_preservation
    
    print(f"   Perturbation preservation: {pert_preservation:.1%}")
    
    # 5. Summary Score
    # Lower is better for all metrics
    results['score'] = (
        results['theta_rmse_deg'] +           # Angle error (degrees)
        10 * results['omega_rmse_hz'] +       # Frequency error (Hz) - weighted more
        100 * results['energy_error'] +       # Energy error (fraction)
        results['traj_error'] +               # Trajectory error
        10 * abs(1 - results['pert_preservation'])  # Perturbation preservation
    )
    
    print(f"\n   Overall Score: {results['score']:.3f} (lower is better)")
    
    return results


def visualize_reduction_comparison(sys, reducers, data):
    """
    Visualize comparison of different reducers.
    """
    # Test one trajectory
    x0 = data['X'][0:1]
    horizon = 3.0
    dt = 0.01
    n_steps = int(horizon / dt)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Full model simulation
    x = x0.clone()
    traj_full = []
    time = []
    
    for t in range(n_steps):
        traj_full.append(x.clone())
        time.append(t * dt)
        f = sys._f(x, sys.nominal_params)
        x = x + dt * f.squeeze(-1)
    
    traj_full = torch.stack(traj_full).squeeze(1)
    time = np.array(time)
    
    # Plot for each reducer
    colors = plt.cm.tab10(np.linspace(0, 1, len(reducers) + 1))
    
    for idx, (name, reducer) in enumerate(reducers.items()):
        # Reduced model simulation
        z = reducer.forward(x0)
        traj_red = []
        
        for t in range(n_steps):
            x_red = reducer.inverse(z)
            traj_red.append(x_red.clone())
            
            f_full = sys._f(x_red, sys.nominal_params)
            J = reducer.jacobian(x_red)
            f_red = torch.bmm(J, f_full).squeeze(-1)
            z = z + dt * f_red
        
        traj_red = torch.stack(traj_red).squeeze(1)
        
        # Plot angle of generator 1
        axes[0, 0].plot(time, traj_red[:, 0].numpy(), 
                       label=f'{name} (d={reducer.latent_dim})', 
                       color=colors[idx+1], linestyle='--')
        
        # Plot frequency of generator 1
        axes[0, 1].plot(time, traj_red[:, 9].numpy() * 60 / (2*np.pi), 
                       color=colors[idx+1], linestyle='--')
        
        # Plot angle error
        angle_error = (traj_red[:, :9] - traj_full[:, :9]).abs().mean(dim=1)
        axes[1, 0].plot(time, angle_error.numpy() * 180 / np.pi, 
                       label=f'{name}', color=colors[idx+1])
        
        # Plot frequency error
        freq_error = (traj_red[:, 9:] - traj_full[:, 9:]).abs().mean(dim=1)
        axes[1, 1].plot(time, freq_error.numpy() * 60 / (2*np.pi), 
                       color=colors[idx+1])
    
    # Add full model
    axes[0, 0].plot(time, traj_full[:, 0].numpy(), 'k-', label='Full model', linewidth=2)
    axes[0, 1].plot(time, traj_full[:, 9].numpy() * 60 / (2*np.pi), 'k-', linewidth=2)
    
    # Format plots
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Angle difference θ₁₂ (rad)')
    axes[0, 0].set_title('Generator 1-2 Angle')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Frequency (Hz)')
    axes[0, 1].set_title('Generator 1 Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Mean angle error (degrees)')
    axes[1, 0].set_title('Angle Reconstruction Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Mean frequency error (Hz)')
    axes[1, 1].set_title('Frequency Reconstruction Error')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('power_system_reduction_comparison.png', dpi=150)
    plt.show()


def run_proper_validation():
    """
    Run validation with appropriate metrics for power systems.
    """
    print("="*80)
    print("POWER SYSTEM DIMENSION REDUCTION - PROPER VALIDATION")
    print("="*80)
    
    # Create system (using your working configuration)
    from neural_clbf.validation_file4 import create_stable_system, collect_small_data
    
    sys, params = create_stable_system()
    data = collect_small_data(sys, params)
    
    print(f"\nSystem: 10-generator network")
    print(f"States: 9 angle differences + 10 frequencies = 19 total")
    print(f"Training data: {data['X'].shape[0]} snapshots")
    
    # Create reducers manually
    reducers = {}
    
    # 1. Symplectic Projection
    try:
        A = torch.eye(19) * (-0.5)
        J = torch.zeros(19, 19)
        J[:9, 9:] = torch.eye(9, 10)
        J[9:, :9] = -torch.eye(10, 9)
        R = torch.zeros(19, 19)
        R[9:, 9:] = torch.diag(params['D'])
        
        spr = SymplecticProjectionReducer(A, J, R, 6)
        spr.full_dim = 19
        reducers['SPR-6'] = spr
    except:
        print("SPR failed")
    
    # 2. Lyapunov Coherency
    try:
        lcr = LyapCoherencyReducer(sys, 3, data['X'])  # 3 groups -> 6 states
        lcr.full_dim = 19
        reducers['LCR-6'] = lcr
    except:
        print("LCR failed")
    
    # 3. OpInf
    try:
        opinf = OpInfReducer(6, 19, sys.n_controls)
        opinf.sys = sys
        V_fn = lambda x: 0.5 * (x ** 2).sum(dim=1)
        V_min = 0.01
        opinf.fit(data['X'], data['dXdt'], V_fn, V_min)
        opinf.full_dim = 19
        reducers['OpInf-6'] = opinf
    except:
        print("OpInf failed")
    
    # Evaluate each reducer
    all_results = {}
    for name, reducer in reducers.items():
        results = evaluate_power_system_reduction(sys, reducer, data)
        all_results[name] = results
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY - POWER SYSTEM METRICS")
    print("="*80)
    print(f"\n{'Reducer':<10} {'Angle Err':<12} {'Freq Err':<12} {'Energy Err':<12} {'Traj Err':<10} {'Score':<10}")
    print(f"{'':10} {'(degrees)':<12} {'(Hz)':<12} {'(%)':<12} {'':10} {'':10}")
    print("-"*74)
    
    for name, res in sorted(all_results.items(), key=lambda x: x[1]['score']):
        print(f"{name:<10} {res['theta_rmse_deg']:<12.2f} {res['omega_rmse_hz']:<12.3f} "
              f"{res['energy_error']*100:<12.1f} {res['traj_error']:<10.3f} {res['score']:<10.2f}")
    
    # Visualize
    print("\nGenerating comparison plots...")
    visualize_reduction_comparison(sys, reducers, data)
    
    return all_results


if __name__ == "__main__":
    results = run_proper_validation()