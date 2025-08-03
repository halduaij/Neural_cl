"""
Linearized System Analysis - Understanding Baseline Behavior
===========================================================

Analyzes the linearized system without any dimension reduction to understand
inherent stability properties.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


def create_power_system():
    """Create the power system"""
    from neural_clbf.systems import SwingEquationSystem
    
    M = torch.tensor([1.4, 1.01, 1.1766666, 0.95333344, 0.8666667,
                      1.16, 0.88, 0.81, 1.1500001, 16.666668], dtype=torch.float32)
    
    D = torch.tensor([0.19666669, 0.28833333, 0.28833333, 0.28833333, 0.28833333,
                      0.28833333, 0.28833333, 0.28833333, 0.30366668, 0.30366668], 
                     dtype=torch.float32) * 3.0
    
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
    
    # Fixed dynamics
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
        
        for i in range(1, 10):
            f[:, i-1, 0] = omega[:, 0] - omega[:, i]
        
        for i in range(10):
            omega_dot = P[i] / M[i] - (D[i] / M[i]) * omega[:, i]
            for j in range(10):
                if i != j:
                    omega_dot -= (K[i, j] / M[i]) * torch.sin(delta[:, i] - delta[:, j])
            f[:, 9+i, 0] = omega_dot
        
        return f
    
    sys._f = _f_fixed
    
    # Get equilibrium
    theta_eq = sys.delta_star[0] - sys.delta_star[1:]
    omega_eq = torch.zeros(10)
    x_eq = torch.cat([theta_eq, omega_eq])
    
    return sys, x_eq


def analyze_linearization(sys, x_eq):
    """Analyze the linearized system dynamics"""
    print("="*80)
    print("LINEARIZED SYSTEM ANALYSIS")
    print("="*80)
    
    # Get linearization at equilibrium
    print("\n1. Computing linearization at equilibrium...")
    
    # Method 1: Using system's linearise method
    try:
        A_torch = sys.compute_A_matrix(sys.nominal_params)
        A = A_torch
        print(f"   Linearization successful using system method")
    except:
        # Method 2: Manual Jacobian computation
        print("   Computing Jacobian manually...")
        x_eq_grad = x_eq.clone().requires_grad_(True)
        
        def dynamics(x):
            return sys._f(x.unsqueeze(0), sys.nominal_params).squeeze()
        
        A_torch = torch.autograd.functional.jacobian(dynamics, x_eq_grad)
        A = A_torch.detach().numpy()
    
    print(f"   System matrix A shape: {A.shape}")
    
    # Eigenvalue analysis
    print("\n2. Eigenvalue Analysis:")
    eigvals = np.linalg.eigvals(A)
    
    # Sort by real part
    idx = np.argsort(eigvals.real)[::-1]
    eigvals = eigvals[idx]
    
    print(f"   Number of eigenvalues: {len(eigvals)}")
    print(f"   Max real part: {eigvals.real.max():.6f}")
    print(f"   Min real part: {eigvals.real.min():.6f}")
    
    # Check stability
    unstable_count = np.sum(eigvals.real > 1e-10)
    if unstable_count > 0:
        print(f"   ⚠️  UNSTABLE: {unstable_count} eigenvalues with positive real parts")
    else:
        print(f"   ✓  STABLE: All eigenvalues have negative real parts")
    
    # Print dominant eigenvalues
    print("\n   Dominant eigenvalues (sorted by real part):")
    for i in range(min(10, len(eigvals))):
        print(f"   λ_{i+1} = {eigvals[i].real:.6f} + {eigvals[i].imag:.6f}j")
    
    return A, eigvals


def simulate_systems(sys, x_eq, A, T_sim=5.0, dt=0.01):
    """Simulate nonlinear, linearized, and various reducers"""
    n_steps = int(T_sim / dt)
    time_steps = np.arange(n_steps + 1) * dt
    
    # Initial perturbation
    x0_pert = 0.02 * torch.randn_like(x_eq)
    x0 = x_eq + x0_pert
    
    # 1. Nonlinear simulation
    print("\n3. Simulating nonlinear system...")
    x_nonlinear = x0.clone()
    traj_nonlinear = [x_nonlinear.numpy()]
    
    for _ in range(n_steps):
        f = sys._f(x_nonlinear.unsqueeze(0), sys.nominal_params).squeeze()
        x_nonlinear = x_nonlinear + dt * f
        traj_nonlinear.append(x_nonlinear.numpy())
    
    traj_nonlinear = np.array(traj_nonlinear)
    
    # 2. Linear simulation (using matrix exponential for accuracy)
    print("   Simulating linearized system...")
    x0_pert_np = x0_pert.numpy()
    traj_linear_pert = []
    
    for i in range(n_steps + 1):
        t = i * dt
        x_pert = expm(A * t) @ x0_pert_np
        traj_linear_pert.append(x_pert)
    
    traj_linear_pert = np.array(traj_linear_pert)
    # Add back equilibrium
    traj_linear = traj_linear_pert + x_eq.numpy()
    
    # 3. Also simulate using Euler integration for comparison
    x_linear_euler = x0.clone()
    traj_linear_euler = [x_linear_euler.numpy()]
    
    for _ in range(n_steps):
        x_pert_euler = x_linear_euler - x_eq
        dx = torch.tensor(A @ x_pert_euler.numpy())
        x_linear_euler = x_linear_euler + dt * dx
        traj_linear_euler.append(x_linear_euler.numpy())
    
    traj_linear_euler = np.array(traj_linear_euler)
    
    return time_steps, traj_nonlinear, traj_linear, traj_linear_euler


def plot_results(time_steps, traj_nonlinear, traj_linear, traj_linear_euler, eigvals):
    """Create comprehensive plots"""
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Eigenvalue plot
    ax1 = plt.subplot(3, 3, 1)
    plt.scatter(eigvals.real, eigvals.imag, c='red', s=100, alpha=0.7, edgecolors='black')
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')
    plt.title('Linearized System Eigenvalues')
    plt.grid(True, alpha=0.3)
    
    # Add zoom inset for near-zero eigenvalues
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax1, width="40%", height="40%", loc='upper left')
    axins.scatter(eigvals.real, eigvals.imag, c='red', s=50, alpha=0.7)
    axins.set_xlim(-0.5, 0.5)
    axins.set_ylim(-2, 2)
    axins.grid(True, alpha=0.3)
    
    # 2. Angle trajectories
    ax2 = plt.subplot(3, 3, 2)
    angle_idx = 0  # First angle difference
    plt.plot(time_steps, traj_nonlinear[:, angle_idx] * 180/np.pi, 'k-', 
             label='Nonlinear', linewidth=2)
    plt.plot(time_steps, traj_linear[:, angle_idx] * 180/np.pi, 'r--', 
             label='Linear (matrix exp)', linewidth=1.5)
    plt.plot(time_steps, traj_linear_euler[:, angle_idx] * 180/np.pi, 'b:', 
             label='Linear (Euler)', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle 1-2 (degrees)')
    plt.title('Angle Trajectory Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Frequency trajectories
    ax3 = plt.subplot(3, 3, 3)
    freq_idx = 9  # First frequency
    plt.plot(time_steps, traj_nonlinear[:, freq_idx] * 60/(2*np.pi), 'k-', 
             label='Nonlinear', linewidth=2)
    plt.plot(time_steps, traj_linear[:, freq_idx] * 60/(2*np.pi), 'r--', 
             label='Linear (matrix exp)', linewidth=1.5)
    plt.plot(time_steps, traj_linear_euler[:, freq_idx] * 60/(2*np.pi), 'b:', 
             label='Linear (Euler)', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency 1 (Hz)')
    plt.title('Frequency Trajectory Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. State norm (stability indicator)
    ax4 = plt.subplot(3, 3, 4)
    norm_nonlinear = np.linalg.norm(traj_nonlinear - traj_nonlinear[0], axis=1)
    norm_linear = np.linalg.norm(traj_linear - traj_linear[0], axis=1)
    norm_linear_euler = np.linalg.norm(traj_linear_euler - traj_linear_euler[0], axis=1)
    
    plt.semilogy(time_steps, norm_nonlinear + 1e-10, 'k-', label='Nonlinear', linewidth=2)
    plt.semilogy(time_steps, norm_linear + 1e-10, 'r--', label='Linear (matrix exp)', linewidth=1.5)
    plt.semilogy(time_steps, norm_linear_euler + 1e-10, 'b:', label='Linear (Euler)', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('||x(t) - x(0)|| (log scale)')
    plt.title('State Deviation from Initial')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Linearization error
    ax5 = plt.subplot(3, 3, 5)
    error = np.linalg.norm(traj_nonlinear - traj_linear, axis=1)
    error_euler = np.linalg.norm(traj_nonlinear - traj_linear_euler, axis=1)
    
    plt.semilogy(time_steps, error + 1e-10, 'r-', label='Matrix exp error', linewidth=2)
    plt.semilogy(time_steps, error_euler + 1e-10, 'b-', label='Euler error', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Linearization Error (log scale)')
    plt.title('Nonlinear vs Linear Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Phase portrait (angle vs frequency)
    ax6 = plt.subplot(3, 3, 6)
    plt.plot(traj_nonlinear[:, 0] * 180/np.pi, traj_nonlinear[:, 9] * 60/(2*np.pi), 
             'k-', label='Nonlinear', linewidth=2)
    plt.plot(traj_linear[:, 0] * 180/np.pi, traj_linear[:, 9] * 60/(2*np.pi), 
             'r--', label='Linear', linewidth=1.5)
    plt.plot(traj_nonlinear[0, 0] * 180/np.pi, traj_nonlinear[0, 9] * 60/(2*np.pi), 
             'go', markersize=10, label='Start')
    plt.xlabel('Angle 1-2 (degrees)')
    plt.ylabel('Frequency 1 (Hz)')
    plt.title('Phase Portrait')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. All angles
    ax7 = plt.subplot(3, 3, 7)
    for i in range(9):
        plt.plot(time_steps, traj_nonlinear[:, i] * 180/np.pi, 
                 label=f'θ_{i+1},{i+2}' if i < 3 else None)
    plt.xlabel('Time (s)')
    plt.ylabel('Angles (degrees)')
    plt.title('All Angle Differences - Nonlinear')
    if i < 3:
        plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. All frequencies
    ax8 = plt.subplot(3, 3, 8)
    for i in range(10):
        plt.plot(time_steps, traj_nonlinear[:, 9+i] * 60/(2*np.pi), 
                 label=f'ω_{i+1}' if i < 3 else None)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequencies (Hz)')
    plt.title('All Frequencies - Nonlinear')
    if i < 3:
        plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Energy
    ax9 = plt.subplot(3, 3, 9)
    # Compute energy for each trajectory
    # This is simplified - would need actual energy function
    energy_nonlinear = np.sum(traj_nonlinear[:, 9:]**2, axis=1)  # Kinetic energy proxy
    energy_linear = np.sum(traj_linear[:, 9:]**2, axis=1)
    
    plt.plot(time_steps, energy_nonlinear, 'k-', label='Nonlinear', linewidth=2)
    plt.plot(time_steps, energy_linear, 'r--', label='Linear', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Kinetic Energy (proxy)')
    plt.title('Energy Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Main analysis"""
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create system
    sys, x_eq = create_power_system()
    
    # Analyze linearization
    A, eigvals = analyze_linearization(sys, x_eq)
    
    # Simulate
    time_steps, traj_nonlinear, traj_linear, traj_linear_euler = simulate_systems(
        sys, x_eq, A, T_sim=5.0
    )
    
    # Plot results
    fig = plot_results(time_steps, traj_nonlinear, traj_linear, traj_linear_euler, eigvals)
    
    # Save and show
    plt.savefig('linearized_system_analysis.png', dpi=150, bbox_inches='tight')
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Check if linearization is valid
    final_error = np.linalg.norm(traj_nonlinear[-1] - traj_linear[-1])
    max_error = np.max(np.linalg.norm(traj_nonlinear - traj_linear, axis=1))
    
    print(f"\n1. Linearization Quality:")
    print(f"   Final error at t=5s: {final_error:.6f}")
    print(f"   Maximum error: {max_error:.6f}")
    print(f"   Relative error: {max_error / np.max(np.linalg.norm(traj_nonlinear, axis=1)):.2%}")
    
    print(f"\n2. Stability Assessment:")
    if np.max(eigvals.real) > 1e-10:
        print(f"   ⚠️  The linearized system is UNSTABLE")
        print(f"   This explains why reduction methods may struggle")
    else:
        print(f"   ✓  The linearized system is stable")
        print(f"   Instabilities in reduced models come from reduction, not linearization")
    
    print(f"\n3. Implications for Reduction:")
    print(f"   - If linearization is unstable, all linear reduction methods will be unstable")
    print(f"   - Nonlinear reduction methods might handle this better")
    print(f"   - Energy-preserving methods (SPR) might stabilize through structure preservation")
    
    plt.show()
    
    return A, eigvals, time_steps, traj_nonlinear, traj_linear


if __name__ == "__main__":
    A, eigvals, time_steps, traj_nonlinear, traj_linear = main()