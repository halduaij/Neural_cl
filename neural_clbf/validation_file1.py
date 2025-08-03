"""
Proper 10-generator power system test cases for transient stability
All nodes are generators - loads are implicit in the net power injections
"""
import torch
import numpy as np


def create_10gen_two_area_system():
    """
    10-generator system with two coherent areas.
    Area 1: 5 generators (northeast)
    Area 2: 5 generators (southwest)
    Connected by weak tie lines.
    
    This is ideal for coherency-based reduction to 2 groups.
    """
    n_gen = 10
    
    # Generator inertia constants (seconds)
    # Area 1: Large thermal units
    # Area 2: Mix of units
    M = torch.zeros(n_gen)
    M[0:5] = torch.tensor([6.5, 6.0, 5.8, 6.2, 5.5])  # Area 1
    M[5:10] = torch.tensor([4.0, 4.5, 3.8, 4.2, 5.0])  # Area 2
    
    # Damping coefficients (pu)
    D = torch.ones(n_gen) * 2.0  # Typical damping
    
    # Net power injections (generation - local load) in pu
    # Must sum to zero for power balance
    P = torch.zeros(n_gen)
    # Area 1: Net exporters (generation > local load)
    P[0:5] = torch.tensor([1.2, 0.8, 0.9, 0.7, 0.6])  # Total: 4.2
    # Area 2: Net importers (generation < local load)  
    P[5:10] = torch.tensor([-0.9, -0.8, -0.7, -0.9, -0.9])  # Total: -4.2
    
    # Reduced network admittance matrix (pu)
    K = torch.zeros(n_gen, n_gen)
    
    # Strong connections within Area 1
    K_area1 = 8.0  # Strong electrical coupling
    area1_connections = [
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)
    ]
    for i, j in area1_connections:
        K[i, j] = K[j, i] = K_area1
    
    # Strong connections within Area 2  
    K_area2 = 7.0
    area2_connections = [
        (5, 6), (5, 7), (6, 7), (6, 8), (7, 8), (7, 9), (8, 9)
    ]
    for i, j in area2_connections:
        i_shifted = i
        j_shifted = j
        K[i_shifted, j_shifted] = K[j_shifted, i_shifted] = K_area2
    
    # Weak tie lines between areas
    K_tie = 0.8  # Weak inter-area connection
    tie_lines = [
        (2, 6),  # Gen 3 to Gen 7
        (3, 7),  # Gen 4 to Gen 8  
    ]
    for i, j in tie_lines:
        K[i, j] = K[j, i] = K_tie
    
    print(f"10-Generator Two-Area System:")
    print(f"  Area 1: Generators 1-5 (thermal units)")
    print(f"  Area 2: Generators 6-10 (mixed units)")
    print(f"  Intra-area coupling: {K_area1:.1f}, {K_area2:.1f} (strong)")
    print(f"  Inter-area coupling: {K_tie:.1f} (weak)")
    print(f"  Power balance check: {P.sum():.6f}")
    print(f"  Expected coherent groups: 2")
    
    return dict(M=M, D=D, P=P, K=K)


def create_10gen_three_area_system():
    """
    10-generator system with three coherent areas.
    Area 1: 4 generators (north)
    Area 2: 3 generators (east)  
    Area 3: 3 generators (south)
    
    Good for testing 3-group coherency reduction.
    """
    n_gen = 10
    
    # Inertia by area
    M = torch.zeros(n_gen)
    M[0:4] = 6.0   # Area 1: Large units
    M[4:7] = 4.5   # Area 2: Medium units
    M[7:10] = 3.5  # Area 3: Smaller units
    
    # Uniform damping
    D = torch.ones(n_gen) * 1.8
    
    # Power injections (balanced)
    P = torch.zeros(n_gen)
    P[0:4] = 0.8    # Area 1: Generation, total 3.2
    P[4:7] = -0.6   # Area 2: Load, total -1.8  
    P[7:10] = -0.467 # Area 3: Load, total -1.4 (rounds to balance)
    P = P - P.mean()  # Ensure exact balance
    
    # Network connections
    K = torch.zeros(n_gen, n_gen)
    
    # Area 1 internal (gens 0-3)
    K_area1 = 10.0
    for i in range(4):
        for j in range(i+1, 4):
            K[i, j] = K[j, i] = K_area1
            
    # Area 2 internal (gens 4-6)
    K_area2 = 9.0
    for i in range(4, 7):
        for j in range(i+1, 7):
            K[i, j] = K[j, i] = K_area2
            
    # Area 3 internal (gens 7-9)
    K_area3 = 8.0
    for i in range(7, 10):
        for j in range(i+1, 10):
            K[i, j] = K[j, i] = K_area3
    
    # Inter-area ties
    K_tie = 1.0
    K[1, 5] = K[5, 1] = K_tie    # Area 1-2
    K[3, 8] = K[8, 3] = K_tie    # Area 1-3
    K[6, 9] = K[9, 6] = K_tie    # Area 2-3
    
    print(f"10-Generator Three-Area System:")
    print(f"  Area 1: Generators 1-4 (large units)")
    print(f"  Area 2: Generators 5-7 (medium units)")  
    print(f"  Area 3: Generators 8-10 (small units)")
    print(f"  Inter-area coupling: {K_tie:.1f} (weak)")
    print(f"  Power balance: {P.sum():.6f}")
    print(f"  Expected coherent groups: 3")
    
    return dict(M=M, D=D, P=P, K=K)


def create_10gen_radial_system():
    """
    10-generator radial/chain system.
    Generators arranged in a line with decreasing size.
    Natural coherency between nearby generators.
    """
    n_gen = 10
    
    # Decreasing inertia along the chain
    M = torch.linspace(8.0, 3.0, n_gen)
    
    # Damping proportional to inertia
    D = 0.25 * M  # D/M ratio around 0.25
    
    # Power: Large generators export, small ones have local load
    P = torch.zeros(n_gen)
    for i in range(n_gen):
        if i < 4:
            P[i] = 1.0 - 0.2 * i  # Decreasing generation
        else:
            P[i] = -0.3  # Uniform load
    P = P - P.mean()  # Balance
    
    # Chain/radial topology with nearest neighbor connections
    K = torch.zeros(n_gen, n_gen)
    
    # Main chain
    for i in range(n_gen - 1):
        # Stronger connections at generation end
        K[i, i+1] = K[i+1, i] = 6.0 - 0.4 * i
    
    # Add some longer-range connections for stability
    for i in range(n_gen - 2):
        K[i, i+2] = K[i+2, i] = 1.0  # Next-nearest neighbor
    
    print(f"10-Generator Radial System:")
    print(f"  Topology: Chain with decreasing unit size")
    print(f"  Inertia range: {M[0]:.1f} to {M[-1]:.1f}")
    print(f"  Natural coherency: Neighboring units")
    print(f"  Power balance: {P.sum():.6f}")
    
    return dict(M=M, D=D, P=P, K=K)


def create_10gen_symmetric_system():
    """
    10-generator system with symmetric parameters.
    Good for testing if reduction preserves symmetries.
    Arranged in two rings of 5 generators each.
    """
    n_gen = 10
    
    # Identical generators
    M = torch.ones(n_gen) * 5.0
    D = torch.ones(n_gen) * 1.5
    
    # Symmetric power pattern
    P = torch.zeros(n_gen)
    # Ring 1: alternating
    for i in range(5):
        P[i] = 0.5 * (-1)**i
    # Ring 2: opposite pattern  
    for i in range(5, 10):
        P[i] = -P[i-5]
    
    # Network: Two coupled rings
    K = torch.zeros(n_gen, n_gen)
    
    # Ring 1 (gens 0-4)
    K_ring = 4.0
    for i in range(5):
        j = (i + 1) % 5
        K[i, j] = K[j, i] = K_ring
        
    # Ring 2 (gens 5-9)
    for i in range(5, 10):
        j = 5 + ((i - 5 + 1) % 5)
        K[i, j] = K[j, i] = K_ring
    
    # Coupling between rings
    K_couple = 2.0
    for i in range(5):
        K[i, i+5] = K[i+5, i] = K_couple
    
    print(f"10-Generator Symmetric System:")
    print(f"  Structure: Two coupled 5-generator rings")
    print(f"  All generators identical: M={M[0]}, D={D[0]}")
    print(f"  Ring coupling: {K_ring}, Inter-ring: {K_couple}")
    print(f"  Power balance: {P.sum():.6f}")
    
    return dict(M=M, D=D, P=P, K=K)


def create_10gen_realistic_system():
    """
    More realistic 10-generator system based on typical power system data.
    Mixed generation types with realistic parameters.
    """
    n_gen = 10
    
    # Realistic inertia constants (H in seconds, M = 2H)
    # Nuclear: H=6-7, Coal: H=4-5, Gas: H=3-4, Hydro: H=2-3
    H = torch.tensor([
        6.5,  # G1: Nuclear
        6.2,  # G2: Nuclear  
        4.8,  # G3: Coal
        4.5,  # G4: Coal
        4.2,  # G5: Coal
        3.5,  # G6: Gas
        3.2,  # G7: Gas
        2.8,  # G8: Hydro
        2.5,  # G9: Hydro
        3.8   # G10: Gas
    ])
    M = 2 * H  # M = 2H in pu
    
    # Damping (typically D = 1-2 pu)
    D = torch.ones(n_gen) * 1.5
    
    # Generation capacities and dispatch (pu)
    # Base case: 60% loading of capacity
    capacity = torch.tensor([1.2, 1.0, 0.8, 0.8, 0.7, 
                           0.6, 0.5, 0.4, 0.4, 0.6])
    dispatch = 0.6 * capacity
    
    # Net injections (generation - load at bus)
    # Assume 30% of generation serves local load
    local_load_factor = 0.3
    P = dispatch * (1 - local_load_factor)
    
    # System must have zero net injection
    total_load = P.sum()
    # Distribute excess as load
    P = P - total_load / n_gen
    
    # Realistic network (based on geographical/electrical proximity)
    K = torch.zeros(n_gen, n_gen)
    
    # Define connections based on "electrical distance"
    connections = [
        # Strong connections (nearby plants)
        (0, 1, 12.0),  # Nuclear plants (usually built in pairs)
        (2, 3, 10.0),  # Coal plants
        (3, 4, 10.0),
        (5, 6, 8.0),   # Gas plants
        (7, 8, 9.0),   # Hydro plants
        
        # Medium connections
        (0, 2, 5.0),
        (1, 3, 5.0),
        (4, 5, 4.0),
        (6, 9, 4.0),
        (8, 9, 5.0),
        
        # Weak connections (long distance)
        (0, 7, 2.0),
        (2, 6, 1.5),
        (4, 8, 2.0),
        (1, 9, 1.8),
    ]
    
    for i, j, k_val in connections:
        K[i, j] = K[j, i] = k_val
    
    print(f"10-Generator Realistic System:")
    print(f"  Types: 2 Nuclear, 3 Coal, 3 Gas, 2 Hydro")
    print(f"  Inertia range: H = {H.min():.1f} to {H.max():.1f} seconds")
    print(f"  Total capacity: {capacity.sum():.1f} pu")
    print(f"  Dispatch: 60% of capacity")
    print(f"  Power balance: {P.sum():.6f}")
    
    # Expected coherency: Nuclear together, Coal together, etc.
    print(f"  Expected groups: Nuclear (G1-2), Coal (G3-5), Fast (G6-10)")
    
    return dict(M=M, D=D, P=P, K=K)

import torch
import numpy as np
import matplotlib.pyplot as plt
from neural_clbf.systems import SwingEquationSystem
from neural_clbf.dimension_reduction.manager import select_reducer
from neural_clbf.eval.reduction_validation import validate_reducer
import time

# Import the 10-generator test cases (add the previous artifact's functions here)
# For now, I'll include the key ones directly

def create_10gen_two_area_system():
    """10-generator system with two coherent areas."""
    n_gen = 10
    
    M = torch.zeros(n_gen)
    M[0:5] = torch.tensor([6.5, 6.0, 5.8, 6.2, 5.5])
    M[5:10] = torch.tensor([4.0, 4.5, 3.8, 4.2, 5.0])
    
    D = torch.ones(n_gen) * 2.0
    
    P = torch.zeros(n_gen)
    P[0:5] = torch.tensor([1.2, 0.8, 0.9, 0.7, 0.6])
    P[5:10] = torch.tensor([-0.9, -0.8, -0.7, -0.9, -0.9])
    
    K = torch.zeros(n_gen, n_gen)
    K_area1 = 8.0
    area1_connections = [(0,1), (0,2), (1,2), (1,3), (2,3), (2,4), (3,4)]
    for i, j in area1_connections:
        K[i, j] = K[j, i] = K_area1
    
    K_area2 = 7.0
    area2_connections = [(5,6), (5,7), (6,7), (6,8), (7,8), (7,9), (8,9)]
    for i, j in area2_connections:
        K[i, j] = K[j, i] = K_area2
    
    K_tie = 0.8
    K[2, 6] = K[6, 2] = K_tie
    K[3, 7] = K[7, 3] = K_tie
    
    return dict(M=M, D=D, P=P, K=K)


def create_10gen_three_area_system():
    """10-generator system with three coherent areas."""
    n_gen = 10
    
    M = torch.zeros(n_gen)
    M[0:4] = 6.0
    M[4:7] = 4.5
    M[7:10] = 3.5
    
    D = torch.ones(n_gen) * 1.8
    
    P = torch.zeros(n_gen)
    P[0:4] = 0.8
    P[4:7] = -0.6
    P[7:10] = -0.467
    P = P - P.mean()
    
    K = torch.zeros(n_gen, n_gen)
    
    # Area 1
    for i in range(4):
        for j in range(i+1, 4):
            K[i, j] = K[j, i] = 10.0
    
    # Area 2
    for i in range(4, 7):
        for j in range(i+1, 7):
            K[i, j] = K[j, i] = 9.0
    
    # Area 3
    for i in range(7, 10):
        for j in range(i+1, 10):
            K[i, j] = K[j, i] = 8.0
    
    # Ties
    K[1, 5] = K[5, 1] = 1.0
    K[3, 8] = K[8, 3] = 1.0
    K[6, 9] = K[9, 6] = 1.0
    
    return dict(M=M, D=D, P=P, K=K)


def create_10gen_realistic_system():
    """Realistic 10-generator system with mixed types."""
    n_gen = 10
    
    H = torch.tensor([6.5, 6.2, 4.8, 4.5, 4.2, 3.5, 3.2, 2.8, 2.5, 3.8])
    M = 2 * H
    D = torch.ones(n_gen) * 1.5
    
    capacity = torch.tensor([1.2, 1.0, 0.8, 0.8, 0.7, 0.6, 0.5, 0.4, 0.4, 0.6])
    dispatch = 0.6 * capacity
    local_load_factor = 0.3
    P = dispatch * (1 - local_load_factor)
    P = P - P.mean()
    
    K = torch.zeros(n_gen, n_gen)
    connections = [
        (0, 1, 12.0), (2, 3, 10.0), (3, 4, 10.0), (5, 6, 8.0), (7, 8, 9.0),
        (0, 2, 5.0), (1, 3, 5.0), (4, 5, 4.0), (6, 9, 4.0), (8, 9, 5.0),
        (0, 7, 2.0), (2, 6, 1.5), (4, 8, 2.0), (1, 9, 1.8)
    ]
    
    for i, j, k_val in connections:
        K[i, j] = K[j, i] = k_val
    
    return dict(M=M, D=D, P=P, K=K)


def run_10gen_validation(test_case="two_area", n_rollouts=20, horizon=2.0, 
                        d_max_values=[4, 6, 8, 10, 12, 14]):
    """
    Run validation on 10-generator test systems.
    """
    print("="*80)
    print(f"10-GENERATOR SYSTEM VALIDATION")
    print(f"Test Case: {test_case}")
    print(f"Validation: {n_rollouts} rollouts, {horizon}s horizon")
    print("="*80)
    
    # Select test case
    test_cases = {
        "two_area": (create_10gen_two_area_system(), "Two-Area (5+5)", 2),
        "three_area": (create_10gen_three_area_system(), "Three-Area (4+3+3)", 3),
        "realistic": (create_10gen_realistic_system(), "Realistic Mixed Types", 3)
    }
    
    if test_case not in test_cases:
        print(f"Unknown test case: {test_case}")
        print(f"Available: {list(test_cases.keys())}")
        return
    
    params, description, expected_groups = test_cases[test_case]
    
    print(f"\n1. Creating {description} system...")
    sys = SwingEquationSystem(params, dt=0.01)
    print(f"   ✓ System dimensions: {sys.n_dims} (10 generators)")
    print(f"   ✓ Expected coherent groups: {expected_groups}")
    print(f"   ✓ Power balance: {params['P'].sum():.6f}")
    
    # Collect training data
    print("\n2. Collecting training trajectories...")
    start_time = time.time()
    
    # Use smaller disturbances for more stable data
    data = sys.collect_random_trajectories(
        N_traj=150,  # More trajectories
        T_steps=50,
        control_excitation=0.05,  # Smaller control inputs
        return_derivative=True
    )
    
    collect_time = time.time() - start_time
    print(f"   ✓ Collected {data['X'].shape[0]} snapshots in {collect_time:.2f}s")
    print(f"   ✓ State range: [{data['X'].min():.3f}, {data['X'].max():.3f}]")
    
    # Dimension reduction and validation
    results_by_d = {}
    
    for d_max in d_max_values:
        if d_max >= sys.n_dims:
            continue
            
        print(f"\n3. Selecting reducer with d_max={d_max}...")
        
        try:
            red = select_reducer(
                sys, data["X"], data["dXdt"], 
                d_max=d_max, 
                verbose=True
            )
            
            print(f"\n   ✓ Selected: {type(red).__name__}")
            print(f"   ✓ Latent dimension: {red.latent_dim}")
            print(f"   ✓ Gamma (robustness): {red.gamma:.6f}")
            
            # For Lyapunov Coherency, show grouping
            if hasattr(red, 'labels'):
                print(f"   ✓ Generator groups: {red.labels.tolist()}")
            
            # Skip if gamma too high
            if red.gamma > 10:
                print(f"   ⚠ Skipping validation (gamma too high)")
                continue
            
            # Validate
            print(f"\n4. Validating reducer...")
            start_time = time.time()
            
            results = validate_reducer(
                sys, red, 
                n_rollouts=n_rollouts,
                horizon=horizon,
                dt=0.01
            )
            
            validate_time = time.time() - start_time
            
            results_by_d[red.latent_dim] = {
                'reducer_type': type(red).__name__,
                'gamma': red.gamma,
                'results': results,
                'validate_time': validate_time
            }
            
            print(f"   ✓ Validation completed in {validate_time:.2f}s")
            print(f"   Mean error: {results['mean_error']:.3f}")
            print(f"   Relative error: {results['relative_error']:.1%}")
            print(f"   Success rate: {results['success_rate']:.1%}")
            
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n{'Dim':<6} {'Type':<25} {'Gamma':<10} {'Mean Err':<12} {'Rel Err':<10} {'Success':<10}")
    print("-"*78)
    
    for d, info in sorted(results_by_d.items()):
        r = info['results']
        print(f"{d:<6} {info['reducer_type']:<25} {info['gamma']:<10.3f} "
              f"{r['mean_error']:<12.3f} {r['relative_error']:<10.1%} "
              f"{r['success_rate']:<10.1%}")
    
    # Plot if we have results
    if len(results_by_d) > 1:
        plot_reduction_results(results_by_d, sys.n_dims, test_case)
    
    return results_by_d


def plot_reduction_results(results_by_d, full_dim, test_case):
    """Create plots for 10-generator validation results."""
    dims = sorted(results_by_d.keys())
    
    # Extract data
    mean_errors = [results_by_d[d]['results']['mean_error'].item() for d in dims]
    rel_errors = [results_by_d[d]['results']['relative_error'].item() for d in dims]
    gammas = [results_by_d[d]['gamma'] for d in dims]
    types = [results_by_d[d]['reducer_type'] for d in dims]
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'10-Generator System Model Reduction ({test_case})', fontsize=16)
    
    # 1. Error vs dimension
    ax1.semilogy(dims, mean_errors, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Reduced Dimension')
    ax1.set_ylabel('Mean Trajectory Error')
    ax1.set_title('Reconstruction Error')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    
    # 2. Relative error
    ax2.plot(dims, [100*e for e in rel_errors], 'g-s', linewidth=2, markersize=8)
    ax2.set_xlabel('Reduced Dimension')
    ax2.set_ylabel('Relative Error (%)')
    ax2.set_title('Relative Error')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='10% target')
    
    # 3. Gamma vs dimension
    ax3.plot(dims, gammas, 'r-^', linewidth=2, markersize=8)
    ax3.set_xlabel('Reduced Dimension')
    ax3.set_ylabel('Gamma (Robustness Margin)')
    ax3.set_title('Stability Margin')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1.0, color='g', linestyle='--', alpha=0.5)
    
    # 4. Compression vs error trade-off
    compressions = [d/full_dim * 100 for d in dims]
    ax4.plot(compressions, rel_errors, 'mo-', linewidth=2, markersize=8)
    ax4.set_xlabel('Compression (%)')
    ax4.set_ylabel('Relative Error')
    ax4.set_title('Compression vs Accuracy Trade-off')
    ax4.grid(True, alpha=0.3)
    
    # Add reducer type annotations
    for i, (c, e, t) in enumerate(zip(compressions, rel_errors, types)):
        if i % 2 == 0:  # Annotate every other point
            ax4.annotate(f'{t[:3]}', (c, e), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'10gen_validation_{test_case}.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Plots saved to '10gen_validation_{test_case}.png'")
    plt.show()


if __name__ == "__main__":
    # Test different 10-generator systems
    test_cases = ["two_area", "three_area", "realistic"]
    
    for test_case in test_cases:
        print(f"\n{'='*80}")
        print(f"Testing {test_case} system")
        print(f"{'='*80}")
        
        results = run_10gen_validation(
            test_case=test_case,
            n_rollouts=20,
            horizon=2.0,  # Shorter horizon for stability
            d_max_values=[4, 6, 8, 10, 12, 14, 16]
        )
        
        print("\n" + "="*80 + "\n")