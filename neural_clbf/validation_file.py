import torch
from neural_clbf.systems import SwingEquationSystem
from neural_clbf.dimension_reduction.manager import select_reducer
from neural_clbf.eval.reduction_validation import validate_reducer

# Create system with 10 nodes
n = 10
params = dict(
    M=torch.ones(n) * 2.0,
    D=torch.ones(n) * 0.1,
    P=torch.zeros(n),
    K=torch.ones(n, n) * 0.5,
)

print("Creating system with 10 nodes...")
print(f"K matrix shape: {params['K'].shape}")
sys = SwingEquationSystem(params)
print(f"System created: n_dims = {sys.n_dims}, n_machines = {sys.n_machines}")

print("\nCollecting trajectories...")
data = sys.collect_random_trajectories(20, return_derivative=True)
print(f"Data collected: X.shape = {data['X'].shape}, dXdt.shape = {data['dXdt'].shape}")

print("\nTesting energy function on single state...")
test_state = data["X"][:1]
print(f"Test state shape: {test_state.shape}")
energy = sys.energy_function(test_state)
print(f"Energy computed successfully: {energy}")

print("\nSelecting reducer...")
red = select_reducer(sys, data["X"], data["dXdt"], d_max=10)
print(f"Reducer selected: {type(red).__name__} with latent_dim = {red.latent_dim}")
print(f"Reducer gamma = {red.gamma}")

print("\nValidating reducer...")
print("This will run 20 rollouts with 2.0 second horizon...")
print("-" * 60)

try:
    results = validate_reducer(sys, red, n_rollouts=20, horizon=2.0)
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS:")
    print("="*60)
    
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                print(f"{key}: {value.item():.6f}")
            else:
                print(f"{key}: shape {value.shape}, mean={value.mean().item():.6f}, std={value.std().item():.6f}")
        else:
            print(f"{key}: {value}")
    
    # Highlight key metrics
    print("\n" + "-"*60)
    print("KEY PERFORMANCE METRICS:")
    print("-"*60)
    
    if 'mean_error' in results:
        print(f"Mean reconstruction error: {results['mean_error'].item():.6f}")
    
    if 'max_error' in results:
        print(f"Max reconstruction error: {results['max_error'].item():.6f}")
    
    if 'relative_error' in results:
        print(f"Relative error: {results['relative_error'].item():.2%}")
    
except Exception as e:
    print(f"\nError during validation: {e}")
    import traceback
    traceback.print_exc()