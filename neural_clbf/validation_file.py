from neural_clbf.systems import SwingEquationSystem
from neural_clbf.dimension_reduction.manager import select_reducer
from neural_clbf.eval.reduction_validation import validate_reducer
import torch

# Create nominal parameters for a 10-node system
n_nodes = 10
nominal_params = {
    "M": torch.ones(n_nodes) * 2.0,  # Inertia constants
    "D": torch.ones(n_nodes) * 0.1,  # Damping coefficients  
    "P": torch.zeros(n_nodes),       # Mechanical power inputs
    "K": torch.ones(n_nodes, n_nodes) * 0.5  # Coupling strength matrix
}

# Initialize system with proper parameters
sys = SwingEquationSystem(nominal_params)

# Collect data and test reducer
data = sys.collect_random_trajectories(N_traj=10, return_derivative=True)
red = select_reducer(sys, data["X"], data["dXdt"], d_max=10)

metrics = validate_reducer(sys, red,
                          n_rollouts=50,
                          horizon=5.0,
                          dt=0.01,
                          input_mode="zero")
print(metrics)