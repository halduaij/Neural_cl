import torch
from neural_clbf.systems import SwingEquationSystem
from neural_clbf.dimension_reduction.manager import select_reducer
from neural_clbf.eval.reduction_validation import validate_reducer

n = 10
params = dict(
    M=torch.ones(n) * 2.0,
    D=torch.ones(n) * 0.1,
    P=torch.zeros(n),
    K=torch.ones(n, n) * 0.5,
)
sys   = SwingEquationSystem(params)
data  = sys.collect_random_trajectories(20, return_derivative=True)
red   = select_reducer(sys, data["X"], data["dXdt"], d_max=10)
print(validate_reducer(sys, red, n_rollouts=20, horizon=2.0))
