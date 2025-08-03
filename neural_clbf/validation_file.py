
from neural_clbf.systems import SwingEquationSystem
from neural_clbf.dimension_reduction.manager import select_reducer
from neural_clbf.eval.reduction_validation import validate_reducer

sys = SwingEquationSystem(n_nodes=10)
data = sys.collect_random_trajectories(N_traj=8000, return_derivative=True)
red  = select_reducer(sys, data["X"], data["dXdt"], d_max=10)

metrics = validate_reducer(sys, red,
                            n_rollouts=50,
                            horizon=5.0,
                            dt=0.01,
                            input_mode="zero")   # or "random"
print(metrics)