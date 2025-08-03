# tests/test_selector.py
def test_selector():
    from neural_clbf.systems import SwingEquationSystem
    from neural_clbf.dimension_reduction.manager import select_reducer

    sys = SwingEquationSystem(n_nodes=4)
    data = sys.collect_random_trajectories(1500, return_derivative=True)
    reducer = select_reducer(sys, data["X"], data["dXdt"], d_max=8)
    assert reducer.latent_dim < sys.n_dims
    assert reducer.gamma < 0.1
