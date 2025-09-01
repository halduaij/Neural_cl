from typing import Tuple, Optional, Union

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import torch
import torch.nn.functional as F
def to_numpy(tensor):
    if tensor.device.type == 'cuda':
        return tensor.cpu().numpy()
    return tensor.numpy()

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.utils import ScenarioList
from neural_clbf.controllers.controller import Controller
from neural_clbf.experiments import ExperimentSuite


class CLFController(Controller):
    """
    A generic CLF-based controller, using the quadratic Lyapunov function found for
    the linearized system.

    This controller and all subclasses assumes continuous-time dynamics.
    """

    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        scenarios: ScenarioList,
        experiment_suite: ExperimentSuite,
        clf_lambda: float = 1.0,
        clf_relaxation_penalty: float = 50.0,
        controller_period: float = 0.01,
        disable_gurobi: bool = True,
        robust_margin_gamma: float = 0.0,
        cache_size: int = 1000,  # Add caching for better performance
    ):
        """Initialize the controller.

        args:
            dynamics_model: the control-affine dynamics of the underlying system
            scenarios: a list of parameter scenarios to train on
            experiment_suite: defines the experiments to run during training
            clf_lambda: convergence rate for the CLF
            clf_relaxation_penalty: the penalty for relaxing CLF conditions.
            controller_period: the timestep to use in simulating forward Vdot
            disable_gurobi: if True, Gurobi will not be used during evaluation.
            cache_size: size of the LRU cache for computed control inputs
        """
        super(CLFController, self).__init__(
            dynamics_model=dynamics_model,
            experiment_suite=experiment_suite,
            controller_period=controller_period,
        )

        # Save gurobi attribute to disable if needed
        self.disable_gurobi = True

        # Save the provided model
        self.scenarios = scenarios
        self.n_scenarios = len(scenarios)

        # Save the experiments suits
        self.experiment_suite = experiment_suite

        # Save the other parameters
        self.clf_lambda = clf_lambda
        self.clf_relaxation_penalty = clf_relaxation_penalty
        self.robust_margin_gamma = robust_margin_gamma
        # Initialize cache for computed control inputs
        self.cache = {}
        self.cache_size = cache_size

        # Since we want to be able to solve the CLF-QP differentiably, we need to set
        # up the CVXPyLayers optimization. First, we define variables for each control
        # input and the relaxation in each scenario
        u = cp.Variable(self.dynamics_model.n_controls)
        clf_relaxations = []
        for scenario in self.scenarios:
            clf_relaxations.append(cp.Variable(1, nonneg=True))

        # Next, we define the parameters that will be supplied at solve-time: the value
        # of the Lyapunov function, its Lie derivatives, the relaxation penalty, and
        # the reference control input
        V_param = cp.Parameter(1, nonneg=True)
        Lf_V_params = []
        Lg_V_params = []
        for scenario in self.scenarios:
            Lf_V_params.append(cp.Parameter(1))
            Lg_V_params.append(cp.Parameter(self.dynamics_model.n_controls))

        clf_relaxation_penalty_param = cp.Parameter(1, nonneg=True)
        u_ref_param = cp.Parameter(self.dynamics_model.n_controls)

        # These allow us to define the constraints
        constraints = []
        for i in range(len(self.scenarios)):
            # CLF decrease constraint (with relaxation)
            constraints.append(
                Lf_V_params[i]
                + Lg_V_params[i] @ u
                + (self.clf_lambda + self.robust_margin_gamma) * V_param
                - clf_relaxations[i]
                <= 0
            )

        # Control limit constraints
        upper_lim, lower_lim = self.dynamics_model.control_limits
        for control_idx in range(self.dynamics_model.n_controls):
            constraints.append(u[control_idx] >= lower_lim[control_idx])
            constraints.append(u[control_idx] <= upper_lim[control_idx])

        # And define the objective
        objective_expression = cp.sum_squares(u - u_ref_param)
        for r in clf_relaxations:
            objective_expression += cp.multiply(clf_relaxation_penalty_param, r)
        objective = cp.Minimize(objective_expression)

        # Finally, create the optimization problem
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        variables = [u] + clf_relaxations
        parameters = Lf_V_params + Lg_V_params
        parameters += [V_param, u_ref_param, clf_relaxation_penalty_param]
        self.differentiable_qp_solver = CvxpyLayer(
            problem, variables=variables, parameters=parameters
        )

    def V_with_jacobian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the CLF value and its Jacobian

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the CLF
        returns:
            V: bs tensor of CLF values
            JV: bs x 1 x self.dynamics_model.n_dims Jacobian of each row of V wrt x
        """
        P = self.dynamics_model.P.type_as(x)
        P = P.reshape(1, self.dynamics_model.n_dims, self.dynamics_model.n_dims)
        V = 0.5 * F.bilinear(x, x, P).squeeze()
        V = V.reshape(x.shape[0])

        P = P.reshape(self.dynamics_model.n_dims, self.dynamics_model.n_dims)
        JV = F.linear(x, P)
        JV = JV.reshape(x.shape[0], 1, self.dynamics_model.n_dims)

        return V, JV

    def V(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the value of the CLF"""
        V, _ = self.V_with_jacobian(x)
        return V

    def V_lie_derivatives(
        self, x: torch.Tensor, scenarios: Optional[ScenarioList] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the Lie derivatives of the CLF V along the control-affine dynamics

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            scenarios: optional list of scenarios. Defaults to self.scenarios
        returns:
            Lf_V: bs x len(scenarios) x 1 tensor of Lie derivatives of V
                  along f
            Lg_V: bs x len(scenarios) x self.dynamics_model.n_controls tensor
                  of Lie derivatives of V along g
        """
        if scenarios is None:
            scenarios = self.scenarios
        n_scenarios = len(scenarios)

        _, gradV = self.V_with_jacobian(x)

        batch_size = x.shape[0]
        Lf_V = torch.zeros(batch_size, n_scenarios, 1).type_as(x)
        Lg_V = torch.zeros(batch_size, n_scenarios, self.dynamics_model.n_controls).type_as(x)

        for i in range(n_scenarios):
            s = scenarios[i]
            f, g = self.dynamics_model.control_affine_dynamics(x, params=s)

            Lf_V[:, i, :] = torch.bmm(gradV, f).squeeze(1)
            Lg_V[:, i, :] = torch.bmm(gradV, g).squeeze(1)

        return Lf_V, Lg_V

    def u_reference(self, x: torch.Tensor) -> torch.Tensor:
        """Determine the reference control input."""
        return self.dynamics_model.u_nominal(x)

    def _get_cache_key(self, x: torch.Tensor) -> str:
        """Generate a cache key for a state tensor."""
        # Convert state to a string representation for caching
        return str(x.detach().cpu().numpy().tobytes())

    def _update_cache(self, key: str, value: torch.Tensor):
        """Update the cache with a new key-value pair."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest item if cache is full
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value

    def _solve_CLF_QP_cvxpylayers(
        self,
        x: torch.Tensor,
        u_ref: torch.Tensor,
        V: torch.Tensor,
        Lf_V: torch.Tensor,
        Lg_V: torch.Tensor,
        relaxation_penalty: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determine the control input for a given state using a QP. Solves the QP using
        CVXPyLayers, which does allow for backpropagation, but is slower and less
        accurate than Gurobi.

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            u_ref: bs x self.dynamics_model.n_controls tensor of reference controls
            V: bs x 1 tensor of CLF values,
            Lf_V: bs x 1 tensor of CLF Lie derivatives,
            Lg_V: bs x self.dynamics_model.n_controls tensor of CLF Lie derivatives,
            relaxation_penalty: the penalty to use for CLF relaxation.
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
            relaxation: bs x 1 tensor of how much the CLF had to be relaxed in each
                        case
        """
        relaxation_penalty = min(relaxation_penalty, 1e6)

        params = []
        for i in range(self.n_scenarios):
            params.append(Lf_V[:, i, :])
        for i in range(self.n_scenarios):
            params.append(Lg_V[:, i, :])
        params.append(V.reshape(-1, 1))
        params.append(u_ref)
        params.append(torch.tensor([relaxation_penalty]).type_as(x))

        # Use solver with increased iterations for better accuracy
        result = self.differentiable_qp_solver(
            *params,
            solver_args={
                "max_iters": 2000,  # Increased from 1000
                "eps": 1e-6,  # Tighter tolerance
                "verbose": False,
            },
        )

        u_result = result[0]
        r_result = torch.hstack(result[1:])

        return u_result.type_as(x), r_result.type_as(x)

    def solve_CLF_QP(
        self,
        x,
        relaxation_penalty: Optional[float] = None,
        u_ref: Optional[torch.Tensor] = None,
        requires_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determine the control input for a given state using a QP"""
        # Check cache first if not requiring gradients
        if not requires_grad:
            cache_key = self._get_cache_key(x)
            if cache_key in self.cache:
                return self.cache[cache_key]

        V = self.V(x)
        Lf_V, Lg_V = self.V_lie_derivatives(x)

        if u_ref is None:
            u_ref = self.u_reference(x)

        if relaxation_penalty is None:
            relaxation_penalty = self.clf_relaxation_penalty

        # Ensure only CVXPY solver is used
        result = self._solve_CLF_QP_cvxpylayers(x, u_ref, V, Lf_V, Lg_V, relaxation_penalty)

        # Cache the result if not requiring gradients
        if not requires_grad:
            self._update_cache(cache_key, result)

        return result

    def u(self, x):
        """Get the control input for a given state"""
        u, _ = self.solve_CLF_QP(x)
        return u