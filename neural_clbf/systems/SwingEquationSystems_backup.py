import torch
import numpy as np
from typing import Tuple, Optional, List, Dict

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList, safe_mask


class SwingEquationSystem(ControlAffineSystem):
    """
    Represents a power system with 2 nodes modeled by the swing equation. 

    The system has state
        x = [theta_1, theta_2, omega_1, omega_2] 
    representing the angles and angular frequencies of 2 nodes, and it
    has control inputs
        u = [u_1, u_2] 
    representing the control input for each node (e.g. governor action).

    The system is parameterized by
        M_1, M_2: inertia constants (scalars)
        D_1, D_2: damping coefficients (scalars)
        P_1, P_2: mechanical power inputs (scalars)
        K: coupling strength (scalar)
    """

    # Number of nodes
    N_NODES = 2
    
    # Number of states and controls
    N_DIMS = 2 * N_NODES - 1  # One less state due to using difference
    N_CONTROLS = N_NODES

    # State indices
    THETA_12_IDX = 0
    OMEGA_START =  N_NODES - 1

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
        goal_point: Optional[torch.Tensor] = None
    ):
        """
        Initialize the swing equation system.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
                            Requires keys ["M_1", "M_2", "D_1", "D_2", "P_1", "P_2", "K"]
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the control discretization. Defaults to dt
            scenarios: optional list of scenarios to consider
            goal_point: optional tensor specifying the goal point for the system
        """
        self._goal_point = goal_point if goal_point is not None else torch.zeros((1, self.n_dims))

        super().__init__(
            nominal_params, dt=dt, controller_dt=controller_dt, scenarios=scenarios
        )

    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid"""
        valid = True
        valid = valid and "M_1" in params and isinstance(params["M_1"], (int, float))
        valid = valid and "M_2" in params and isinstance(params["M_2"], (int, float))
        valid = valid and "D_1" in params and isinstance(params["D_1"], (int, float))
        valid = valid and "D_2" in params and isinstance(params["D_2"], (int, float))
        valid = valid and "P_1" in params and isinstance(params["P_1"], (int, float))
        valid = valid and "P_2" in params and isinstance(params["P_2"], (int, float))
        valid = valid and "K" in params and isinstance(params["K"], (int, float))
        valid = valid and params["M_1"] > 0
        valid = valid and params["M_2"] > 0
        valid = valid and params["D_1"] > 0
        valid = valid and params["D_2"] > 0
        valid = valid and params["K"] > 0
        return valid

    @property
    def n_dims(self) -> int:
        return SwingEquationSystem.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return list(range(self.N_NODES))

    @property
    def n_controls(self) -> int:
        return SwingEquationSystem.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        upper_limit = torch.ones(self.n_dims)
        upper_limit[:self.N_NODES] = torch.pi  # theta limits
        upper_limit[self.N_NODES:] = 2.0  # omega limits

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        upper_limit = torch.ones(self.n_controls)
        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def goal_point(self) -> torch.Tensor:
        """
        Return the goal point for the system
        """
        return self._goal_point

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating points in the goal set"""
        goal_tolerance = 0.1
        return torch.all(torch.abs(x - self.goal_point) < goal_tolerance, dim=1)

    def safe_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating safe regions"""
        return torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating unsafe regions"""
        return torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)

    def _f(self, x: torch.Tensor, params: Scenario) -> torch.Tensor:
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system.
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)

        M_1, M_2 = params["M_1"], params["M_2"]
        D_1, D_2 = params["D_1"], params["D_2"]
        P_1, P_2 = params["P_1"], params["P_2"]
        K = params["K"]
        
        theta_12 = x[:, self.THETA_12_IDX]
        omega = x[:, self.OMEGA_START:]

        # Theta_12 dynamics
        f[:, self.THETA_12_IDX, 0] = omega[:, 0] - omega[:, 1]

        # Omega dynamics
        f[:, self.OMEGA_START, 0] = (P_1 - D_1 * omega[:, 0] - K * torch.sin(theta_12)) / M_1
        f[:, self.OMEGA_START + 1, 0] = (P_2 - D_2 * omega[:, 1] + K * torch.sin(theta_12)) / M_2

        return f

    def _g(self, x: torch.Tensor, params: Scenario) -> torch.Tensor:
        """
        Return the control-dependent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system.
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls))
        g = g.type_as(x)

        M_1, M_2 = params["M_1"], params["M_2"]

        g[:, self.OMEGA_START, 0] = 1 / M_1
        g[:, self.OMEGA_START + 1, 1] = 1 / M_2

        return g

    def u_nominal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the nominal control input for a given state.
        Each control is a function of its corresponding omega only.

        args:
            x: bs x self.n_dims tensor of state
        returns:
            u_nominal: bs x self.n_controls tensor of nominal control inputs
        """
        K = 1.0  # Some nominal gain
        omega = x[:, self.OMEGA_START:]
        u_nominal = -K * omega
        return u_nominal
