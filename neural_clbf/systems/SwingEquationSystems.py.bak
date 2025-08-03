import torch
import numpy as np
from typing import Tuple, Optional, List, Dict

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList

class SwingEquationSystem(ControlAffineSystem):
    """
    Represents a power system with n nodes modeled by the swing equation. 

    The system has state
        x = [theta_12, ..., theta_1n, omega_1, ..., omega_n] 
    representing the angles (n-1) and angular frequencies (n) of nodes, and it
    has control inputs
        u = [u_1, ..., u_n] 
    representing the control input for each node (e.g. governor action).

    The system is parameterized by
        M: inertia constants (vector of length n)
        D: damping coefficients (vector of length n)
        P: mechanical power inputs (vector of length n)
        K: coupling strength matrix (n x n)
    """

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
        goal_point: Optional[torch.Tensor] = None,
        max_rocof: float = 1.0  # Maximum allowed Rate of Change of Frequency
    ):
        """
        Initialize the swing equation system.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
                            Requires keys ["M", "D", "P", "K"]
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the control discretization. Defaults to dt
            scenarios: optional list of scenarios to consider
            goal_point: optional tensor specifying the goal point for the system
            max_rocof: maximum allowed Rate of Change of Frequency (Hz/s)
        """
        self.N_NODES = len(nominal_params["M"])
        self.N_DIMS = 2 * self.N_NODES - 1  # One less state due to using difference
        self.N_CONTROLS = self.N_NODES
        self.max_rocof = max_rocof  # Store maximum allowed RoCoF

        self._goal_point = goal_point if goal_point is not None else torch.zeros((1, self.n_dims))

        super().__init__(
            nominal_params, dt=dt, controller_dt=controller_dt, scenarios=scenarios
        )

    def validate_params(self, params: Scenario) -> bool:
        """
        Check whether the given parameters are valid for this system.

        args:
            params: a dictionary giving the parameter values for the system.
        returns:
            valid: True if the parameters are valid, False otherwise
        """
        required_keys = ["M", "D", "P", "K"]
        for key in required_keys:
            if key not in params:
                return False
        return True

    @property
    def n_dims(self) -> int:
        return self.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return list(range(self.N_NODES - 1))

    @property
    def n_controls(self) -> int:
        return self.N_CONTROLS
    
    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        upper_limit = torch.ones(self.n_dims)
        upper_limit[:self.N_NODES - 1] = torch.pi  # theta limits
        upper_limit[self.N_NODES - 1:] = 2.0  # omega limits

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

    def compute_rocof(self, x: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the Rate of Change of Frequency (RoCoF) for each generator.
        
        RoCoF is the time derivative of the frequency, which is directly
        related to omega_dot in our state space formulation.
        
        args:
            x: bs x self.n_dims tensor of state
            u: bs x self.n_controls tensor of control inputs (optional)
        
        returns:
            rocof: bs x self.N_NODES tensor of RoCoF values for each node
        """
        batch_size = x.shape[0]
        
        # Get dynamics with the nominal scenario parameters
        f = self._f(x, self.nominal_params)
        
        # If control inputs are provided, include their effect
        if u is not None:
            g = self._g(x, self.nominal_params)
            u_reshaped = u.reshape(-1, self.n_controls, 1)
            dynamics = f + torch.bmm(g, u_reshaped)
        else:
            # Just use unforced dynamics (or use nominal control)
            u_nominal = self.u_nominal(x)
            g = self._g(x, self.nominal_params)
            u_reshaped = u_nominal.reshape(-1, self.n_controls, 1)
            dynamics = f + torch.bmm(g, u_reshaped)
        
        # Extract the dynamics of omega (which gives us RoCoF)
        # We multiply by 60/(2*pi) to convert from rad/s^2 to Hz/s if needed
        # (standard power system frequency unit, assuming 60Hz system)
        conversion_factor = 60.0 / (2.0 * np.pi)
        rocof = dynamics[:, self.N_NODES-1:, 0] * conversion_factor
        
        return rocof

    def safe_mask(self, x: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return the mask of x indicating safe regions based on RoCoF constraints.
        
        args:
            x: bs x self.n_dims tensor of state
            u: bs x self.n_controls tensor of control inputs (optional)
            
        returns:
            safe_mask: bs tensor of booleans indicating safe states
        """
        # First check if state is within usual limits
        upper_limit, lower_limit = self.state_limits
        within_limits = torch.all((x <= upper_limit) & (x >= lower_limit), dim=1)
        
        # Then compute RoCoF for each generator
        rocof = self.compute_rocof(x, u)
        
        # Check if RoCoF is within acceptable limits for all generators
        rocof_safe = torch.all(torch.abs(rocof) <= self.max_rocof, dim=1)
        
        # State is safe if both conditions are met
        return within_limits & rocof_safe

    def unsafe_mask(self, x: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return the mask of x indicating unsafe regions based on RoCoF constraints.
        
        args:
            x: bs x self.n_dims tensor of state
            u: bs x self.n_controls tensor of control inputs (optional)
            
        returns:
            unsafe_mask: bs tensor of booleans indicating unsafe states
        """
        # First check if state is outside usual limits
        upper_limit, lower_limit = self.state_limits
        outside_limits = torch.any((x > upper_limit) | (x < lower_limit), dim=1)
        
        # Then compute RoCoF for each generator
        rocof = self.compute_rocof(x, u)
        
        # Check if RoCoF exceeds acceptable limits for any generator
        rocof_unsafe = torch.any(torch.abs(rocof) > self.max_rocof, dim=1)
        
        # State is unsafe if either condition is met
        return outside_limits | rocof_unsafe

    def get_rocof_barrier_value(self, x: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the barrier function value based on RoCoF constraints.
        
        The barrier function is positive when the state is safe (RoCoF is within limits)
        and negative when the state is unsafe (RoCoF exceeds limits).
        
        args:
            x: bs x self.n_dims tensor of state
            u: bs x self.n_controls tensor of control inputs (optional)
            
        returns:
            barrier: bs tensor of barrier function values
        """
        # Compute RoCoF for each generator
        rocof = self.compute_rocof(x, u)
        
        # Compute the margin to the RoCoF limit for each generator
        # This is positive when RoCoF is within limits, negative when it exceeds
        margins = self.max_rocof - torch.abs(rocof)
        
        # The barrier value is the minimum margin across all generators
        # If any generator exceeds the limit, the barrier will be negative
        barrier = torch.min(margins, dim=1)[0]
        
        return barrier

    def _f(self, x: torch.Tensor, params: Scenario) -> torch.Tensor:
        """
        Return the drift dynamics of the control-affine system.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system.
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)

        M = params["M"].expand(batch_size, -1)
        D = params["D"].expand(batch_size, -1)
        P = params["P"].expand(batch_size, -1)
        K = params["K"]
        
        theta = x[:, :self.N_NODES - 1]
        omega = x[:, self.N_NODES - 1:]

        # Theta dynamics (6b)
        for i in range(1, self.N_NODES):
            f[:, i - 1, 0] = omega[:, 0] - omega[:, i]

        # Omega dynamics (6c)
        # For omega_1 (i = 1)
        i = 0
        coupling_sum_1 = torch.zeros(batch_size).type_as(x)
        for j in range(1, self.N_NODES):
            coupling_sum_1 += K[i, j] * torch.sin(theta[:, j - 1])
        f[:, self.N_NODES - 1 + i, 0] = (P[:, i] - D[:, i] * omega[:, i] - coupling_sum_1) / M[:, i]

        # For omega_i (i = 2, ..., n)
        for i in range(1, self.N_NODES):
            coupling_sum_i = torch.zeros(batch_size).type_as(x)
            # B_1i * sin(theta_1i)
            coupling_sum_i += K[i, 0] * torch.sin(theta[:, i - 1])
            # Sum B_ij * sin(theta_1i - theta_1j), j != i
            for j in range(1, self.N_NODES):
                if i != j:
                    coupling_sum_i += K[i, j] * torch.sin(theta[:, i - 1] - theta[:, j - 1])
            f[:, self.N_NODES - 1 + i, 0] = (P[:, i] - D[:, i] * omega[:, i] + coupling_sum_i) / M[:, i]

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

        M = params["M"].expand(batch_size, -1)

        for i in range(self.N_NODES):
            g[:, self.N_NODES - 1 + i, i] = 1 / M[:, i]

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
        omega = x[:, self.N_NODES - 1:]
        u_nominal = -K * omega
        return u_nominal
        
    def create_realistic_scenarios(self, n_scenarios: int = 5, variation: float = 0.2) -> ScenarioList:
        """
        Create a list of realistic scenarios with varying parameters.
        
        args:
            n_scenarios: number of scenarios to create
            variation: the relative variation in parameters
        returns:
            scenarios: list of scenarios
        """
        scenarios = []
        
        # Use the nominal parameters as the base
        base_params = self.nominal_params
        
        for i in range(n_scenarios):
            # Create variations of the parameters
            M_var = base_params["M"] * (1 + variation * (2 * torch.rand_like(base_params["M"]) - 1))
            D_var = base_params["D"] * (1 + variation * (2 * torch.rand_like(base_params["D"]) - 1))
            P_var = base_params["P"] + 0.1 * variation * (2 * torch.rand_like(base_params["P"]) - 1)
            
            # Keep K matrix structure but vary the coupling strengths
            K_var = base_params["K"] * (1 + variation * (2 * torch.rand_like(base_params["K"]) - 1))
            
            # Create scenario
            scenario = {
                "M": M_var,
                "D": D_var,
                "P": P_var, 
                "K": K_var
            }
            
            scenarios.append(scenario)
        
        return scenarios
