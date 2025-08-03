import torch
import numpy as np
from typing import Tuple, Optional, List, Dict
import math
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
        B_matrix: coupling strength matrix (n x n) - renamed from K to avoid conflict
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
        
        # Store parameters with different names to avoid conflict with parent class
        # Parent class uses self.P for Lyapunov matrix and self.K for controller gain
        self.M_inertia = torch.as_tensor(nominal_params["M"])   # (N,)
        self.D_damping = torch.as_tensor(nominal_params["D"])
        self.P_mechanical = torch.as_tensor(nominal_params["P"])
        self.B_matrix = torch.as_tensor(nominal_params["K"])   # (N,N) - coupling matrix
        
        # Create aliases for backward compatibility
        self.M = self.M_inertia
        self.D = self.D_damping
        # Don't create alias for P to avoid conflict
        
        # Verify B_matrix has the correct shape
        assert self.B_matrix.shape == (self.N_NODES, self.N_NODES), \
            f"Coupling matrix has wrong shape: {self.B_matrix.shape}, expected ({self.N_NODES}, {self.N_NODES})"

        self.n_machines = self.N_NODES        # alias used by reducers
        self._goal_point = goal_point if goal_point is not None else torch.zeros((1, self.n_dims))

        super().__init__(
            nominal_params, dt=dt, controller_dt=controller_dt, scenarios=scenarios,
            use_linearized_controller=False  # Don't compute linearized controller
        )
        
    # Property to maintain compatibility with code that expects self.K
    @property
    def K(self):
        """Return the coupling matrix (for backward compatibility)"""
        return self.B_matrix
    
    def compute_linearized_controller(self, scenarios: Optional[ScenarioList] = None):
        """
        Override parent's method to set dummy P and K matrices that the parent expects.
        The swing equation system doesn't use the linearized controller from the parent class.
        """
        # Set dummy P (Lyapunov matrix) and K (controller gain) for parent class
        # These are not used by our system but parent class methods might expect them
        self.P = torch.eye(self.n_dims, dtype=torch.float32)
        self.K = torch.zeros(self.n_controls, self.n_dims, dtype=torch.float32)
    
    def simulate(self, x_init, num_steps, *args, **kwargs):
        """
        Override parent's simulate to handle the validation's u parameter.
        
        The validation passes u as a keyword argument, which the parent doesn't accept.
        This method extracts u, converts it to a controller, and calls parent's simulate.
        """
        # Ensure num_steps is an integer (fix for the string error)
        try:
            num_steps = int(num_steps)
        except (ValueError, TypeError) as e:
            raise TypeError(f"num_steps must be convertible to int, got {type(num_steps).__name__}: {num_steps}")
        
        # Extract u if provided
        u = kwargs.pop('u', None)
        
        # Extract other expected kwargs for parent's simulate
        controller_period = kwargs.pop('controller_period', None)
        guard = kwargs.pop('guard', None)
        params = kwargs.pop('params', None)
        
        # Check if there are any unexpected kwargs remaining
        if kwargs:
            unexpected = list(kwargs.keys())
            print(f"Warning: Unexpected keyword arguments will be ignored: {unexpected}")
        
        # Handle controller - either from args or create from u
        if len(args) >= 1:
            # Controller was passed as positional argument
            controller = args[0]
            remaining_args = args[1:]
        else:
            # No positional controller, check if we have u
            if u is not None:
                # Ensure u is a tensor, not a string or other type
                if not isinstance(u, torch.Tensor):
                    try:
                        u = torch.tensor(u, dtype=torch.float32)
                    except Exception as e:
                        raise TypeError(f"Could not convert u to tensor: {e}")
                
                # Create controller from u
                timestep_counter = {'t': 0}
                
                def u_controller(x):
                    t = timestep_counter['t']
                    if u.dim() == 3 and t < u.shape[1]:
                        control = u[:, t, :]
                    elif u.dim() == 2:
                        control = u
                    else:
                        control = self.u_nominal(x)
                    timestep_counter['t'] += 1
                    return control
                
                controller = u_controller
            else:
                # Use nominal controller
                controller = self.u_nominal
            remaining_args = ()
        
        # Validate x_init
        if not isinstance(x_init, torch.Tensor):
            try:
                x_init = torch.tensor(x_init, dtype=torch.float32)
            except Exception as e:
                raise TypeError(f"Could not convert x_init to tensor: {e}")
        
        # Call parent's simulate with only the parameters it expects
        return super().simulate(
            x_init, 
            num_steps, 
            controller, 
            controller_period=controller_period,
            guard=guard,
            params=params
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

        # Get parameters - use self attributes which maintain correct shapes
        M = self.M.to(x.device).expand(batch_size, -1)
        D = self.D.to(x.device).expand(batch_size, -1)
        P = self.P_mechanical.to(x.device).expand(batch_size, -1)
        B = self.B_matrix.to(x.device)  # Coupling matrix
        
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
            coupling_sum_1 += B[i, j] * torch.sin(theta[:, j - 1])
        f[:, self.N_NODES - 1 + i, 0] = (P[:, i] - D[:, i] * omega[:, i] - coupling_sum_1) / M[:, i]

        # For omega_i (i = 2, ..., n)
        for i in range(1, self.N_NODES):
            coupling_sum_i = torch.zeros(batch_size).type_as(x)
            # B_1i * sin(theta_1i)
            coupling_sum_i += B[i, 0] * torch.sin(theta[:, i - 1])
            # Sum B_ij * sin(theta_1i - theta_1j), j != i
            for j in range(1, self.N_NODES):
                if i != j:
                    coupling_sum_i += B[i, j] * torch.sin(theta[:, i - 1] - theta[:, j - 1])
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

        M = self.M.to(x.device).expand(batch_size, -1)

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
        K_gain = 1.0  # Controller gain (different from coupling matrix)
        omega = x[:, self.N_NODES - 1:]
        u_nominal = -K_gain * omega
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
        
    def solve_equilibrium(self,
                          tol: float = 1e-10,
                          max_iter: int = 50,
                          verbose: bool = False) -> torch.Tensor:
        """
        Solve  P_i = Σ_j B_ij sin(δ_i−δ_j)  for δ  (ω = 0).

        Fix δ_0 = 0 to remove the rotational degree of freedom, then apply a
        Newton–Raphson iteration.  Returns a tensor  δ*  (N,) in radians.

        Raises RuntimeError if convergence fails.
        """
        N = self.n_machines
        P = torch.as_tensor(self.P_mechanical)          # mechanical inputs
        B = torch.as_tensor(self.B_matrix)   # coupling matrix

        # unknowns: δ[1:]  (set δ0 = 0)
        δ = torch.zeros(N, dtype=P.dtype)

        def mismatch(delta):
            """f(delta) = P - Σ B sin(δ_i-δ_j); returns shape (N-1,)"""
            δ_full = torch.cat((torch.zeros(1, dtype=delta.dtype), delta))
            diff = δ_full[:, None] - δ_full[None, :]
            sin_mat = torch.sin(diff)
            Pe = (B * sin_mat).sum(1)
            return (P - Pe)[1:]              # exclude reference bus 0

        for it in range(max_iter):
            f = mismatch(δ[1:])
            norm_f = f.norm().item()
            if verbose:
                print(f"iter {it}: ‖f‖ = {norm_f:.3e}")
            if norm_f < tol:
                break

            # Jacobian H_ij = ∂f_i/∂δ_j   (i,j ≥ 1)
            δ_full = torch.cat((torch.zeros(1, dtype=δ.dtype), δ[1:]))
            diff = δ_full[:, None] - δ_full[None, :]
            cos_mat = torch.cos(diff)
            H = -B * cos_mat
            # row/col 0 correspond to reference bus → remove
            H = H[1:, 1:]

            # Newton step
            try:
                Δ = torch.linalg.solve(H, f)
            except RuntimeError:
                raise RuntimeError("Jacobian is singular; "
                                   "choose another reference machine.")

            δ[1:] -= Δ

            if torch.isnan(δ).any():
                raise RuntimeError("Newton diverged (NaN encountered)")

        else:
            raise RuntimeError("Equilibrium solver did not converge")

        if verbose:
            print(f"Converged in {it+1} iterations; max |Δ| = {Δ.abs().max():.2e}")

        return δ

    def linearise(self, return_JR: bool = False):
        """
        Linearise around (δ*, ω* = 0) where δ* is either supplied in
        `self.delta_star` or computed via solve_equilibrium().
        """
        if not hasattr(self, "delta_star"):
            self.delta_star = self.solve_equilibrium()

        δeq = torch.as_tensor(self.delta_star)
        N = self.n_machines
        M = torch.as_tensor(self.M)
        D = torch.as_tensor(self.D)
        B = torch.as_tensor(self.B_matrix)  # coupling matrix

        # Hessian of potential at δeq
        cos_mat = torch.cos(δeq[:, None] - δeq[None, :])
        H_δδ = torch.diag((B * cos_mat).sum(1)) - B * cos_mat

        # Hessian of kinetic energy
        H_ωω = torch.diag(M)

        Hess = torch.zeros(2*N, 2*N, dtype=M.dtype)
        Hess[:N, :N] = H_δδ
        Hess[N:, N:] = H_ωω

        # canonical J and damping R
        I = torch.eye(N, dtype=M.dtype)
        J = torch.block_diag(torch.zeros_like(I), I) - \
            torch.block_diag(I, torch.zeros_like(I))
        R = torch.block_diag(torch.zeros_like(I), torch.diag(D))

        A = (J - R) @ Hess
        return (A, J, R) if return_JR else A

    def potential_energy_per_machine(self, delta: torch.Tensor) -> torch.Tensor:
        """
        delta : (B, N) absolute angles
        returns: (B, N) energy of each machine
        """
        diff = delta.unsqueeze(2) - delta.unsqueeze(1)
        U_pair = 0.5 * self.B_matrix.to(delta.device) * (1 - torch.cos(diff))
        return U_pair.sum(2)

    def energy_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Total energy for batch x = [θ_12 … θ_1n, ω_1 … ω_n].
        """
        B = x.shape[0]
        N = self.N_NODES

        delta = self.state_to_absolute_angles(x)             # (B, N)
        omega = x[:, N - 1 :]                                # (B, N)

        # kinetic energy
        M_device = self.M.to(x.device)
        H_kin = 0.5 * (M_device * omega ** 2).sum(1)

        # potential energy
        delta_i = delta.unsqueeze(2)    # (B, N, 1)
        delta_j = delta.unsqueeze(1)    # (B, 1, N)
        diff = delta_i - delta_j        # (B, N, N) via broadcasting
        
        # Use B_matrix (coupling matrix)
        B_device = self.B_matrix.to(x.device)
        if B_device.dim() == 2:
            B_device = B_device.unsqueeze(0)  # (1, N, N)
        
        # Compute potential energy
        cos_diff = torch.cos(diff)
        potential_matrix = B_device * (1 - cos_diff)
        H_pot = 0.5 * potential_matrix.sum(dim=(1, 2))

        return H_kin + H_pot

    @torch.no_grad()
    def collect_random_trajectories(
        self,
        N_traj: int,
        T_steps: int = 40,
        control_excitation: float = 0.0,
        return_derivative: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate open‑loop data for model‑order reduction.
        """
        up, lo = self.state_limits
        x_init = torch.rand(N_traj, self.n_dims) * (up - lo) + lo

        X, U, dX = [], [], []
        u_min, u_max = self.control_limits

        for i in range(N_traj):
            x = x_init[i : i + 1]
            for _ in range(T_steps):
                # random mechanical power perturbation
                if control_excitation > 0:
                    u = torch.rand(1, self.n_controls) * (u_max - u_min) + u_min
                    u = u * control_excitation
                else:
                    u = torch.zeros(1, self.n_controls)

                f = self._f(x, self.nominal_params)
                g = self._g(x, self.nominal_params)
                xdot = f + torch.bmm(g, u.view(1, self.n_controls, 1))

                X.append(x.squeeze(0))
                U.append(u.squeeze(0))
                if return_derivative:
                    # Properly squeeze xdot from (1, n_dims, 1) to (n_dims,)
                    dX.append(xdot.squeeze())

                x = x + self.dt * xdot.squeeze(2)           # Euler step

        out = {"X": torch.stack(X), "U": torch.stack(U)}
        if return_derivative:
            out["dXdt"] = torch.stack(dX)
        return out

    def state_to_absolute_angles(self, x: torch.Tensor) -> torch.Tensor:
        """
        x = [θ_12 … θ_1n, ω]  →  δ = [0, δ_2 … δ_n].

        Returns
        -------
        delta : (B, N) tensor
        """
        B = x.shape[0]
        delta_rel = x[:, : self.N_NODES - 1]
        return torch.cat(
            (torch.zeros(B, 1, device=x.device, dtype=x.dtype), delta_rel), dim=1
        )
    
    def to(self, device):
        """
        Move the system to a specific device (CPU/GPU).
        
        args:
            device: torch device or string ('cpu', 'cuda', etc.)
        
        returns:
            self (for chaining)
        """
        # Convert string to device if needed
        if isinstance(device, str):
            device = torch.device(device)
            
        # Move all tensor attributes to the device
        self.M_inertia = self.M_inertia.to(device)
        self.M = self.M_inertia  # Update alias
        self.D_damping = self.D_damping.to(device)
        self.D = self.D_damping  # Update alias
        self.P_mechanical = self.P_mechanical.to(device)
        self.B_matrix = self.B_matrix.to(device)
        
        if hasattr(self, '_goal_point'):
            self._goal_point = self._goal_point.to(device)
            
        if hasattr(self, 'delta_star'):
            self.delta_star = self.delta_star.to(device)
            
        # Call parent's to method if it exists
        if hasattr(super(), 'to'):
            try:
                super().to(device)
            except:
                # Parent might not have a to method or might handle it differently
                pass
            
        return self