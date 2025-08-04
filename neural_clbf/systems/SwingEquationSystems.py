import torch
import numpy as np
from typing import Tuple, Optional, List, Dict
import math
import logging
from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList

logger = logging.getLogger(__name__)

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
        if controller_dt is None:
            controller_dt = dt  # Use the parameter directly, not self.dt        
        # Create aliases for backward compatibility
        self.M = self.M_inertia
        self.D = self.D_damping
        # Don't create alias for P to avoid conflict
        
        # Verify B_matrix has the correct shape
        assert self.B_matrix.shape == (self.N_NODES, self.N_NODES), \
            f"Coupling matrix has wrong shape: {self.B_matrix.shape}, expected ({self.N_NODES}, {self.N_NODES})"

        self.n_machines = self.N_NODES        # alias used by reducers
        self._specified_goal_point = goal_point

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
        Override parent's simulate to robustly handle 'u' and 'controller' kwargs.
        """
        # Ensure num_steps is an integer (fix for the string error)
        try:
            num_steps = int(num_steps)
        except (ValueError, TypeError) as e:
            raise TypeError(f"num_steps must be convertible to int, got {type(num_steps).__name__}: {num_steps}")
        
        # Extract known arguments
        u = kwargs.pop('u', None)
        controller_period = kwargs.pop('controller_period', None)
        guard = kwargs.pop('guard', None)
        params = kwargs.pop('params', None)
        
        # FIX: Extract 'controller' from kwargs explicitly
        controller_kw = kwargs.pop('controller', None)
        
        # Check for unexpected kwargs (Warning check)
        if kwargs:
            unexpected = list(kwargs.keys())
            if unexpected:
               logger.warning(f"Unexpected keyword arguments will be ignored: {unexpected}")
        
        # Determine the controller: prioritize positional, then kwarg, then u, then nominal
        if len(args) >= 1:
            controller = args[0]
        elif controller_kw is not None:
            controller = controller_kw
        elif u is not None:
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
            controller = self.u_nominal
        
        # Validate x_init
        if not isinstance(x_init, torch.Tensor):
            try:
                x_init = torch.tensor(x_init, dtype=torch.float32)
            except Exception as e:
                raise TypeError(f"Could not convert x_init to tensor: {e}")
        
        # Call parent's simulate (controller must be passed positionally)
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
        Return the drift dynamics. FIX: Corrected sign convention for physical consistency.
        State x = [theta_12, ..., theta_1n, omega_1, ..., omega_n], where theta_1i = delta_1 - delta_i.
        Dynamics: M_i * dot(omega_i) = P_i - D_i*omega_i - Sum_j (B_ij * sin(delta_i - delta_j))
        
        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system.
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1), device=x.device, dtype=x.dtype)

        # Get parameters - use self attributes which maintain correct shapes
        M = self.M.to(x.device).expand(batch_size, -1)
        D = self.D.to(x.device).expand(batch_size, -1)
        P_mech = self.P_mechanical.to(x.device).expand(batch_size, -1)
        B = self.B_matrix.to(x.device)  # Coupling matrix
        
        theta = x[:, :self.N_NODES - 1]
        omega = x[:, self.N_NODES - 1:]

        # Theta dynamics: dot(theta_1i) = omega_1 - omega_i
        for i in range(1, self.N_NODES):
            f[:, i - 1, 0] = omega[:, 0] - omega[:, i]

        # Omega dynamics

        # For omega_1 (i=0)
        # P_elec_1 = Sum_{j>1} B_1j * sin(delta_1 - delta_j) = Sum_{j>1} B_1j * sin(theta_1j)
        i = 0
        coupling_sum_1 = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        for j in range(1, self.N_NODES):
            coupling_sum_1 += B[i, j] * torch.sin(theta[:, j - 1])
        
        # dot(omega_1) = (P_1 - D_1*omega_1 - P_elec_1) / M_1
        f[:, self.N_NODES - 1 + i, 0] = (P_mech[:, i] - D[:, i] * omega[:, i] - coupling_sum_1) / M[:, i]

        # For omega_i (i > 0)
        for i in range(1, self.N_NODES):
            coupling_sum_i = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
            
            # Term j=0 (Reference node 1): B_i1 * sin(delta_i - delta_1) = B_i1 * sin(-theta_1i) = -B_i1 * sin(theta_1i)
            coupling_sum_i += -B[i, 0] * torch.sin(theta[:, i - 1])
            
            # Terms j>0, j!=i
            # delta_i - delta_j = (delta_i - delta_1) + (delta_1 - delta_j) = -theta_1i + theta_1j
            for j in range(1, self.N_NODES):
                if i != j:
                    coupling_sum_i += B[i, j] * torch.sin(theta[:, j - 1] - theta[:, i - 1])
            
            # FIX: Ensure MINUS sign before the coupling sum (P_elec)
            f[:, self.N_NODES - 1 + i, 0] = (P_mech[:, i] - D[:, i] * omega[:, i] - coupling_sum_i) / M[:, i]

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
        
    def solve_equilibrium(self, tol: float = 1e-10, max_iter: int = 50, verbose: bool = False) -> torch.Tensor:
        """
        Delegate to the robust solver
        """
        return self.solve_equilibrium_robust(tol, max_iter, verbose)

    def solve_equilibrium_robust(self, tol=1e-10, max_iter=100, verbose=False):
        '''Robust equilibrium solver with power balance correction and DC initialization.'''
        N = self.N_NODES
        device = self.M.device
        
        P = self.P_mechanical.to(device).clone()
        K = self.B_matrix.to(device)
        
        # FIX: Correct power balance (Crucial for convergence)
        P_sum = P.sum()
        if abs(P_sum) > 1e-6:
            if verbose:
                logger.info(f"Correcting power imbalance: {P_sum:.6f}")
            P = P - P_sum / N
        
        # Initialize guesses (Including DC Power Flow)
        initial_guesses = [torch.zeros(N, device=device)]
        try:
            # DC approximation: B_dc * delta = P
            B_dc_full = torch.diag(K.sum(dim=1)) - K
            B_dc_red = B_dc_full[1:, 1:]
            P_red = P[1:]
            delta_dc_red = torch.linalg.solve(B_dc_red, P_red)
            delta_dc = torch.zeros(N, device=device)
            delta_dc[1:] = delta_dc_red
            initial_guesses.append(delta_dc)
        except:
            initial_guesses.append(0.1 * torch.randn(N, device=device))
        
        best_delta = None
        best_error = float('inf')
        
        # Newton-Raphson Iteration
        for guess_idx, delta_init in enumerate(initial_guesses):
            delta = delta_init.clone()
            
            for iteration in range(max_iter):
                # Power flow equations
                P_calc = torch.zeros(N, device=device)
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            P_calc[i] += K[i, j] * torch.sin(delta[i] - delta[j])
                
                mismatch = P - P_calc
                error = mismatch.norm().item()
                
                if error < tol:
                    if verbose:
                        logger.info(f"Converged with guess {guess_idx} in {iteration} iterations")
                    # Ensure reference is 0
                    return delta - delta[0]
                
                if error < best_error:
                    best_error = error
                    best_delta = delta.clone()
                
                # Newton step
                J_power = torch.zeros(N, N, device=device)
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            cos_ij = torch.cos(delta[i] - delta[j])
                            J_power[i, j] = -K[i, j] * cos_ij
                            J_power[i, i] += K[i, j] * cos_ij
                
                # Remove reference
                J_red = J_power[1:, 1:]
                mismatch_red = mismatch[1:]
                
                try:
                    # Check conditioning
                    cond = torch.linalg.cond(J_red)
                    if cond > 1e12:
                        J_red = J_red + 1e-8 * torch.eye(N-1, device=device)
                    
                    d_delta = torch.linalg.solve(J_red, mismatch_red)
                    
                    # Line search
                    alpha = 1.0
                    for _ in range(10):
                        delta_new = delta.clone()
                        delta_new[1:] += alpha * d_delta
                        
                        P_new = torch.zeros(N, device=device)
                        for i in range(N):
                            for j in range(N):
                                if i != j:
                                    P_new[i] += K[i, j] * torch.sin(delta_new[i] - delta_new[j])
                        
                        if (P - P_new).norm() < error:
                            delta = delta_new
                            break
                        alpha *= 0.5
                        
                except:
                    break
        
        if best_delta is None:
            if verbose:
                logger.warning("No valid solution found, returning zeros")
            return torch.zeros(N, device=device)
            
        if verbose:
            logger.warning(f"Did not fully converge. Best error = {best_error:.2e}")
        # Ensure reference is 0
        return best_delta - best_delta[0]


    def potential_energy_per_machine(self, delta: torch.Tensor) -> torch.Tensor:
        """
        delta : (B, N) absolute angles
        returns: (B, N) energy of each machine
        """
        diff = delta.unsqueeze(2) - delta.unsqueeze(1)
        U_pair = 0.5 * self.B_matrix.to(delta.device) * (1 - torch.cos(diff))
        return U_pair.sum(2)
 
    # -------------------------------------------------------------------------
    #  NEW unified energy routine – drop‑in replacement
    # -------------------------------------------------------------------------
    def energy_function(
        self,
        x: torch.Tensor,
        *,
        relative: bool = True,
    ) -> torch.Tensor:
        """
        Hamiltonian (kinetic + potential) of the n‑machine swing system.

        Parameters
        ----------
        x : (B, n_dims) or (n_dims,) tensor
            State in *relative‑angle* coordinates
            ``x = [θ_12 … θ_1n, ω_1 … ω_n]``.
        relative : bool, default **True**
            • ``True``  – return the **Lyapunov candidate**

                  V_rel = T + (V – V_eq)

              so V_rel(x_eq)=0.

            • ``False`` – return the **absolute Hamiltonian**

                  H = T + V

              so H(x_eq)=T_eq+V_eq = V_eq.

        Notes
        -----
        *The only difference from the old code is the extra keyword and the
        caching of V_eq; the numerical expressions are unchanged.*
        """
        # ---- tensor hygiene --------------------------------------------------
        if x.dim() == 1:
            x = x.unsqueeze(0)
        Bsz, N_tot = x.shape
        assert N_tot == self.N_DIMS, f"expected {self.N_DIMS}-vector, got {N_tot}"
        device, dtype = x.device, x.dtype
        n = self.N_NODES

        θ_rel = x[:, : n - 1]                 # (B, n‑1)
        ω      = x[:, n - 1:]                 # (B, n)

        # ---- kinetic energy --------------------------------------------------
        T = 0.5 * (self.M.to(device, dtype) * ω**2).sum(dim=1)   # (B,)

        # ---- potential energy (count each i<j once) --------------------------
        δ = torch.zeros(Bsz, n, device=device, dtype=dtype)      # δ_1 = 0
        δ[:, 1:] = -θ_rel                                        # δ_i = −θ_1i

        Δ = δ.unsqueeze(2) - δ.unsqueeze(1)                      # (B,n,n)

        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), 1)
        V = ( self.B_matrix.to(device, dtype)[mask] *
              (1.0 - torch.cos(Δ[:, mask])) ).sum(dim=1)         # (B,)

        # ---- subtract equilibrium offset if asked for -----------------------
        if relative:
            if not hasattr(self, "_V_eq_cache"):
                θ_eq = self.goal_point.squeeze()[: n - 1]        # (n‑1,)
                δ_eq = torch.zeros(n, device=device, dtype=dtype)
                δ_eq[1:] = -θ_eq
                Δ_eq = δ_eq.unsqueeze(1) - δ_eq.unsqueeze(0)
                V_eq = ( self.B_matrix.to(device, dtype)[mask] *
                         (1.0 - torch.cos(Δ_eq[mask])) ).sum()
                self._V_eq_cache = V_eq.detach()
            V = V - self._V_eq_cache.to(device)

        return T + V


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
        x = [θ_12 … θ_1n, ω].  θ_1i = δ_1 - δ_i.
        Assuming reference δ_1 = 0. Then δ_i = -θ_1i for i > 1.

        Returns
        -------
        delta : (B, N) tensor
        """
        B = x.shape[0]
        if x.dim() == 1:
            x = x.unsqueeze(0)
            B = 1

        theta_1i = x[:, : self.N_NODES - 1]
        
        delta_1 = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
        delta_i = -theta_1i
        
        return torch.cat((delta_1, delta_i), dim=1)
    
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
        
    # --- Equilibrium and Linearization Fixes (Crucial for SPR) ---

    def compute_equilibrium_point(self):
        '''Compute and cache the true equilibrium point in state coordinates'''
        if hasattr(self, '_equilibrium_point_cached'):
             return self._equilibrium_point_cached

        if hasattr(self, '_specified_goal_point') and self._specified_goal_point is not None:
            self._equilibrium_point_cached = self._specified_goal_point.squeeze()
            return self._equilibrium_point_cached

        if not hasattr(self, 'delta_star') or self.delta_star is None:
            self.delta_star = self.solve_equilibrium_robust()
        
        device = self.M.device
        delta_star = self.delta_star.to(device)

        # Convert absolute angles to relative angles (theta_1i = delta_1 - delta_i)
        # The robust solver ensures delta_1 = 0.
        theta_eq = delta_star[0] - delta_star[1:]
        omega_eq = torch.zeros(self.N_NODES, device=device)
        self._equilibrium_point_cached = torch.cat([theta_eq, omega_eq])
        
        # Verification
        f_eq = self._f(self._equilibrium_point_cached.unsqueeze(0), self.nominal_params).squeeze()
        if f_eq.norm() > 1e-6:
            logger.warning(f"Equilibrium verification error = {f_eq.norm():.2e}")
        
        return self._equilibrium_point_cached

    @property
    def goal_point(self):
        '''Return true equilibrium'''
        return self.compute_equilibrium_point().unsqueeze(0)

    def linearise(self, return_JR=False):
        """
        Compute linearization at equilibrium using autograd for reliability.
        """
        N = self.N_NODES
        n_states = 2 * N - 1
        device = self.M.device
        
        # Ensure we have equilibrium
        if not hasattr(self, 'delta_star') or self.delta_star is None:
            logger.info("Computing equilibrium point...")
            self.delta_star = self.solve_equilibrium()
        
        # Get equilibrium in state coordinates  
        x_eq = self.compute_equilibrium_point()
        
        # Verify equilibrium
        f_eq = self._f(x_eq.unsqueeze(0), self.nominal_params).squeeze()
        eq_error = f_eq.norm().item()
        logger.info(f"    Equilibrium verification: ||f|| = {eq_error:.2e}")
        
        # Use autograd for accurate linearization
        x_eq_grad = x_eq.clone().requires_grad_(True)
        
        def dynamics(x):
            return self._f(x.unsqueeze(0), self.nominal_params).squeeze()
        
        A = torch.autograd.functional.jacobian(dynamics, x_eq_grad)
        if A.dim() > 2:
            A = A.squeeze()
        
        # Check eigenvalues
        try:
            eigvals = torch.linalg.eigvals(A)
            max_real = eigvals.real.max().item()
            logger.info(f"    Linearization max eigenvalue: {max_real:.6f}")
        except Exception as e:
            logger.warning(f"    eigvals failed ({e}); skipping eigenvalue check")
        
        if return_JR:
            # Build J and R matrices inline (they're specific to this system)
            J = torch.zeros(n_states, n_states, device=device)
            R = torch.zeros(n_states, n_states, device=device)
            
            # J matrix for relative coordinates
            # J represents the symplectic structure of the system
            for j in range(1, N):
                J[j-1, N-1] = 1.0      # ∂θ_1j/∂ω_1
                J[j-1, N-1+j] = -1.0   # ∂θ_1j/∂ω_j
                J[N-1, j-1] = -1.0     # ∂ω_1/∂θ_1j
                J[N-1+j, j-1] = 1.0    # ∂ω_j/∂θ_1j
            
            # R matrix (damping)
            D = self.D.to(device)
            for i in range(N):
                R[N-1+i, N-1+i] = D[i]
            
            return A, J, R
        else:
            return A

    def compute_A_matrix(self, scenario=None) -> np.ndarray:
        """
        Compute linearized A matrix (delegates to linearise) and convert to numpy for parent compatibility
        """
        A_tensor = self.linearise(return_JR=False)
        return A_tensor.detach().cpu().numpy()