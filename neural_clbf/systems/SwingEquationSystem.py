#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IEEE39HybridSystem - A Control-Affine Model for Neural Lyapunov Training
=========================================================================

This module refactors the detailed IEEE-39 DAE model into a control-affine
ordinary differential equation (ODE) system suitable for gradient-based
learning techniques like Neural Control Lyapunov Functions.

Key Transformations and Features:
---------------------------------
1.  **Control-Affine Structure**: The system is explicitly modeled in the form
    dx/dt = f(x) + g(x)u, where 'x' is the state vector and 'u' are the
    control inputs.

2.  **Hybrid Generation**: Each of the 10 generator buses is modeled with a
    customizable ratio of synchronous generation (SG) and photovoltaic (PV)
    inverter-based generation. The `pv_ratio` parameter controls this mix.
    SG parameters (inertia, damping) are scaled accordingly.

3.  **PV Inverter Control**: The control inputs `u` are the active (P) and
    reactive (Q) power setpoints for the PV inverters at each of the 10 buses.
    This allows for direct control over the renewable generation.

4.  **State Augmentation for Differentiability**: To avoid non-differentiable
    iterative algebraic solvers within the dynamics, the system's state is
    augmented.
    - The dynamics of the PV inverters' power output are modeled as first-order
      lags, making `P_pv` and `Q_pv` state variables.
    - The network's algebraic constraints (the power flow equations) are
      relaxed into fast-timescale differential equations using a singular
      perturbation approach. This adds the bus voltage magnitudes `V` and
      angles `theta` to the state vector, turning the original Differential
      Algebraic Equation (DAE) system into a stiff ODE system.

5.  **Full PyTorch Implementation**: All dynamic computations, including complex
    phasor arithmetic and network equations, are implemented in PyTorch. This
    ensures that the entire model is differentiable, a critical requirement
    for training neural network-based controllers.

6.  **Preservation of Detail**: The detailed models for the synchronous machine,
    AVR, governor, and turbine from the original script are preserved and
    ported to PyTorch. The network structure is built from the original
    IEEE-39 data and converted to PyTorch tensors for use in the dynamics.

The resulting system is a high-fidelity, 98-state, 20-control input model
of the IEEE-39 network, ready for advanced control design and analysis.

"""
import logging
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList

# Configure logging
logger = logging.getLogger(__name__)


class IEEE39HybridSystem(ControlAffineSystem):
    """
    Represents the IEEE-39 bus system with a hybrid of synchronous generators (SG)
    and PV inverters. The system is modeled as a control-affine ODE for use in
    neural network-based control.

    The state vector `x` includes:
    - SG dynamics: relative rotor angles, frequencies, transient EMFs, etc.
    - PV dynamics: active and reactive power outputs.
    - Network dynamics: bus voltage magnitudes and angles.

    The control vector `u` consists of:
    - PV setpoints: active (P) and reactive (Q) power references.
    """

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
    ):
        """
        Initialize the hybrid IEEE-39 system.

        Args:
            nominal_params (Scenario): A dictionary containing the system parameters.
                Must include:
                - 'pv_ratio': A 10-element tensor specifying the fraction of PV
                  generation at each generator bus (0.0 to 1.0).
                - 'T_pv': A 2-element tensor for [P, Q] inverter time constants.
                - 'tau_network': A 2-element tensor for [V, theta] network
                  relaxation time constants.
            dt (float): The simulation timestep.
            controller_dt (Optional[float]): The controller timestep.
        """
        # Run network setup and parameter initialization before calling super().__init__
        self._initialize_system_constants()
        self._setup_parameters(nominal_params)

        super().__init__(
            nominal_params,
            dt=dt,
            controller_dt=controller_dt,
            scenarios=scenarios,
            use_linearized_controller=False,  # We define P and K manually
        )

        # Find and set the equilibrium point (goal point) after initialization
        self.goal_point_x = self._find_equilibrium()
        logger.info(f"System initialized with {self.n_dims} states and {self.n_controls} controls.")
        self.P = torch.eye(self.n_dims)
        self.K = torch.zeros(self.n_controls, self.n_dims)


    def _initialize_system_constants(self):
        """Set up fixed constants and network topology."""
        self.n_gen = 10
        self.base_freq = 60.0
        self.omega_s = 2 * torch.pi * self.base_freq

        # Build network matrices (Y, G, B) using numpy for setup
        Y_np, G_np, B_np, PL_np, QL_np, Pg_target_np, Vset_np = self._build_network_np()
        # Convert to torch tensors for use in dynamics
        self.Y = torch.from_numpy(Y_np).cfloat()
        self.G = torch.from_numpy(G_np).float()
        self.B = torch.from_numpy(B_np).float()
        self.PL_base = torch.from_numpy(PL_np).float()
        self.QL_base = torch.from_numpy(QL_np).float()
        self.Pg_target_total = torch.from_numpy(Pg_target_np).float()
        self.Vset = torch.from_numpy(Vset_np).float()

        # Synchronous Generator Machine Parameters (from original script, as torch tensors)
        self.H_base = torch.tensor([15.15, 21.0, 17.9, 14.3, 13.0, 17.4, 13.2, 12.15, 17.25, 250.0], dtype=torch.float)
        self.D_base = torch.tensor([17.3, 11.8, 17.3, 17.3, 17.3, 17.3, 17.3, 17.3, 18.22, 18.22], dtype=torch.float)
        self.Xd_prime = torch.tensor([0.0697, 0.0310, 0.0531, 0.0436, 0.1320, 0.0500, 0.0490, 0.0570, 0.0570, 0.0060], dtype=torch.float)
        self.Td0_prime = torch.tensor([6.56, 10.2, 5.70, 5.69, 5.40, 7.30, 5.66, 6.70, 4.79, 7.00], dtype=torch.float)

        # AVR & Governor/Turbine Parameters
        self.Ka = torch.full((self.n_gen,), 50.0, dtype=torch.float)
        self.Ta = torch.full((self.n_gen,), 0.001, dtype=torch.float)
        self.R = torch.full((self.n_gen,), 0.05, dtype=torch.float)
        self.Tg = torch.full((self.n_gen,), 0.05, dtype=torch.float)
        self.Tt = torch.full((self.n_gen,), 2.1, dtype=torch.float)
        
        # ZIP Load Model Coefficients
        self.kP_P, self.kI_P, self.kZ_P = 0.10, 0.10, 0.80
        self.kP_Q, self.kI_Q, self.kZ_Q = 0.10, 0.10, 0.80


    def _setup_parameters(self, params: Scenario):
        """Configure parameters based on the provided scenario, including hybrid mix."""
        self.pv_ratio = params["pv_ratio"].float()
        self.sg_ratio = 1.0 - self.pv_ratio

        # Scale SG parameters by their generation ratio.
        # Use a small epsilon to avoid division by zero if a generator is 100% PV.
        self.H = self.H_base * self.sg_ratio.clamp(min=1e-6)
        self.D = self.D_base * self.sg_ratio
        
        # Active power setpoint for SGs
        self.Pg_target_sg = self.Pg_target_total * self.sg_ratio
        # Store slack bus power for equilibrium calculation
        self.pg_slack_total = self.Pg_target_total[0]

        # Inverter and Network dynamics parameters
        self.T_p_pv, self.T_q_pv = params["T_pv"][0], params["T_pv"][1]
        self.tau_V, self.tau_theta = params["tau_network"][0], params["tau_network"][1]

        # Define state and control dimensions
        # SG states: delta_rel (n-1), omega (n), Eq_prime (n), Efd (n), Pm (n), Pvalve (n)
        self.n_sg_states = self.n_gen - 1 + 5 * self.n_gen
        # PV states: P_pv (n), Q_pv (n)
        self.n_pv_states = 2 * self.n_gen
        # Network states: V (n), theta_rel (n-1)
        self.n_network_states = self.n_gen + self.n_gen - 1
        
        self._n_dims = self.n_sg_states + self.n_pv_states + self.n_network_states
        self._n_controls = 2 * self.n_gen

        # Define indices for slicing the state vector, for clarity
        n = self.n_gen
        self.idx_delta_rel = slice(0, n - 1)
        self.idx_omega = slice(n - 1, 2 * n - 1)
        self.idx_Eq_prime = slice(2 * n - 1, 3 * n - 1)
        self.idx_Efd = slice(3 * n - 1, 4 * n - 1)
        self.idx_Pm = slice(4 * n - 1, 5 * n - 1)
        self.idx_Pvalve = slice(5 * n - 1, 6 * n - 1)
        self.idx_P_pv = slice(6 * n - 1, 7 * n - 1)
        self.idx_Q_pv = slice(7 * n - 1, 8 * n - 1)
        self.idx_V = slice(8 * n - 1, 9 * n - 1)
        self.idx_theta_rel = slice(9 * n - 1, 10 * n - 2)
        
        # Define indices for control vector
        self.idx_u_p = slice(0, n)
        self.idx_u_q = slice(n, 2 * n)

    def _find_equilibrium(self) -> torch.Tensor:
        """
        Computes the equilibrium point of the system via power flow.
        This is a complex, non-linear solve performed once at initialization.
        """
        logger.info("Solving for system equilibrium point...")
        # Use the numpy-based power flow from the original script to get initial V, theta
        V_eq_np, theta_eq_np, Pg_used_np, Qg_used_np, _ = self._newton_pf_np(self.Vset.numpy())
        
        V_eq = torch.from_numpy(V_eq_np).float()
        theta_eq = torch.from_numpy(theta_eq_np).float()
        Pg_eq = torch.from_numpy(Pg_used_np).float()
        Qg_eq = torch.from_numpy(Qg_used_np).float()

        # Distribute equilibrium generation between SG and PV based on the ratio
        P_sg_eq = Pg_eq * self.sg_ratio
        Q_sg_eq = Qg_eq * self.sg_ratio
        P_pv_eq = Pg_eq * self.pv_ratio
        Q_pv_eq = Qg_eq * self.pv_ratio

        # Back-calculate the SG state variables at equilibrium
        Vc_eq = V_eq * torch.exp(1j * theta_eq)
        Sg_eq = P_sg_eq + 1j * Q_sg_eq
        
        # Avoid division by zero for buses with 100% PV
        Igen_sg_eq = torch.zeros_like(Vc_eq)
        mask = V_eq > 1e-6
        Igen_sg_eq[mask] = (Sg_eq[mask] / Vc_eq[mask]).conj()
        
        Eprime_eq_phasor = Vc_eq + 1j * self.Xd_prime * Igen_sg_eq
        
        delta_eq = torch.angle(Eprime_eq_phasor)
        Eq_prime_eq = torch.abs(Eprime_eq_phasor)
        
        Efd_eq = Eq_prime_eq.clone()
        Vref_eq = V_eq + Efd_eq / self.Ka
        self.Vref = Vref_eq  # Store for use in dynamics

        Pm_eq = P_sg_eq.clone()
        Pvalve_eq = Pm_eq.clone()
        omega_eq = torch.ones(self.n_gen, dtype=torch.float)
        
        Pref_eq = Pm_eq.clone()
        self.Pref0 = Pref_eq # Store for use in dynamics

        # Assemble the full equilibrium state vector x_eq
        delta_rel_eq = delta_eq[1:] - delta_eq[0]
        theta_rel_eq = theta_eq[1:] - theta_eq[0]

        x_eq = torch.cat([
            delta_rel_eq,
            omega_eq,
            Eq_prime_eq,
            Efd_eq,
            Pm_eq,
            Pvalve_eq,
            P_pv_eq,
            Q_pv_eq,
            V_eq,
            theta_rel_eq
        ])
        
        logger.info("Equilibrium point found.")
        return x_eq.unsqueeze(0)

    @property
    def n_dims(self) -> int:
        return self._n_dims

    @property
    def n_controls(self) -> int:
        return self._n_controls

    @property
    def angle_dims(self) -> List[int]:
        """
        **CRITICAL FIX**: Return indices for ALL angle variables.
        This includes the SG relative rotor angles and the network relative
        bus voltage angles. This is essential for correct state normalization
        in the controller.
        """
        sg_angles = list(range(self.idx_delta_rel.start, self.idx_delta_rel.stop))
        network_angles = list(range(self.idx_theta_rel.start, self.idx_theta_rel.stop))
        return sg_angles + network_angles

    @property
    def goal_point(self):
        return self.goal_point_x.to(self.Y.device)
        
    def to(self, device):
        """Move all tensor attributes to the specified device."""
        super().to(device)
        # Manually move all tensor attributes
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(device))
        return self

    def validate_params(self, params: Scenario) -> bool:
        """Check if the provided parameters are valid."""
        required_keys = ["pv_ratio", "T_pv", "tau_network"]
        for key in required_keys:
            if key not in params:
                logger.error(f"Missing required parameter: {key}")
                return False
        if not (isinstance(params["pv_ratio"], torch.Tensor) and params["pv_ratio"].shape == (self.n_gen,)):
            return False
        return True

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the state limits (upper, lower)."""
        n = self.n_gen
        # A sensible default; can be tuned.
        upper_limits = torch.cat([
            torch.full((n - 1,), 2 * torch.pi),    # delta_rel
            torch.full((n,), 1.1),                 # omega
            torch.full((n,), 3.0),                 # Eq_prime
            torch.full((n,), 5.0),                 # Efd
            torch.full((n,), 1.5 * self.Pg_target_total.abs().max()), # Pm
            torch.full((n,), 1.5 * self.Pg_target_total.abs().max()), # Pvalve
            torch.full((n,), 1.5 * self.Pg_target_total.abs().max()), # P_pv
            torch.full((n,), 1.5 * self.Pg_target_total.abs().max()), # Q_pv
            torch.full((n,), 1.5),                 # V
            torch.full((n - 1,), 2 * torch.pi),    # theta_rel
        ])
        lower_limits = torch.cat([
            torch.full((n - 1,), -2 * torch.pi),   # delta_rel
            torch.full((n,), 0.9),                 # omega
            torch.full((n,), 0.0),                 # Eq_prime
            torch.full((n,), -2.0),                # Efd
            torch.full((n,), -0.5),                # Pm
            torch.full((n,), -0.5),                # Pvalve
            torch.full((n,), -0.5),                # P_pv
            torch.full((n,), -1.5 * self.Pg_target_total.abs().max()), # Q_pv
            torch.full((n,), 0.5),                 # V
            torch.full((n - 1,), -2 * torch.pi),   # theta_rel
        ])
        return lower_limits, upper_limits

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the control limits (upper, lower)."""
        # Allow PV to generate up to its rated power and absorb/provide some Q
        max_p = self.Pg_target_total * self.pv_ratio * 1.2
        max_q = self.Pg_target_total * self.pv_ratio * 0.8
        
        upper_limits = torch.cat([max_p, max_q])
        lower_limits = torch.cat([torch.zeros_like(max_p), -max_q])
        
        return lower_limits, upper_limits

    def safe_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return a mask indicating safe states."""
        lower, upper = self.state_limits
        # Consider safe if within 95% of limits
        safe_lower = lower + 0.05 * (upper - lower)
        safe_upper = upper - 0.05 * (upper - lower)
        
        mask = torch.all(x >= safe_lower, dim=1) & torch.all(x <= safe_upper, dim=1)
        return mask

    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return a mask indicating unsafe states."""
        lower, upper = self.state_limits
        mask = torch.any(x < lower, dim=1) | torch.any(x > upper, dim=1)
        return mask

    def _f(self, x: torch.Tensor, params: Scenario) -> torch.Tensor:
        """
        Computes the control-independent part of the dynamics, dx/dt = f(x).
        This is the core of the ODE model.
        """
        # Ensure parameters from the scenario are loaded
        self._setup_parameters(params)
        
        # Move parameters to the correct device
        self.to(x.device)

        batch_size = x.shape[0]
        n = self.n_gen
        
        # Unpack state vector
        delta_rel = x[:, self.idx_delta_rel]
        omega = x[:, self.idx_omega]
        Eq_prime = x[:, self.idx_Eq_prime]
        Efd = x[:, self.idx_Efd]
        Pm = x[:, self.idx_Pm]
        Pvalve = x[:, self.idx_Pvalve]
        P_pv = x[:, self.idx_P_pv]
        Q_pv = x[:, self.idx_Q_pv]
        V = x[:, self.idx_V]
        theta_rel = x[:, self.idx_theta_rel]

        # Reconstruct absolute angles (reference bus 0 is slack)
        delta0 = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        delta = torch.cat([delta0, delta_rel + delta0], dim=1)
        theta0 = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        theta = torch.cat([theta0, theta_rel + theta0], dim=1)

        # --- Intermediate Calculations (Phasors, Powers) ---
        Vc = V.cfloat() * torch.exp(1j * theta.cfloat())
        Eprime_phasor = Eq_prime.cfloat() * torch.exp(1j * delta.cfloat())

        # SG electrical power injection
        # Use a mask to handle buses that are 100% PV (sg_ratio = 0)
        sg_mask = self.sg_ratio > 1e-6
        I_sg = torch.zeros(batch_size, n, device=x.device, dtype=torch.cfloat)
        Xd_prime_c = self.Xd_prime[sg_mask].cfloat()
        I_sg[:, sg_mask] = (Eprime_phasor[:, sg_mask] - Vc[:, sg_mask]) / (1j * Xd_prime_c)
        S_sg = Vc * I_sg.conj()
        P_sg, Q_sg = S_sg.real, S_sg.imag

        # Total load power (using ZIP model)
        PL_zip = self.PL_base * (self.kZ_P * V**2 + self.kI_P * V + self.kP_P)
        QL_zip = self.QL_base * (self.kZ_Q * V**2 + self.kI_Q * V + self.kP_Q)

        # Total power injection at each bus
        P_inj = P_sg + P_pv - PL_zip
        Q_inj = Q_sg + Q_pv - QL_zip
        S_inj = P_inj + 1j * Q_inj

        # Network current mismatch for network dynamics
        I_inj_calc = (S_inj / Vc).conj()
        I_net = Vc @ self.Y.T
        I_mismatch = I_inj_calc - I_net
        
        # --- Assemble state derivatives `f` ---
        f = torch.zeros(batch_size, self.n_dims, 1, device=x.device, dtype=x.dtype)

        # 1. SG Dynamics
        f[:, self.idx_delta_rel, 0] = self.omega_s * (omega[:, 1:] - omega[:, 0].unsqueeze(1))
        d_omega_dt = (self.omega_s / (2 * self.H)) * (Pm - P_sg - self.D * (omega - 1.0))
        f[:, self.idx_omega, 0] = d_omega_dt
        f[:, self.idx_Eq_prime, 0] = (Efd - Eq_prime) / self.Td0_prime
        f[:, self.idx_Efd, 0] = (self.Ka * (self.Vref - V) - Efd) / self.Ta
        f[:, self.idx_Pm, 0] = (Pvalve - Pm) / self.Tt
        Pref = self.Pref0 - (omega - 1.0) / self.R
        f[:, self.idx_Pvalve, 0] = (Pref - Pvalve) / self.Tg

        # 2. PV Inverter Dynamics (control-independent part)
        f[:, self.idx_P_pv, 0] = -P_pv / self.T_p_pv
        f[:, self.idx_Q_pv, 0] = -Q_pv / self.T_q_pv

        # 3. Network Dynamics (Singular Perturbation)
        # We model d(theta)/dt for the relative angles
        P_mismatch = I_mismatch.real
        Q_mismatch = I_mismatch.imag
        f[:, self.idx_V, 0] = Q_mismatch / self.tau_V
        f[:, self.idx_theta_rel, 0] = (P_mismatch[:, 1:] - P_mismatch[:, 0].unsqueeze(1)) / self.tau_theta

        return f

    def _g(self, x: torch.Tensor, params: Scenario) -> torch.Tensor:
        """
        Computes the control-dependent part of the dynamics, g(x).
        """
        batch_size = x.shape[0]
        g = torch.zeros(batch_size, self.n_dims, self.n_controls, device=x.device, dtype=x.dtype)

        # The control inputs u = [P_ref_pv, Q_ref_pv] only affect the PV dynamics.
        # d(P_pv_i)/dt = (u_p_i - P_pv_i) / T_p  => g_p = 1/T_p
        # d(Q_pv_i)/dt = (u_q_i - Q_pv_i) / T_q  => g_q = 1/T_q
        
        # Create indices for diagonal assignment
        p_rows = torch.arange(self.idx_P_pv.start, self.idx_P_pv.stop)
        p_cols = torch.arange(self.idx_u_p.start, self.idx_u_p.stop)
        
        q_rows = torch.arange(self.idx_Q_pv.start, self.idx_Q_pv.stop)
        q_cols = torch.arange(self.idx_u_q.start, self.idx_u_q.stop)

        g[:, p_rows, p_cols] = 1.0 / self.T_p_pv
        g[:, q_rows, q_cols] = 1.0 / self.T_q_pv

        return g

    # =========================================================================
    # Helper functions ported from the original script for initialization
    # These use numpy and are only called once.
    # =========================================================================
    def _build_network_np(self) -> Tuple:
        """Builds the IEEE-39 reduced network using numpy."""
        base_MVA = 100.0
        nbus = 39
        branches = [
            (1, 2, 0.0035, 0.0411, 0.6987, 0.0), (1, 39, 0.001, 0.025, 0.75, 0.0),
            (2, 3, 0.0013, 0.0151, 0.2572, 0.0), (2, 25, 0.007, 0.0086, 0.146, 0.0),
            (3, 4, 0.0013, 0.0213, 0.2214, 0.0), (3, 18, 0.0011, 0.0133, 0.2138, 0.0),
            (4, 5, 0.0008, 0.0128, 0.1342, 0.0), (4, 14, 0.0008, 0.0129, 0.1382, 0.0),
            (5, 6, 0.0002, 0.0026, 0.0434, 0.0), (5, 8, 0.0008, 0.0112, 0.1476, 0.0),
            (6, 7, 0.0006, 0.0092, 0.113, 0.0), (6, 11, 0.0007, 0.0082, 0.1389, 0.0),
            (7, 8, 0.0004, 0.0046, 0.078, 0.0), (8, 9, 0.0023, 0.0363, 0.3804, 0.0),
            (9, 39, 0.001, 0.025, 1.2, 0.0), (10, 11, 0.0004, 0.0043, 0.0729, 0.0),
            (10, 13, 0.0004, 0.0043, 0.0729, 0.0), (13, 14, 0.0009, 0.0101, 0.1723, 0.0),
            (14, 15, 0.0018, 0.0217, 0.366, 0.0), (15, 16, 0.0009, 0.0094, 0.171, 0.0),
            (16, 17, 0.0007, 0.0089, 0.1342, 0.0), (16, 19, 0.0016, 0.0195, 0.304, 0.0),
            (16, 21, 0.0008, 0.0135, 0.2548, 0.0), (16, 24, 0.0003, 0.0059, 0.068, 0.0),
            (17, 18, 0.0007, 0.0082, 0.1319, 0.0), (17, 27, 0.0013, 0.0173, 0.3216, 0.0),
            (21, 22, 0.0008, 0.014, 0.2565, 0.0), (22, 23, 0.0006, 0.0096, 0.1846, 0.0),
            (23, 24, 0.0022, 0.035, 0.361, 0.0), (25, 26, 0.0032, 0.0323, 0.513, 0.0),
            (26, 27, 0.0014, 0.0147, 0.2396, 0.0), (26, 28, 0.0043, 0.0474, 0.7802, 0.0),
            (26, 29, 0.0057, 0.0625, 1.029, 0.0), (28, 29, 0.0014, 0.0151, 0.249, 0.0),
            (12, 11, 0.0016, 0.0435, 0.0, 1.006), (12, 13, 0.0016, 0.0435, 0.0, 1.006),
            (6, 31, 0.0, 0.025, 0.0, 1.07), (10, 32, 0.0, 0.02, 0.0, 1.07),
            (19, 33, 0.0007, 0.0142, 0.0, 1.07), (20, 34, 0.0009, 0.018, 0.0, 1.009),
            (22, 35, 0.0, 0.0143, 0.0, 1.025), (23, 36, 0.0005, 0.0272, 0.0, 1.0),
            (25, 37, 0.0006, 0.0232, 0.0, 1.025), (2, 30, 0.0, 0.0181, 0.0, 1.025),
            (29, 38, 0.0008, 0.0156, 0.0, 1.025), (19, 20, 0.0007, 0.0138, 0.0, 1.06),
        ]
        Y = np.zeros((nbus, nbus), dtype=complex)
        for f, t, r, x, b, tap in branches:
            f -= 1; t -= 1
            y = 1.0 / complex(r, x) if (r != 0 or x != 0) else 0.0
            bsh = 1j * (b / 2.0)
            a = 1.0 if tap == 0.0 else float(tap)
            Y[f, f] += (y + bsh) / (a * a); Y[t, t] += y + bsh
            Y[f, t] -= y / a; Y[t, f] -= y / a

        gen_bus_order = [31, 30, 32, 33, 34, 35, 36, 37, 38, 39]
        keep = [b - 1 for b in gen_bus_order]
        elim = [i for i in range(nbus) if i not in keep]
        Yaa, Ybb = Y[np.ix_(keep, keep)], Y[np.ix_(elim, elim)]
        Yab, Yba = Y[np.ix_(keep, elim)], Y[np.ix_(elim, keep)]
        Yred = Yaa - Yab @ np.linalg.solve(Ybb, Yba)

        P_MW = {3:322.0, 4:500.0, 7:233.8, 8:522.0, 12:7.5, 15:320.0, 16:329.0,
                18:158.0, 20:628.0, 21:274.0, 23:247.5, 24:308.6, 25:224.0,
                26:139.0, 27:281.0, 28:206.0, 29:283.5, 31:9.2, 39:1104.0}
        Q_MVAr = {3:2.4, 4:184.0, 7:84.0, 8:176.0, 12:88.0, 15:153.0, 16:32.3,
                  18:30.0, 20:103.0, 21:115.0, 23:84.6, 24:-92.0, 25:47.2,
                  26:17.0, 27:75.5, 28:27.6, 29:26.9, 31:4.6, 39:250.0}
        P, Q = np.zeros(nbus), np.zeros(nbus)
        for k, v in P_MW.items(): P[k - 1] = v / base_MVA
        for k, v in Q_MVAr.items(): Q[k - 1] = v / base_MVA

        Vset = np.array([0.982, 1.0475, 0.9831, 0.9972, 1.0123, 1.0493, 1.0635, 1.0278, 1.0265, 1.03])
        I_b = (P[elim] - 1j * Q[elim])
        I_eq = -Yab @ np.linalg.solve(Ybb, I_b)
        S_eq = Vset * np.conj(I_eq)
        S_dir = P[keep] + 1j * Q[keep]
        S_tot = S_eq + S_dir
        PL, QL = S_tot.real, S_tot.imag

        gen_P_MW = {30: 250.0, 32: 650.0, 33: 632.0, 34: 508.0, 35: 650.0,
                    36: 560.0, 37: 540.0, 38: 830.0, 39: 1000.0}
        Pg_pu = np.zeros(10)
        for idx, bus in enumerate(gen_bus_order):
            if bus in gen_P_MW: Pg_pu[idx] = gen_P_MW[bus] / base_MVA
        
        return Yred, Yred.real, Yred.imag, PL, QL, Pg_pu, Vset

    def _newton_pf_np(self, Vset, slack=0, tol=1e-7, itmax=20):
        """Numpy-based Newton-Raphson power flow for initialization."""
        n = self.n_gen
        P_spec = self.Pg_target_total.numpy() - self.PL_base.numpy()
        theta, V = np.zeros(n), Vset.copy()
        
        for it in range(itmax):
            Vc = V * np.exp(1j * theta)
            S_inj = Vc * np.conj(self.Y.numpy() @ Vc)
            P, Q = S_inj.real, S_inj.imag
            
            mis_P = P_spec - P
            mis_V = Vset**2 - V**2
            
            # Simplified Decoupled Power Flow Jacobian
            J11 = -self.B.numpy()
            J22 = -self.B.numpy()
            
            # Solve for PV and PQ buses
            pvpq_idx = list(range(1, n))
            J11_red = J11[np.ix_(pvpq_idx, pvpq_idx)]
            J22_red = J22[np.ix_(pvpq_idx, pvpq_idx)] # Assuming all are PV for init
            
            d_theta = np.linalg.solve(J11_red, mis_P[pvpq_idx] / V[pvpq_idx])
            d_V = np.linalg.solve(J22_red, mis_V[pvpq_idx] / V[pvpq_idx])
            
            theta[pvpq_idx] += d_theta
            V[pvpq_idx] += d_V
            
            if max(np.abs(d_theta).max(), np.abs(d_V).max()) < tol:
                break
        
        Vc = V * np.exp(1j * theta)
        S_inj = Vc * np.conj(self.Y.numpy() @ Vc)
        Pg_used = S_inj.real + self.PL_base.numpy()
        Qg_used = S_inj.imag + self.QL_base.numpy()
        
        return V, theta, Pg_used, Qg_used, None # bus_type not needed here


if __name__ == '__main__':
    # Define nominal parameters for the system
    # Let's assume a 20% PV penetration at each generator bus
    pv_penetration = 0.20
    nominal_params: Scenario = {
        "pv_ratio": torch.full((10,), pv_penetration),
        "T_pv": torch.tensor([0.05, 0.05]),      # P, Q inverter time constants (s)
        "tau_network": torch.tensor([0.01, 0.01]), # V, theta network dynamics (s)
    }

    # Instantiate the system
    try:
        ieee39_system = IEEE39HybridSystem(nominal_params=nominal_params)
        
        # Print some information about the system
        print("=" * 50)
        print("IEEE-39 Hybrid System Instantiation Test")
        print("=" * 50)
        print(f"Number of state dimensions: {ieee39_system.n_dims}")
        print(f"Number of control dimensions: {ieee39_system.n_controls}")
        
        # Check the equilibrium point
        x_eq = ieee39_system.goal_point
        print(f"Equilibrium point shape: {x_eq.shape}")
        
        # Test the dynamics at the equilibrium point (should be near zero)
        f_eq = ieee39_system._f(x_eq, nominal_params)
        print(f"Norm of f(x_eq): {torch.norm(f_eq).item():.4e}")
        
        # Test a single simulation step
        print("\nTesting one simulation step from equilibrium...")
        x_next = ieee39_system.closed_loop_dynamics(x_eq, ieee39_system.u_nominal(x_eq))
        print(f"Norm of dx/dt at equilibrium: {torch.norm(x_next).item():.4e}")

        # Test a random point
        print("\nTesting dynamics at a random point...")
        lower, upper = ieee39_system.state_limits
        x_rand = torch.rand(1, ieee39_system.n_dims) * (upper - lower) + lower
        f_rand = ieee39_system._f(x_rand, nominal_params)
        g_rand = ieee39_system._g(x_rand, nominal_params)
        print(f"f(x_rand) shape: {f_rand.shape}")
        print(f"g(x_rand) shape: {g_rand.shape}")
        
        print("\nSystem instantiated successfully!")

    except Exception as e:
        print(f"\nAn error occurred during instantiation: {e}")
        import traceback
        traceback.print_exc()
