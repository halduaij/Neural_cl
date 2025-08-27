
# fixed_power_dae_ieee39_bestpractice3.py
# Fully expanded control-affine rewrite of the IEEE-39 (New England) reduced network
# with per-generator mixing of synchronous machine (SG) and PV inverter.
#
# The control inputs are the PV inverter active/reactive power setpoints (per kept bus):
#   u = [ P_sp(1..n), Q_sp(1..n) ]
#
# The system is strictly affine in u:
#   dP_cmd/dt = (P_sp - P_cmd) / T_P
#   dQ_cmd/dt = (Q_sp - Q_cmd) / T_Q
#
# All other dynamics are in the drift f(x). The network algebraics are enforced with a
# differentiable gradient flow on the KCL residual energy (no implicit solves in training).
#
# State layout (n = number of kept generator buses = 10):
#   x = [ δ(1..n), ω(1..n), E'q(1..n), Efd(1..n), Pm(1..n), Pvalve(1..n),
#         V(1..n), θ_rel(1..n-1), P_cmd(1..n), Q_cmd(1..n) ]   ∈ R^{10n-1}
#
# Author: (your project)
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, List

import torch
from torch import Tensor

from control_affine_system import ControlAffineSystem


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def smooth_clamp(x: Tensor, lo: Tensor, hi: Tensor, sharpness: float = 20.0) -> Tensor:
    """
    Two-sided differentiable clamp using sigmoids; behaves like a soft limiter
    that approaches lo/hi outside the band without producing zero gradients.
    """
    s_lo = torch.sigmoid(sharpness * (x - lo))
    s_hi = torch.sigmoid(sharpness * (hi - x))
    w = s_lo * s_hi
    return w * x + (1 - w) * (lo * (x < lo) + hi * (x > hi)).type_as(x)


def complexify(x: Tensor) -> Tensor:
    """Ensure a tensor is complex64."""
    return x.to(dtype=torch.complex64) if x.dtype != torch.complex64 else x


# -----------------------------------------------------------------------------
# Main system
# -----------------------------------------------------------------------------

class IEEE39PVSGControlAffineSystem(ControlAffineSystem):
    """
    Control-affine IEEE-39 reduced-bus system with SG + PV mix per generator bus.

    State x ∈ R^{10n-1} with blocks:
      δ, ω, E'q, Efd, Pm, Pvalve ∈ R^n
      V ∈ R^n
      θ_rel ∈ R^{n-1}   (absolute θ = [0, -θ_rel])
      P_cmd, Q_cmd ∈ R^n

    Control u ∈ R^{2n}:
      u = [ P_sp(1..n), Q_sp(1..n) ]

    Dynamics:
      dx/dt = f(x) + g(x) u

      - SG (scaled by sg_ratio):
          dδ = ω_s (ω - 1)
          dω = (ω_s/(2 H_eff)) [ Pm_eff - Pe_sg - D_eff (ω - 1) ]
          dE'q = (Efd - E'q)/Td0'
          dEfd = (sat(Ka(Vref - V)) - Efd)/Ta
          dPm  = (Pvalve - Pm)/Tt
          dPvalve = (Pref(ω) - Pvalve)/Tg,  Pref = Pref_0 + (1-ω)/R

      - Network gradient flow ODE:
          Φ = 1/2 ||k||², k = KCL mismatch (complex)
          dV     = -kV * ∂Φ/∂V
          dθ_rel = -kθ * ∂Φ/∂θ_rel

      - PV command filters (affine in u):
          dP_cmd = (P_sp - P_cmd)/T_P
          dQ_cmd = (Q_sp - Q_cmd)/T_Q

    Injections at each kept bus:
      I_sg = sg_ratio * (-j/Xd') * (E' - Vc)         (Norton)
      I_pv = pv_ratio * conj(S_cmd) / conj(Vc)       with S_cmd = P_cmd + j Q_cmd

    Loads:
      ZIP with a C¹-smooth constant-power current limiter.
    """

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    def __init__(
        self,
        nominal_params: Dict,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[List[Dict]] = None,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        net_relax_kV: float = 25.0,
        net_relax_kth: float = 25.0,
        avr_limit_softness: float = 20.0,
        pv_TP: float = 0.05,
        pv_TQ: float = 0.05,
    ):
        if device is None:
            device = torch.device("cpu")
        self.dev = device
        self.tdtype = dtype
        self.cdtype = torch.complex64

        if not self.validate_params(nominal_params):
            raise ValueError("Missing required parameters for IEEE39PVSGControlAffineSystem")

        # Sizes
        self.n = int(nominal_params["n_gen"])  # number of kept generator buses
        assert self.n == 10, "This class assumes the 10 kept generator buses of IEEE-39."
        self.NC = 2 * self.n  # control dims

        # Helper to tensor-ize
        def tt(x, dtype=self.tdtype):
            return torch.as_tensor(x, dtype=dtype, device=self.dev)

        # System constants
        self.omega_s = tt(2 * math.pi * 60.0)

        # SG + controls
        self.H = tt(nominal_params["H"])
        self.D = tt(nominal_params["D"])
        self.Xd_prime = tt(nominal_params["Xd_prime"])
        self.Td0_prime = tt(nominal_params["Td0_prime"])

        self.Ka = tt(nominal_params["Ka"])
        self.Ta = tt(nominal_params["Ta"])
        self.R = tt(nominal_params["R"])
        self.Tg = tt(nominal_params["Tg"])
        self.Tt = tt(nominal_params["Tt"])

        # SG/PV mixing
        self.sg_ratio = tt(nominal_params.get("sg_ratio", torch.ones(self.n)))
        self.pv_ratio = 1.0 - self.sg_ratio

        # Reduced network and static loads
        self.Y = complexify(tt(nominal_params["Y"], dtype=torch.complex64))
        self.Vset = tt(nominal_params["Vset"])
        self.PL = tt(nominal_params["PL"])
        self.QL = tt(nominal_params["QL"])

        # ZIP parameters
        zpar = nominal_params.get("zip", {})
        self.kP_P = tt(zpar.get("kP_P", 0.10))
        self.kI_P = tt(zpar.get("kI_P", 0.10))
        self.kZ_P = tt(zpar.get("kZ_P", 0.80))
        self.kP_Q = tt(zpar.get("kP_Q", 0.10))
        self.kI_Q = tt(zpar.get("kI_Q", 0.10))
        self.kZ_Q = tt(zpar.get("kZ_Q", 0.80))

        # CPL limiter
        cpl = nominal_params.get("cpl", {})
        self.CPL_SMOOTH_EPS = float(cpl.get("smooth_eps", 0.06))
        self.PQBRAK_VBREAK = float(cpl.get("vbreak", 0.85))
        self.CPL_Imax_gamma = float(cpl.get("gamma", 1.0 / max(self.PQBRAK_VBREAK, 1e-6)))
        self.V_nom = float(cpl.get("V_nom", 1.0))

        # Derived
        self.M_swing = 2.0 * self.H / self.omega_s

        # AVR refs
        self.Vref = tt(nominal_params.get("Vref", self.Vset))
        self.Pref_0 = tt(nominal_params.get("Pref_0", torch.zeros(self.n)))

        # PV command filters
        self.pv_TP = float(pv_TP)
        self.pv_TQ = float(pv_TQ)

        # Network relaxation gains
        self.kV = float(net_relax_kV)
        self.kth = float(net_relax_kth)

        # AVR soft limit parameters
        self.avr_soft = float(avr_limit_softness)
        self.Efd_min_gain = 0.6
        self.Efd_max_gain = 1.8

        # Equilibrium cache
        self._x_eq: Optional[Tensor] = None

        # Initialize base class
        super().__init__(
            nominal_params,
            dt=dt,
            controller_dt=controller_dt,
            scenarios=scenarios,
            use_linearized_controller=False,
        )

        # Calibrate ZIP and build an initial feasible point
        self._calibrate_zip(self.Vset)
        self._initialize_equilibrium()

    # -------------------------------------------------------------------------
    # ControlAffineSystem API
    # -------------------------------------------------------------------------

    def compute_linearized_controller(self, scenarios=None):
        """Provide benign P, K for optional base functionality."""
        self.P = torch.eye(self.n_dims, dtype=self.tdtype, device=self.dev)
        self.K = torch.zeros(self.n_controls, self.n_dims, dtype=self.tdtype, device=self.dev)

    @property
    def n_dims(self) -> int:
        # 6n (SG) + n (V) + (n-1) (θ_rel) + 2n (PVcmd) = 10n - 1
        return 10 * self.n - 1

    @property
    def n_controls(self) -> int:
        return self.NC

    @property
    def angle_dims(self) -> List[int]:
        # angle-like: δ and θ_rel
        idx = list(range(self.n))  # δ
        base = 6 * self.n + self.n  # start of θ_rel
        idx.extend(base + k for k in range(self.n - 1))
        return idx

    @property
    def state_limits(self) -> Tuple[Tensor, Tensor]:
        up = torch.ones(self.n_dims, dtype=self.tdtype, device=self.dev)
        lo = -up.clone()

        # δ and θ bounds
        for i in self.angle_dims:
            up[i] = math.pi
            lo[i] = -math.pi

        # V in [0.35, 1.6]
        iV0 = 6 * self.n
        up[iV0 : iV0 + self.n] = 1.6
        lo[iV0 : iV0 + self.n] = 0.35

        # ω around 1 ± 0.2
        iω0 = self.n
        up[iω0 : iω0 + self.n] = 1.2
        lo[iω0 : iω0 + self.n] = 0.8

        # E'q, Efd positive-ish
        iEq0 = 2 * self.n
        iEfd0 = 3 * self.n
        up[iEq0 : iEq0 + self.n] = 2.5
        lo[iEq0 : iEq0 + self.n] = 0.0
        up[iEfd0 : iEfd0 + self.n] = 3.0
        lo[iEfd0 : iEfd0 + self.n] = 0.0

        # Pm, Pvalve ranges
        iPm0 = 4 * self.n
        iPv0 = 5 * self.n
        up[iPm0 : iPm0 + self.n] = 2.0
        lo[iPm0 : iPm0 + self.n] = -0.2
        up[iPv0 : iPv0 + self.n] = 2.0
        lo[iPv0 : iPv0 + self.n] = -0.2

        # PV command ranges
        iPc0 = 6 * self.n + self.n + (self.n - 1)
        up[iPc0 : iPc0 + self.n] = 2.0
        lo[iPc0 : iPc0 + self.n] = -0.2
        up[iPc0 + self.n : iPc0 + 2 * self.n] = 2.0
        lo[iPc0 + self.n : iPc0 + 2 * self.n] = -2.0

        return (up, lo)

    @property
    def control_limits(self) -> Tuple[Tensor, Tensor]:
        up = torch.cat(
            [
                2.0 * torch.ones(self.n, device=self.dev, dtype=self.tdtype),
                2.0 * torch.ones(self.n, device=self.dev, dtype=self.tdtype),
            ],
            dim=0,
        )
        lo = torch.cat(
            [
                -0.2 * torch.ones(self.n, device=self.dev, dtype=self.tdtype),
                -2.0 * torch.ones(self.n, device=self.dev, dtype=self.tdtype),
            ],
            dim=0,
        )
        return (up, lo)

    def validate_params(self, params: Dict) -> bool:
        req = [
            "n_gen",
            "H",
            "D",
            "Xd_prime",
            "Td0_prime",
            "Ka",
            "Ta",
            "R",
            "Tg",
            "Tt",
            "Y",
            "Vset",
            "PL",
            "QL",
        ]
        return all(k in params for k in req)

    # -------------------------------------------------------------------------
    # Indexing helpers
    # -------------------------------------------------------------------------

    def _idx(self):
        n = self.n
        iδ0 = 0
        iω0 = iδ0 + n
        iEq0 = iω0 + n
        iEfd0 = iEq0 + n
        iPm0 = iEfd0 + n
        iPv0 = iPm0 + n
        iV0 = iPv0 + n
        iθr0 = iV0 + n
        iPc0 = iθr0 + (n - 1)
        iQc0 = iPc0 + n
        return iδ0, iω0, iEq0, iEfd0, iPm0, iPv0, iV0, iθr0, iPc0, iQc0

    def _split_state(self, x: Tensor):
        """
        Split a batch of states into named blocks.
        x: (B, n_dims)
        returns a tuple of (B,n) tensors (θ_rel is (B,n-1)).
        """
        n = self.n
        iδ0, iω0, iEq0, iEfd0, iPm0, iPv0, iV0, iθr0, iPc0, iQc0 = self._idx()
        δ = x[:, iδ0 : iδ0 + n]
        ω = x[:, iω0 : iω0 + n]
        Eqp = x[:, iEq0 : iEq0 + n]
        Efd = x[:, iEfd0 : iEfd0 + n]
        Pm = x[:, iPm0 : iPm0 + n]
        Pval = x[:, iPv0 : iPv0 + n]
        V = x[:, iV0 : iV0 + n]
        θrel = x[:, iθr0 : iθr0 + (n - 1)]
        Pcmd = x[:, iPc0 : iPc0 + n]
        Qcmd = x[:, iQc0 : iQc0 + n]
        return δ, ω, Eqp, Efd, Pm, Pval, V, θrel, Pcmd, Qcmd

    def _θ_abs(self, θrel: Tensor) -> Tensor:
        """Build absolute bus angles with slack at 0: θ = [0, -θ_rel]."""
        B = θrel.shape[0]
        zero = torch.zeros(B, 1, device=θrel.device, dtype=θrel.dtype)
        return torch.cat([zero, -θrel], dim=1)

    # -------------------------------------------------------------------------
    # Loads (ZIP + C¹ CPL limiter)
    # -------------------------------------------------------------------------

    def _load_currents_zip_cpl(self, V: Tensor, θ: Tensor, PL_eff: Tensor, QL_eff: Tensor) -> Tensor:
        """
        Compute complex load currents at each bus using ZIP + smooth-limited CPL.
        """
        ejθ = torch.exp(1j * θ)
        Vc = V * ejθ  # complex node voltages
        Vnom = torch.full_like(V, fill_value=self.V_nom)

        # Split complex powers
        SP = (self.kP_P * PL_eff) + 1j * (self.kP_Q * QL_eff)
        SI = (self.kI_P * PL_eff) + 1j * (self.kI_Q * QL_eff)
        SZ = (self.kZ_P * PL_eff) + 1j * (self.kZ_Q * QL_eff)

        # Z part: I = Yz * V
        Yz = torch.conj(SZ) / (Vnom**2)
        I_Z = Yz * Vc

        # I part: current magnitude fixed at nominal, aligned with V
        I_I_nom = torch.conj(SI) / Vnom
        I_I = I_I_nom * ejθ

        # P (constant power) with C¹-limited current
        I_unsat = torch.conj(SP) / torch.conj(Vc)
        Imax = self.CPL_Imax_gamma * torch.abs(SP) / max(self.V_nom, 1e-6)
        r = torch.where(Imax > 1e-12, torch.abs(I_unsat) / Imax, torch.zeros_like(Imax))

        eps = self.CPL_SMOOTH_EPS
        scale = torch.ones_like(r)
        hard = r >= (1.0 + eps)
        mid = (r > 1.0) & (~hard)

        # Hard cap
        scale = torch.where(hard, 1.0 / torch.clamp(r, min=1e-6), scale)

        # Smooth Hermite blend
        if mid.any():
            s = (r[mid] - 1.0) / eps
            w = s * s * (3.0 - 2.0 * s)
            scale[mid] = (1.0 - w) + w / torch.clamp(r[mid], min=1e-6)

        I_P = I_unsat * scale
        return I_Z + I_I + I_P

    # -------------------------------------------------------------------------
    # Network residual and energy
    # -------------------------------------------------------------------------

    def _kcl_residual(
        self,
        δ: Tensor,
        Eqp: Tensor,
        V: Tensor,
        θ: Tensor,
        Pcmd: Tensor,
        Qcmd: Tensor,
    ) -> Tensor:
        """
        Complex KCL mismatch at each kept bus:
          k = Y V + I_load - I_sg - I_pv
        """
        ejδ = torch.exp(1j * δ)
        ejθ = torch.exp(1j * θ)
        Vc = V * ejθ

        # Network injection
        Y = self.Y.unsqueeze(0).expand(V.shape[0], -1, -1)
        Inet = torch.bmm(Y, Vc.unsqueeze(2)).squeeze(2)

        # SG Norton current
        Yg = -1j / complexify(self.Xd_prime).unsqueeze(0)
        Eprime = Eqp * ejδ
        I_sg = self.sg_ratio.unsqueeze(0) * (Yg * (Eprime - Vc))

        # PV constant-power current injection
        S_cmd = Pcmd + 1j * Qcmd
        I_pv = self.pv_ratio.unsqueeze(0) * torch.conj(S_cmd) / torch.conj(Vc)

        # ZIP loads
        PL_eff = self.PL_zip.unsqueeze(0).expand_as(V)
        QL_eff = self.QL_zip.unsqueeze(0).expand_as(V)
        I_load = self._load_currents_zip_cpl(V, θ, PL_eff, QL_eff)

        # KCL residual
        return Inet + I_load - I_sg - I_pv

    def _network_energy(self, δ, Eqp, V, θabs, Pcmd, Qcmd) -> Tensor:
        k = self._kcl_residual(δ, Eqp, V, θabs, Pcmd, Qcmd)
        return 0.5 * (k.real.square() + k.imag.square()).sum(dim=1)

    # -------------------------------------------------------------------------
    # ZIP calibration
    # -------------------------------------------------------------------------

    def _calibrate_zip(self, Vset: Tensor):
        """
        Choose PL_zip, QL_zip so that at V=Vset the static load equals the
        Ward-equivalent totals PL, QL.
        """
        denomP = self.kZ_P * Vset**2 + self.kI_P * Vset + self.kP_P
        denomQ = self.kZ_Q * Vset**2 + self.kI_Q * Vset + self.kP_Q
        self.PL_zip = self.PL / torch.clamp(denomP, min=1e-6)
        self.QL_zip = self.QL / torch.clamp(denomQ, min=1e-6)

    # -------------------------------------------------------------------------
    # Initialization (pseudo-PF via gradient flow)
    # -------------------------------------------------------------------------

    def _initialize_equilibrium(self, iters: int = 150, step: float = 1e-2):
        """
        Obtain a KCL-feasible V, θ by minimizing Φ with fixed SG/PV states.
        (Runs once and caches an equilibrium; not required during training.)
        """
        B = 1
        n = self.n

        # Crude initial guesses
        δ = torch.zeros(B, n, device=self.dev, dtype=self.tdtype)
        Eqp = self.Vset.view(1, n).clone()
        ω = torch.ones(B, n, device=self.dev, dtype=self.tdtype)
        Efd = Eqp.clone()
        Pm = torch.clamp(self.PL.sum() / n, min=0.1).repeat(B, n)
        Pval = Pm.clone()
        V = self.Vset.view(1, n).clone()
        θrel = torch.zeros(B, n - 1, device=self.dev, dtype=self.tdtype)
        Pcmd = torch.zeros(B, n, device=self.dev, dtype=self.tdtype)
        Qcmd = torch.zeros(B, n, device=self.dev, dtype=self.tdtype)

        # Optimize V, θ_rel to reduce KCL residual energy
        V.requires_grad_(True)
        θrel.requires_grad_(True)
        opt = torch.optim.SGD([V, θrel], lr=step, momentum=0.9)
        for _ in range(iters):
            opt.zero_grad(set_to_none=True)
            Φ = self._network_energy(δ, Eqp, V, self._θ_abs(θrel), Pcmd, Qcmd).sum()
            Φ.backward()
            opt.step()
            with torch.no_grad():
                V.clamp_(0.6, 1.6)
                θrel.clamp_(-math.pi, math.pi)

        # Compute SG electrical powers at this point
        ejδ = torch.exp(1j * δ)
        ejθ = torch.exp(1j * self._θ_abs(θrel))
        Vc = V * ejθ
        Yg = -1j / complexify(self.Xd_prime).unsqueeze(0)
        Eprime = Eqp * ejδ
        I_sg = self.sg_ratio.unsqueeze(0) * (Yg * (Eprime - Vc))
        S_sg = Vc * torch.conj(I_sg)
        Pe_sg = S_sg.real

        # Set references
        self.Pref_0 = Pe_sg.squeeze(0).detach().clamp(min=0.05)
        self.Vref = V.squeeze(0).detach() + Efd.squeeze(0).detach() / torch.clamp(self.Ka, min=1e-6)
        self.Efd_min = 0.6 * Efd.squeeze(0).detach()
        self.Efd_max = 1.8 * Efd.squeeze(0).detach()

        # Cache an equilibrium state
        self._x_eq = torch.zeros(self.n_dims, dtype=self.tdtype, device=self.dev)
        iδ0, iω0, iEq0, iEfd0, iPm0, iPv0, iV0, iθr0, iPc0, iQc0 = self._idx()
        self._x_eq[iδ0 : iδ0 + n] = δ.squeeze(0)
        self._x_eq[iω0 : iω0 + n] = ω.squeeze(0)
        self._x_eq[iEq0 : iEq0 + n] = Eqp.squeeze(0)
        self._x_eq[iEfd0 : iEfd0 + n] = Efd.squeeze(0)
        self._x_eq[iPm0 : iPm0 + n] = self.Pref_0
        self._x_eq[iPv0 : iPv0 + n] = self.Pref_0
        self._x_eq[iV0 : iV0 + n] = V.squeeze(0)
        self._x_eq[iθr0 : iθr0 + (n - 1)] = θrel.squeeze(0)
        self._x_eq[iPc0 : iPc0 + n] = 0.0
        self._x_eq[iQc0 : iQc0 + n] = 0.0

    @property
    def goal_point(self) -> Tensor:
        if self._x_eq is None:
            self._initialize_equilibrium()
        return self._x_eq.view(1, -1)

    @property
    def u_eq(self) -> Tensor:
        return torch.zeros(1, self.n_controls, dtype=self.tdtype, device=self.dev)

    # -------------------------------------------------------------------------
    # Dynamics maps
    # -------------------------------------------------------------------------

    def _f(self, x: Tensor, params: Dict) -> Tensor:
        """
        Control-independent drift dynamics f(x).
        Returns (B, n_dims, 1).
        """
        B = x.shape[0]
        n = self.n
        iδ0, iω0, iEq0, iEfd0, iPm0, iPv0, iV0, iθr0, iPc0, iQc0 = self._idx()
        δ, ω, Eqp, Efd, Pm, Pval, V, θrel, Pcmd, Qcmd = self._split_state(x)

        # SG electrical power via Norton internal
        ejδ = torch.exp(1j * δ)
        ejθ = torch.exp(1j * self._θ_abs(θrel))
        Vc = V * ejθ
        Yg = -1j / complexify(self.Xd_prime).unsqueeze(0)
        Eprime = Eqp * ejδ
        I_sg = self.sg_ratio.unsqueeze(0) * (Yg * (Eprime - Vc))
        S_sg = Vc * torch.conj(I_sg)
        Pe_sg = S_sg.real

        # SG states
        dδ = self.omega_s * (ω - 1.0)
        H_eff = torch.clamp(self.sg_ratio.unsqueeze(0) * self.H, min=1e-4)
        D_eff = self.sg_ratio.unsqueeze(0) * self.D
        dω = (self.omega_s / (2.0 * H_eff)) * (
            self.sg_ratio.unsqueeze(0) * Pm - Pe_sg - D_eff * (ω - 1.0)
        )
        dEqp = (Efd - Eqp) / self.Td0_prime.unsqueeze(0)

        # AVR with soft limiter
        Vr = self.Ka.unsqueeze(0) * (self.Vref.unsqueeze(0) - V)
        Efd_lo = self.Efd_min.unsqueeze(0)
        Efd_hi = self.Efd_max.unsqueeze(0)
        Vr_sat = smooth_clamp(Vr, Efd_lo, Efd_hi, sharpness=self.avr_soft)
        dEfd = (Vr_sat - Efd) / self.Ta.unsqueeze(0)

        # Turbine/Governor
        dPm = (Pval - Pm) / self.Tt.unsqueeze(0)
        Pref = self.Pref_0.unsqueeze(0) + (1.0 - ω) / torch.clamp(self.R.unsqueeze(0), min=1e-6)
        dPval = (Pref - Pval) / self.Tg.unsqueeze(0)

        # Network gradient flow (V, θ_rel) from energy Φ
        V_req = V.clone().requires_grad_(True)
        θrel_req = θrel.clone().requires_grad_(True)
        Φ = self._network_energy(δ, Eqp, V_req, self._θ_abs(θrel_req), Pcmd, Qcmd)
        gV, gθ = torch.autograd.grad(Φ.sum(), (V_req, θrel_req), create_graph=True, allow_unused=True)
        dV = -self.kV * (gV if gV is not None else torch.zeros_like(V))
        dθrel = -self.kth * (gθ if gθ is not None else torch.zeros_like(θrel))

        # PV command filters (homogeneous part; control enters via g)
        dPcmd = -(Pcmd) / max(self.pv_TP, 1e-4)
        dQcmd = -(Qcmd) / max(self.pv_TQ, 1e-4)

        # Pack
        f = torch.zeros(B, self.n_dims, 1, dtype=self.tdtype, device=self.dev)
        f[:, iδ0 : iδ0 + n, 0] = dδ
        f[:, iω0 : iω0 + n, 0] = dω
        f[:, iEq0 : iEq0 + n, 0] = dEqp
        f[:, iEfd0 : iEfd0 + n, 0] = dEfd
        f[:, iPm0 : iPm0 + n, 0] = dPm
        f[:, iPv0 : iPv0 + n, 0] = dPval
        f[:, iV0 : iV0 + n, 0] = dV
        f[:, iθr0 : iθr0 + (n - 1), 0] = dθrel
        f[:, iPc0 : iPc0 + n, 0] = dPcmd
        f[:, iQc0 : iQc0 + n, 0] = dQcmd
        return f

    def _g(self, x: Tensor, params: Dict) -> Tensor:
        """
        Control matrix g(x). Nonzero only on the dP_cmd and dQ_cmd rows.
        Returns (B, n_dims, n_controls).
        """
        B = x.shape[0]
        n = self.n
        _, _, _, _, _, _, _, _, iPc0, iQc0 = self._idx()

        g = torch.zeros(B, self.n_dims, self.n_controls, dtype=self.tdtype, device=self.dev)
        gainP = 1.0 / max(self.pv_TP, 1e-4)
        gainQ = 1.0 / max(self.pv_TQ, 1e-4)
        for i in range(n):
            g[:, iPc0 + i, i] = gainP         # P_sp(i) acts on dP_cmd(i)
            g[:, iQc0 + i, n + i] = gainQ     # Q_sp(i) acts on dQ_cmd(i)
        return g

    # -------------------------------------------------------------------------
    # Safety masks (optional)
    # -------------------------------------------------------------------------

    def safe_mask(self, x: Tensor) -> Tensor:
        up, lo = self.state_limits
        return torch.all((x <= up) & (x >= lo), dim=1)

    def unsafe_mask(self, x: Tensor) -> Tensor:
        up, lo = self.state_limits
        return torch.any((x > up) | (x < lo), dim=1)

    # -------------------------------------------------------------------------
    # Factory from fixed-power defaults
    # -------------------------------------------------------------------------

    @staticmethod
    def from_fixedpower_defaults(
        dt: float = 0.01,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        sg_ratio: Optional[Tensor] = None,
    ) -> "IEEE39PVSGControlAffineSystem":
        """
        Build the system with nominal IEEE-39 reduced-bus parameters.
        """
        if device is None:
            device = torch.device("cpu")

        Y, PL, QL = _build_ieee39_reduced_Y_and_loads_torch(device=device, dtype=dtype)

        # Parameters (example/nominal values)
        H = torch.tensor([15.15, 21.0, 17.9, 14.3, 13.0, 17.4, 13.2, 12.15, 17.25, 250.0], dtype=dtype, device=device)
        D = torch.tensor([17.3, 11.8, 17.3, 17.3, 17.3, 17.3, 17.3, 17.3, 18.22, 18.22], dtype=dtype, device=device)
        Xd_prime = torch.tensor([0.0697, 0.0310, 0.0531, 0.0436, 0.1320, 0.0500, 0.0490, 0.0570, 0.0570, 0.0060], dtype=dtype, device=device)
        Td0_prime = torch.tensor([6.56, 10.2, 5.70, 5.69, 5.40, 7.30, 5.66, 6.70, 4.79, 7.00], dtype=dtype, device=device)
        Ka = torch.full((10,), 50.0, dtype=dtype, device=device)
        Ta = torch.full((10,), 0.001, dtype=dtype, device=device)
        R = torch.full((10,), 0.05, dtype=dtype, device=device)
        Tg = torch.full((10,), 0.05, dtype=dtype, device=device)
        Tt = torch.full((10,), 2.1, dtype=dtype, device=device)
        Vset = torch.tensor([0.9820, 1.0475, 0.9831, 0.9972, 1.0123, 1.0493, 1.0635, 1.0278, 1.0265, 1.0300], dtype=dtype, device=device)

        nominal = {
            "n_gen": 10,
            "H": H,
            "D": D,
            "Xd_prime": Xd_prime,
            "Td0_prime": Td0_prime,
            "Ka": Ka,
            "Ta": Ta,
            "R": R,
            "Tg": Tg,
            "Tt": Tt,
            "Y": Y,
            "Vset": Vset,
            "PL": PL,
            "QL": QL,
            "zip": {
                "kP_P": 0.10,
                "kI_P": 0.10,
                "kZ_P": 0.80,
                "kP_Q": 0.10,
                "kI_Q": 0.10,
                "kZ_Q": 0.80,
            },
            "cpl": {"vbreak": 0.85, "smooth_eps": 0.06, "gamma": 1.0 / 0.85, "V_nom": 1.0},
        }
        if sg_ratio is not None:
            nominal["sg_ratio"] = sg_ratio.to(device=device, dtype=dtype)

        return IEEE39PVSGControlAffineSystem(nominal, dt=dt, device=device, dtype=dtype)


# -----------------------------------------------------------------------------
# Reduced network builder (Kron/Ward) in torch
# -----------------------------------------------------------------------------

def _build_ieee39_reduced_Y_and_loads_torch(
    *, device: torch.device, dtype: torch.dtype
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Rebuild the reduced-bus admittance matrix Y and Ward-equivalent loads PL, QL.
    """
    branches = [
        (1, 2, 0.0035, 0.0411, 0.6987, 0.0),
        (1, 39, 0.0010, 0.0250, 0.7500, 0.0),
        (2, 3, 0.0013, 0.0151, 0.2572, 0.0),
        (2, 25, 0.0070, 0.0086, 0.1460, 0.0),
        (3, 4, 0.0013, 0.0213, 0.2214, 0.0),
        (3, 18, 0.0011, 0.0133, 0.2138, 0.0),
        (4, 5, 0.0008, 0.0128, 0.1342, 0.0),
        (4, 14, 0.0008, 0.0129, 0.1382, 0.0),
        (5, 6, 0.0002, 0.0026, 0.0434, 0.0),
        (5, 8, 0.0008, 0.0112, 0.1476, 0.0),
        (6, 7, 0.0006, 0.0092, 0.1130, 0.0),
        (6, 11, 0.0007, 0.0082, 0.1389, 0.0),
        (7, 8, 0.0004, 0.0046, 0.0780, 0.0),
        (8, 9, 0.0023, 0.0363, 0.3804, 0.0),
        (9, 39, 0.0010, 0.0250, 1.2000, 0.0),
        (10, 11, 0.0004, 0.0043, 0.0729, 0.0),
        (10, 13, 0.0004, 0.0043, 0.0729, 0.0),
        (13, 14, 0.0009, 0.0101, 0.1723, 0.0),
        (14, 15, 0.0018, 0.0217, 0.3660, 0.0),
        (15, 16, 0.0009, 0.0094, 0.1710, 0.0),
        (16, 17, 0.0007, 0.0089, 0.1342, 0.0),
        (16, 19, 0.0016, 0.0195, 0.3040, 0.0),
        (16, 21, 0.0008, 0.0135, 0.2548, 0.0),
        (16, 24, 0.0003, 0.0059, 0.0680, 0.0),
        (17, 18, 0.0007, 0.0082, 0.1319, 0.0),
        (17, 27, 0.0013, 0.0173, 0.3216, 0.0),
        (21, 22, 0.0008, 0.0140, 0.2565, 0.0),
        (22, 23, 0.0006, 0.0096, 0.1846, 0.0),
        (23, 24, 0.0022, 0.0350, 0.3610, 0.0),
        (25, 26, 0.0032, 0.0323, 0.5130, 0.0),
        (26, 27, 0.0014, 0.0147, 0.2396, 0.0),
        (26, 28, 0.0043, 0.0474, 0.7802, 0.0),
        (26, 29, 0.0057, 0.0625, 1.0290, 0.0),
        (28, 29, 0.0014, 0.0151, 0.2490, 0.0),
        # transformers with taps
        (12, 11, 0.0016, 0.0435, 0.0, 1.006),
        (12, 13, 0.0016, 0.0435, 0.0, 1.006),
        (6, 31, 0.0000, 0.0250, 0.0, 1.070),
        (10, 32, 0.0000, 0.0200, 0.0, 1.070),
        (19, 33, 0.0007, 0.0142, 0.0, 1.070),
        (20, 34, 0.0009, 0.0180, 0.0, 1.009),
        (22, 35, 0.0000, 0.0143, 0.0, 1.025),
        (23, 36, 0.0005, 0.0272, 0.0, 1.000),
        (25, 37, 0.0006, 0.0232, 0.0, 1.025),
        (2, 30, 0.0000, 0.0181, 0.0, 1.025),
        (29, 38, 0.0008, 0.0156, 0.0, 1.025),
        (19, 20, 0.0007, 0.0138, 0.0, 1.060),
    ]
    nbus = 39
    Y = torch.zeros(nbus, nbus, dtype=torch.complex64, device=device)

    for f, t, r, x, b, tap in branches:
        f -= 1
        t -= 1
        y = 0.0 if (r == 0.0 and x == 0.0) else 1.0 / complex(r, x)
        bsh = 1j * (b / 2.0)
        a = 1.0 if (tap == 0.0 or tap is None) else float(tap)
        Y[f, f] += (y + bsh) / (a * a)
        Y[t, t] += (y + bsh)
        Y[f, t] += -y / a
        Y[t, f] += -y / a

    # kept buses generator order:
    keep_buses = [31, 30, 32, 33, 34, 35, 36, 37, 38, 39]
    keep = torch.tensor([b - 1 for b in keep_buses], device=device)
    elim = torch.tensor([i for i in range(nbus) if i not in keep.tolist()], device=device)

    # Kron reduce
    Yaa = Y[keep][:, keep]
    Ybb = Y[elim][:, elim]
    Yab = Y[keep][:, elim]
    Yba = Y[elim][:, keep]
    sol = torch.linalg.solve(Ybb, Yba)
    Yred = Yaa - torch.matmul(Yab, sol)

    # Loads (in MW/MVAr converted to pu on 100 MVA base)
    P_MW = {
        3: 322.0,
        4: 500.0,
        7: 233.8,
        8: 522.0,
        12: 7.5,
        15: 320.0,
        16: 329.0,
        18: 158.0,
        20: 628.0,
        21: 274.0,
        23: 247.5,
        24: 308.6,
        25: 224.0,
        26: 139.0,
        27: 281.0,
        28: 206.0,
        29: 283.5,
        31: 9.2,
        39: 1104.0,
    }
    Q_MVAr = {
        3: 2.4,
        4: 184.0,
        7: 84.0,
        8: 176.0,
        12: 88.0,
        15: 153.0,
        16: 32.3,
        18: 30.0,
        20: 103.0,
        21: 115.0,
        23: 84.6,
        24: -92.0,
        25: 47.2,
        26: 17.0,
        27: 75.5,
        28: 27.6,
        29: 26.9,
        31: 4.6,
        39: 250.0,
    }
    base_MVA = 100.0
    P = torch.zeros(nbus, dtype=dtype, device=device)
    Q = torch.zeros(nbus, dtype=dtype, device=device)
    for k, v in P_MW.items():
        P[k - 1] = v / base_MVA
    for k, v in Q_MVAr.items():
        Q[k - 1] = v / base_MVA

    # Nominal V for kept buses
    Vset = torch.tensor(
        [0.9820, 1.0475, 0.9831, 0.9972, 1.0123, 1.0493, 1.0635, 1.0278, 1.0265, 1.0300],
        dtype=dtype,
        device=device,
    )

    # Ward equivalence at eliminated buses (approx at 1∠0)
    elim_idx = [i for i in range(nbus) if i not in keep.tolist()]
    I_b = P[elim_idx] - 1j * Q[elim_idx]
    I_eq = -torch.matmul(Yab, torch.linalg.solve(Ybb, I_b))
    S_eq = Vset * torch.conj(I_eq)
    S_dir = P[keep] + 1j * Q[keep]
    S_tot = S_eq + S_dir
    PL = S_tot.real.to(dtype)
    QL = S_tot.imag.to(dtype)
    return Yred, PL, QL
