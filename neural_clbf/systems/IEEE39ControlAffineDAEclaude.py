#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IEEE39ControlAffineDAE — Control‑affine PyTorch port of the IEEE‑39 DAE

This module implements a control‑affine swing‑equation system with PV states
while preserving the algebraic network constraints (V, theta) via a differentiable
Newton solve in PyTorch. It is designed to be a closer match to a DAE model:
- PV powers are states driven by inputs (to keep u entering affinely).
- Bus voltages/angles are NOT states; they are solved from KCL each RHS call.
- ZIP loads are calibrated at the equilibrium to match Ward loads.
- u_eq is set to PV equilibrium to avoid PV state drift at the goal point.
"""

from typing import Optional, Tuple, List, Dict
import math
import numpy as np
import torch
from torch import Tensor
from torch.autograd.functional import jacobian

# Import your base class
from .control_affine_system import ControlAffineSystem
def _as_1d(v: torch.Tensor) -> torch.Tensor:
    return v if v.dim() == 1 else v.reshape(-1)


def _to_torch_complex(a: np.ndarray, dtype=torch.complex64) -> Tensor:
    """Convert numpy array (possibly complex) to a torch complex tensor."""
    if np.iscomplexobj(a):
        real = torch.tensor(a.real, dtype=torch.float32)
        imag = torch.tensor(a.imag, dtype=torch.float32)
        return torch.complex(real, imag).to(dtype)
    else:
        return torch.tensor(a, dtype=torch.float32).to(dtype)


class IEEE39ControlAffineDAE(ControlAffineSystem):
    """Control‑affine PyTorch implementation of the IEEE‑39 swing/AVR/gov system
    with PV states and an algebraic (V, theta) network solve at each RHS call.
    """


    def __init__(
        self,
        nominal_params: Dict = None,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[Dict] = None,
    ) -> None:


        import inspect, sys
        print("[EQ-DEBUG] module:", __name__)
        print("[EQ-DEBUG] file  :", inspect.getfile(self.__class__))
        print("[EQ-DEBUG] has _project_to_consistency:", hasattr(self, "_project_to_consistency"))
        print("[EQ-DEBUG] solve_equilibrium code hash:",
            hash(inspect.getsource(self._solve_equilibrium)))

        import inspect
        print("[EQ-DEBUG] _solve_equilibrium code hash:", hash(inspect.getsource(self._solve_equilibrium)))
        print("[EQ-DEBUG] _solve_kcl_newton code hash :", hash(inspect.getsource(self._solve_kcl_newton)))
        print("[EQ-DEBUG] _angle_reconstruct code hash:", hash(inspect.getsource(self._angle_reconstruct)))

        if nominal_params is None:
            nominal_params = {}

        super().__init__(
            nominal_params=nominal_params,
            dt=dt,
            controller_dt=controller_dt,
            use_linearized_controller=False,
            scenarios=scenarios,
        )

        # ----- core sizes -----
        self.n_gen = 10

        # ----- reduced network and static data -----
        (Yred_np, G_np, B_np, PL_base_np, QL_base_np, Pg_target_np, Vset_np) = (
            self._build_reduced_network_and_loads()
        )
        self.Y: Tensor = _to_torch_complex(Yred_np)      # (n,n) complex
        self.G: Tensor = torch.tensor(G_np, dtype=torch.float32)
        self.B: Tensor = torch.tensor(B_np, dtype=torch.float32)
        self.PL_base: Tensor = torch.tensor(PL_base_np, dtype=torch.float32)
        self.QL_base: Tensor = torch.tensor(QL_base_np, dtype=torch.float32)
        self.Pg_target_total: Tensor = torch.from_numpy(Pg_target_np).float()
        self.Vset: Tensor = torch.from_numpy(Vset_np).float()

        # ----- machine & controller parameters (from DAE best‑practice) -----
        H = np.array([15.15, 21.0, 17.9, 14.3, 13.0, 17.4, 13.2, 12.15, 17.25, 250.0], dtype=float)
        D = np.array([17.3, 11.8, 17.3, 17.3, 17.3, 17.3, 17.3, 17.3, 18.22, 18.22], dtype=float)
        Xd_prime = np.array([0.0697, 0.0310, 0.0531, 0.0436, 0.1320, 0.0500, 0.0490, 0.0570, 0.0570, 0.0060], dtype=float)
        Td0_prime = np.array([6.56, 10.2, 5.70, 5.69, 5.40, 7.30, 5.66, 6.70, 4.79, 7.00], dtype=float)
        self.H_base: Tensor = torch.from_numpy(H).float()
        self.D_base: Tensor = torch.from_numpy(D).float()
        self.Xd_prime: Tensor = torch.from_numpy(Xd_prime).float()
        self.Td0_prime: Tensor = torch.from_numpy(Td0_prime).float()

        # AVR and governor/turbine (uniform here; adjust if needed)
        self.Ka: Tensor = torch.full((self.n_gen,), 50.0, dtype=torch.float32)
        self.Ta: Tensor = torch.full((self.n_gen,), 0.001, dtype=torch.float32)
        self.R:  Tensor = torch.full((self.n_gen,), 0.05,  dtype=torch.float32)
        self.Tg: Tensor = torch.full((self.n_gen,), 0.05,  dtype=torch.float32)
        self.Tt: Tensor = torch.full((self.n_gen,), 2.10,  dtype=torch.float32)

        # ----- PV proportions and PV first‑order lags (for control‑affine) -----
        pv_ratio = nominal_params.get("pv_ratio", np.zeros(self.n_gen, dtype=float))
        pv_ratio = np.asarray(pv_ratio, dtype=float)
        if pv_ratio.shape != (self.n_gen,):
            raise ValueError("pv_ratio must be length-10 for the 10 generator buses.")
        self.pv_ratio: Tensor = torch.from_numpy(pv_ratio).float().clamp(0.0, 1.0)
        self.sg_ratio: Tensor = (1.0 - self.pv_ratio).float()

        T_pv = nominal_params.get("T_pv", (0.05, 0.05))  # (Tp, Tq) in seconds
        self.Tp_pv: Tensor = torch.tensor(float(T_pv[0]), dtype=torch.float32)
        self.Tq_pv: Tensor = torch.tensor(float(T_pv[1]), dtype=torch.float32)

        # Effective inertia/damping if part of generation is PV
        eps = 1e-6
        self.H: Tensor = self.H_base * self.sg_ratio.clamp(min=eps)
        self.D: Tensor = self.D_base * self.sg_ratio

        # ZIP fractions; we calibrate sP,sQ after PF to match PL_base, QL_base at V*
        self.kZ_P, self.kI_P, self.kP_P = 0.80, 0.10, 0.10
        self.kZ_Q, self.kI_Q, self.kP_Q = 0.80, 0.10, 0.10

        # base frequency and ω_s
        self.base_freq = 60.0
        self.omega_s = 2.0 * math.pi * self.base_freq

        # ----- equilibrium solve & references -----
        (
            V_eq, theta_eq, Pg_eq, Qg_eq,
            P_sg_eq, Q_sg_eq, P_pv_eq, Q_pv_eq,
            delta_eq, Eqp_eq, Efd_eq, Pm_eq, Pvalve_eq, Pref0
        ) = self._solve_equilibrium()

        # ZIP calibration so ZIP(V_eq) == (PL_base, QL_base)
     #   self.sP, self.sQ = self._calibrate_zip(V_eq, P_ref=self.PL_base, Q_ref=self.QL_base)

        # Equilibrium control for PV states (prevents PV drift at the goal point)
        self._u_eq = torch.cat([P_pv_eq, Q_pv_eq]).unsqueeze(0)  # shape (1, n_controls)


        # Goal state (relative angles)
        self._goal_point = self._pack_state(
            delta_rel=delta_eq[1:] - delta_eq[0],
            omega=torch.ones(self.n_gen),
            Eqp=Eqp_eq,
            Efd=Efd_eq,
            Pm=Pm_eq,
            Pvalve=Pvalve_eq,
            Ppv=P_pv_eq,
            Qpv=Q_pv_eq,
        ).unsqueeze(0)  

        # Dimensions
        self._n_dims = (self.n_gen - 1) + 5*self.n_gen + 2*self.n_gen  # no network states (algebraic)
        self._n_controls = 2 * self.n_gen



    # inside __init__
        self.kcl_row_drop = self._select_best_drop_row()


        # Reasonable state limits
        self._x_lo = torch.cat([
            -math.pi*torch.ones(self.n_gen-1),  # delta_rel
            0.9*torch.ones(self.n_gen),         # omega ~ 1 pu
            0.0*torch.ones(self.n_gen),         # Eqp
            0.0*torch.ones(self.n_gen),         # Efd
            0.0*torch.ones(self.n_gen),         # Pm
            0.0*torch.ones(self.n_gen),         # Pvalve
            -2.0*torch.ones(self.n_gen),        # Ppv
            -2.0*torch.ones(self.n_gen),        # Qpv
        ])
        self._x_hi = torch.cat([
            math.pi*torch.ones(self.n_gen-1),
            1.1*torch.ones(self.n_gen),
            5.0*torch.ones(self.n_gen),
            5.0*torch.ones(self.n_gen),
            5.0*torch.ones(self.n_gen),
            5.0*torch.ones(self.n_gen),
            2.0*torch.ones(self.n_gen),
            2.0*torch.ones(self.n_gen),
        ])

    
    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (upper_u_lim, lower_u_lim) for controls u = [uP, uQ].
        We allow PV active power setpoints up to 120% of their rated PV share and
        reactive setpoints up to 80% in magnitude (symmetric).
        """
        max_p = self.Pg_target_total * self.pv_ratio * 1.2
        max_q = self.Pg_target_total * self.pv_ratio * 0.8
        upper = torch.cat([max_p,  max_q])
        lower = torch.cat([torch.zeros_like(max_p), -max_q])
        return upper, lower

    def safe_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Conservative "safe" region: inside 95% of state limits on all dims."""
        upper, lower = self.state_limits
        safe_lower = lower + 0.05 * (upper - lower)
        safe_upper = upper - 0.05 * (upper - lower)
        return torch.all(x >= safe_lower, dim=1) & torch.all(x <= safe_upper, dim=1)

    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Unsafe if outside the hard state limits on any dimension."""
        upper, lower = self.state_limits
        too_low  = torch.any(x < lower, dim=1)
        too_high = torch.any(x > upper, dim=1)
        return too_low | too_high
# ---------------- Interface required by ControlAffineSystem ----------------

    def _select_best_drop_row(self):
        # test ('real',0) vs ('imag',0) on the goal; pick the smaller masked ||F||
        candidates = [('real',0), ('imag',0)]
        best = None; best_val = float('inf')
        delta_rel, _, Eqp, _, _, _, Ppv, Qpv = self._unpack_state(self.goal_point[0])
        delta = self._angle_reconstruct(delta_rel)
        for kind,bus in candidates:
            F_full = self._kcl_residual(torch.cat([torch.zeros(self.n_gen-1), self.Vset]),
                                        delta, Eqp, Ppv, Qpv)
            mask = torch.ones(2*self.n_gen, dtype=torch.bool)
            mask[(bus if kind=='real' else self.n_gen + bus)] = False
            val = float(F_full[mask].norm())
            if val < best_val: best_val, best = val, (kind,bus)
        return best
    def validate_params(self, params: Dict) -> bool:
        return True

    @property
    def n_dims(self) -> int:
        return self._n_dims

    @property
    def n_controls(self) -> int:
        return self._n_controls

    @property
    def angle_dims(self) -> List[int]:
        # Only rotor relative angles are angles; network angles are algebraic
        return list(range(0, self.n_gen-1))
    @property
    def goal_point(self) -> torch.Tensor:
        gp = self._goal_point  # store as either (n_dims,) or (1, n_dims)
        return gp if gp.dim() == 2 else gp.unsqueeze(0)


    @property
    def state_limits(self) -> Tuple[Tensor, Tensor]:
        # Base expects (upper, lower)
        return self._x_hi, self._x_lo

    # ---------------- packing and helpers ----------------

    def _pack_state(self, *, delta_rel: Tensor, omega: Tensor, Eqp: Tensor, Efd: Tensor,
                    Pm: Tensor, Pvalve: Tensor, Ppv: Tensor, Qpv: Tensor) -> Tensor:
        return torch.cat([delta_rel, omega, Eqp, Efd, Pm, Pvalve, Ppv, Qpv], dim=-1)

    def _unpack_state(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        n = self.n_gen
        i0 = 0
        delta_rel = x[..., i0:i0+n-1]; i0 += (n-1)
        omega     = x[..., i0:i0+n];   i0 += n
        Eqp       = x[..., i0:i0+n];   i0 += n
        Efd       = x[..., i0:i0+n];   i0 += n
        Pm        = x[..., i0:i0+n];   i0 += n
        Pvalve    = x[..., i0:i0+n];   i0 += n
        Ppv       = x[..., i0:i0+n];   i0 += n
        Qpv       = x[..., i0:i0+n];   i0 += n
        return delta_rel, omega, Eqp, Efd, Pm, Pvalve, Ppv, Qpv
    def _angle_reconstruct(self, delta_rel: Tensor) -> Tensor:
        delta0 = getattr(self, "delta0_goal", torch.zeros((), dtype=delta_rel.dtype, device=delta_rel.device))
        if delta0.dim() == 0:
            delta0 = delta0.view(1)
        return torch.cat([delta0, delta0 + delta_rel], dim=-1)

    def _sg_mask(self) -> Tensor:
        return (self.sg_ratio > 0.0)

    # ---------------- core dynamics f(x) ----------------
    def _f(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Batched control-independent dynamics f(x).
        Runs the algebraic KCL solve per batch element (B loops), keeping autograd.
        """
        device = x.device
        n = self.n_gen
        B = x.shape[0]

        # Unpack (batched)
        delta_rel, omega, Eqp, Efd, Pm, Pvalve, Ppv, Qpv = self._unpack_state(x)

        f_list = []
        for b in range(B):
            delta_rel_b = delta_rel[b]                # (n-1,)
            omega_b     = omega[b]                    # (n,)
            Eqp_b       = Eqp[b]                      # (n,)
            Efd_b       = Efd[b]
            Pm_b        = Pm[b]
            Pvalve_b    = Pvalve[b]
            Ppv_b       = Ppv[b]
            Qpv_b       = Qpv[b]

            delta_b = self._angle_reconstruct(delta_rel_b)  # (n,)

            # Algebraic solve (1-D inputs, 1-D outputs), square Newton like DAE
            V_b, theta_b = self._solve_kcl_newton(delta_b, Eqp_b, Ppv_b, Qpv_b)

            Vc_b     = V_b * torch.exp(1j * theta_b)         # (n,)
            Eprime_b = Eqp_b * torch.exp(1j * delta_b)       # (n,)

            # Norton generator currents where SG exists
            sg_mask = self._sg_mask()
            I_sg_b = torch.zeros(n, dtype=torch.complex64, device=device)
            if sg_mask.any():
                Xd_c = self.Xd_prime[sg_mask].to(device).cfloat()
                I_sg_b[sg_mask] = (Eprime_b[sg_mask] - Vc_b[sg_mask]) / (1j * Xd_c)

            S_sg_b = Vc_b * I_sg_b.conj()
            Pe_b   = S_sg_b.real

            # AVR (unclipped here; optional soft limits below)
            Vref = getattr(self, "Vref", self.Vset.to(device))
            Vr_b = self.Ka.to(device) * (Vref - V_b.real)

            # Optional: soft Efd limits (set enable_efd_soft_clip, Efd_min/max in __init__)
            if getattr(self, "enable_efd_soft_clip", False):
                Efd_min = getattr(self, "Efd_min", torch.zeros(n, device=device))
                Efd_max = getattr(self, "Efd_max", 5.0*torch.ones(n, device=device))
                beta_efd = getattr(self, "beta_efd", 20.0)
                # smooth clip target before lag:
                Vr_b = Vr_b + 0.0  # Vr_b drives Efd; we clip Efd in its state rate below
                def softclip(v, lo, hi, beta):
                    # tanh-smoothing, C^1, matches lo/hi asymptotically
                    return lo + (hi - lo) * (torch.tanh(beta*( (v-lo)/(hi-lo) - 0.5 )) + 1)/2
                Efd_target = softclip(Efd_b, Efd_min, Efd_max, beta_efd)
                dEfd_b = (Efd_target - Efd_b) / self.Ta.to(device)
            else:
                dEfd_b = (self.Ka.to(device)*(Vref - V_b.real) - Efd_b) / self.Ta.to(device)

            # Swing + AVR + turbine-governor
            domega_b = (self.omega_s / (2.0 * self.H.to(device))) * (Pm_b - Pe_b - self.D.to(device)*(omega_b - 1.0))
            ddelta_b = omega_b - omega_b[..., :1]  # relative to machine 1
            dEqp_b = (Efd_b - Eqp_b) / self.Td0_prime.to(device)
            dPm_b  = (Pvalve_b - Pm_b) / self.Tt.to(device)

            Pref0 = getattr(self, "Pref0", Pm_b.detach().clone())
            Pref_b  = Pref0 + (1.0 - omega_b) / self.R.to(device)
            dPval_b = (Pref_b - Pvalve_b) / self.Tg.to(device)

            f_b = torch.cat([
                ddelta_b[1:], domega_b, dEqp_b, dEfd_b, dPm_b, dPval_b,
                torch.zeros_like(Ppv_b),      # PV states are driven by inputs via g(x)u
                torch.zeros_like(Qpv_b),
            ], dim=-1)
            f_list.append(f_b)

        return torch.stack(f_list, dim=0)  # (B, n_dims)

    # ---------------- control matrix g(x) ----------------
    def _g(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Batched control matrix: shape (B, n_dims, n_controls)
        Only PV rows are nonzero (1/Tp and 1/Tq).
        """
        B = x.shape[0]
        G = x.new_zeros(B, self.n_dims, self.n_controls)
        row_P = (self.n_gen - 1) + 5*self.n_gen
        row_Q = row_P + self.n_gen
        Tp = float(self.Tp_pv); Tq = float(self.Tq_pv)
        for i in range(self.n_gen):
            G[:, row_P + i, i]              = 1.0 / Tp
            G[:, row_Q + i, self.n_gen + i] = 1.0 / Tq
        return G

    # ---------------- algebraic network solve ----------------
    def _zip_currents(self, V: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        ZIP load currents at each bus (complex), using calibrated sP, sQ.
        Correct current laws:
        I_Z = ((Pz - jQz)/V) e^{jθ},  I_I = ((Pi - jQi)/V) e^{jθ},  I_P = ((Pp - jQp)/V) e^{jθ}
        """
        ejth = torch.exp(1j * theta)
        V_safe = torch.clamp(V, min=1e-4)

        # voltage-dependent powers at this V
        Pz = self.sP * self.kZ_P * self.PL_base * (V**2)
        Pi = self.sP * self.kI_P * self.PL_base * V
        Pp = self.sP * self.kP_P * self.PL_base

        Qz = self.sQ * self.kZ_Q * self.QL_base * (V**2)
        Qi = self.sQ * self.kI_Q * self.QL_base * V
        Qp = self.sQ * self.kP_Q * self.QL_base

        I_Z = ((Pz - 1j*Qz) / V_safe) * ejth
        I_I = ((Pi - 1j*Qi) / V_safe) * ejth   # ← THIS was missing
        I_P = ((Pp - 1j*Qp) / V_safe) * ejth

        return I_Z + I_I + I_P



    def _pf_theta_only(self, Vset: torch.Tensor, P_spec: torch.Tensor, itmax: int = 40, tol: float = 1e-10):
        """
        Solve the PV-style power flow with fixed |V|=Vset (all kept buses are generators),
        unknown angles θ_2..θ_n from active-power equations:
            P_inj_net_i(Vset, θ) = P_spec_i,  for i=2..n (slack i=1 holds θ_1=0).
        Returns θ (n,) with θ_1=0.
        """
        n = self.n_gen
        device = Vset.device
        G = self.Y.real.to(torch.float64).to(device)
        B = self.Y.imag.to(torch.float64).to(device)
        V = Vset.to(torch.float64)

        theta = torch.zeros(n, dtype=torch.float64, device=device)
        for _ in range(itmax):
            # Compute P_inj_net at current theta
            c = torch.cos(theta[:, None] - theta[None, :])
            s = torch.sin(theta[:, None] - theta[None, :])
            P_inj = V * torch.sum( V[None, :] * (G*c + B*s), dim=1 )  # (n,)

            # Mismatch (ignore slack bus 1)
            mis = P_inj[1:] - P_spec.to(torch.float64)[1:]            # (n-1,)

            # Converged?
            if torch.norm(mis).item() < tol:
                break

            # Jacobian H ≈ ∂P/∂θ for i,j>1
            H = torch.zeros(n-1, n-1, dtype=torch.float64, device=device)
            for i in range(1, n):
                for j in range(1, n):
                    if i == j:
                        # dP_i/dθ_i = - V_i * sum_j V_j (G_ij sinθ_ij - B_ij cosθ_ij)
                        H[i-1, j-1] = -V[i] * torch.sum( V * ( G[i,:]*s[i,:] - B[i,:]*c[i,:] ) )
                    else:
                        # dP_i/dθ_j =  V_i * V_j ( G_ij sinθ_ij - B_ij cosθ_ij )
                        H[i-1, j-1] =  V[i] * V[j] * ( G[i,j]*s[i,j] - B[i,j]*c[i,j] )

            # Newton step
            try:
                dtheta = torch.linalg.solve(H, -mis)
            except RuntimeError:
                dtheta = torch.linalg.pinv(H, rcond=1e-12) @ (-mis)

            theta[1:] = theta[1:] + dtheta

        return theta.to(Vset.dtype)
    def _project_to_consistency(self, delta, Eqp, Efd, Pm, Pvalve, Ppv, Qpv, max_passes=2):
        for _ in range(max_passes):
            # Algebraic solve at current states
            V, theta = self._solve_kcl_newton(delta, Eqp, Ppv, Qpv)
            Vc = V * torch.exp(1j*theta)

            # S_net and S_zip at (V,theta)
            I_net = (self.Y.to(Vc.dtype) @ Vc)
            S_net = Vc * I_net.conj()
            Pz = self.sP * self.kZ_P * self.PL_base * (V**2)
            Pi = self.sP * self.kI_P * self.PL_base * V
            Pp = self.sP * self.kP_P * self.PL_base
            Qz = self.sQ * self.kZ_Q * self.QL_base * (V**2)
            Qi = self.sQ * self.kI_Q * self.QL_base * V
            Qp = self.sQ * self.kP_Q * self.QL_base
            S_zip = (Pz + Pi + Pp) + 1j*(Qz + Qi + Qp)

            # Total generation and split
            S_gen = S_net + S_zip.to(S_net.dtype)
            S_pv  = (self.pv_ratio * S_gen.real).to(S_gen.dtype) \
                + 1j*(self.pv_ratio * S_gen.imag).to(S_gen.dtype)
            S_sg  = S_gen - S_pv

            # **Project PV states** to match S_pv at this (V,theta)
            Ppv = S_pv.real.float().clone()
            Qpv = S_pv.imag.float().clone()

            # Rebuild E' so Norton matches S_sg
            I_sg = torch.zeros_like(Vc, dtype=Vc.dtype)
            mask = self.sg_ratio > 0
            if mask.any():
                Xd_c = self.Xd_prime[mask].to(Eqp.dtype).to(Vc.dtype)
                I_sg[mask] = (S_sg[mask] / Vc[mask]).conj()
            Eprime = Vc.clone()
            Eprime[mask] = Vc[mask] + 1j*Xd_c*I_sg[mask]
            delta = torch.angle(Eprime).float()
            Eqp   = torch.abs(Eprime).float()

            # Fast channels exact at this point
            Efd     = Eqp.clone()
            self.Vref = V.real + Efd / self.Ka
            Pm      = S_sg.real.float().clone()
            Pvalve  = Pm.clone()

        # **Keep u_eq consistent with the projected PV states**
        self._u_eq = torch.cat([Ppv, Qpv]).unsqueeze(0)
        # Optionally, update the cached goal point here too

        rep0 = self.audit_pf_at(V, theta, delta, Eqp, Ppv, Qpv)
        print("[EQ-DEBUG] proj BEFORE: KCL", rep0["k_norm"], "dPmax", rep0["dS_P_max"], "dQmax", rep0["dS_Q_max"])
        # ... your split + state updates ...
        rep1 = self.audit_pf_at(V, theta, delta, Eqp, Ppv, Qpv)
        print("[EQ-DEBUG] proj AFTER : KCL", rep1["k_norm"], "dPmax", rep1["dS_P_max"], "dQmax", rep1["dS_Q_max"])

        self._goal_point = self._pack_state(
            delta_rel=delta[1:] - delta[0],
            omega=torch.ones(self.n_gen),
            Eqp=Eqp, Efd=Efd, Pm=Pm, Pvalve=Pvalve, Ppv=Ppv, Qpv=Qpv
        ).unsqueeze(0)



        # at the end of _project_to_consistency
        self._last_V = V.detach().clone()
        self._last_theta = theta.detach().clone()
        self._V_goal = self._last_V.clone()         # for your debug print
        self._theta_goal = self._last_theta.clone()
        # inside _project_to_consistency, just before return
        print("[EQ-DEBUG][proj] ||sP||,||sQ||:", float(self.sP.norm()), float(self.sQ.norm()))
        print("[EQ-DEBUG][proj] ||Y||_F:", float(torch.linalg.norm(torch.view_as_real(self.Y))))
        print("[EQ-DEBUG][proj] ||Eqp||,||Pm||:", float(Eqp.norm()), float(Pm.norm()))

        # at the very beginning of _solve_kcl_newton
        print("[EQ-DEBUG][runtime] ||sP||,||sQ||:", float(self.sP.norm()), float(self.sQ.norm()))
        print("[EQ-DEBUG][runtime] ||Y||_F:", float(torch.linalg.norm(torch.view_as_real(self.Y))))

        return delta, Eqp, Efd, Pm, Pvalve, Ppv, Qpv



    def _solve_equilibrium(self):
        """
        Build equilibrium ensuring AVR references match the actual Newton-solved voltage.
        """
        n = self.n_gen
        device = self.Vset.device
        
        # 1. Calibrate ZIP
        self.sP, self.sQ = self._calibrate_zip(self.Vset, P_ref=self.PL_base, Q_ref=self.QL_base)
        
        # 2. Initial guess: flat angles, nominal voltages
        theta_eq = torch.zeros(n, dtype=self.Vset.dtype, device=device)
        V_eq = self.Vset.clone()
        
        # 3. Compute generation at this point (your existing code)
        Vc_eq = V_eq * torch.exp(1j * theta_eq)
        I_net = (self.Y.to(torch.complex64) @ Vc_eq.to(torch.complex64))
        S_net = Vc_eq.to(torch.complex64) * I_net.conj()
        
        # ZIP loads
        Pz = self.sP * self.kZ_P * self.PL_base * (V_eq**2)
        Pi = self.sP * self.kI_P * self.PL_base * V_eq
        Pp = self.sP * self.kP_P * self.PL_base
        Qz = self.sQ * self.kZ_Q * self.QL_base * (V_eq**2)
        Qi = self.sQ * self.kI_Q * self.QL_base * V_eq
        Qp = self.sQ * self.kP_Q * self.QL_base
        S_load = (Pz + Pi + Pp) + 1j*(Qz + Qi + Qp)
        
        S_gen = S_net + S_load.to(torch.complex64)
        
        # Split SG/PV
        sg = self.sg_ratio.to(torch.float32)
        pv = self.pv_ratio.to(torch.float32)
        S_sg_eq = (sg * S_gen.real).to(torch.complex64) + 1j*(sg * S_gen.imag).to(torch.complex64)
        S_pv_eq = (pv * S_gen.real).to(torch.complex64) + 1j*(pv * S_gen.imag).to(torch.complex64)
        
        P_sg_eq = S_sg_eq.real.to(torch.float32)
        Q_sg_eq = S_sg_eq.imag.to(torch.float32)
        P_pv_eq = S_pv_eq.real.to(torch.float32)
        Q_pv_eq = S_pv_eq.imag.to(torch.float32)
        
        # Build E'
        I_sg_eq = torch.zeros_like(Vc_eq, dtype=torch.complex64)
        mask = self.sg_ratio > 0
        I_sg_eq[mask] = (S_sg_eq[mask] / Vc_eq[mask]).conj()
        Eprime_eq = Vc_eq.to(torch.complex64)
        Eprime_eq[mask] = Vc_eq[mask].to(torch.complex64) + 1j*self.Xd_prime[mask].to(torch.complex64)*I_sg_eq[mask]
        
        delta_eq = torch.angle(Eprime_eq).float()
        Eqp_eq = torch.abs(Eprime_eq).float()
        
        # 4. CRITICAL: Find the actual (V, θ) that Newton will use with these states
        V_actual, theta_actual = self._solve_kcl_newton(delta_eq, Eqp_eq, P_pv_eq, Q_pv_eq)
        
        # 5. Set AVR reference using the ACTUAL voltage Newton finds
        Efd_eq = Eqp_eq.clone()
        self.Vref = V_actual + Efd_eq / self.Ka  # <-- KEY FIX: use V_actual, not V_eq!
        
        Pm_eq = P_sg_eq.clone()
        Pvalve_eq = Pm_eq.clone()
        self.Pref0 = Pm_eq.clone()
        
        # 6. Cache the actual Newton solution
        self._last_V = V_actual.detach().clone()
        self._last_theta = theta_actual.detach().clone()
        self._V_goal = V_actual.clone()  
        self._theta_goal = theta_actual.clone()
        
        # 7. Skip or minimize _project_to_consistency since we already have consistency
        
        # Build goal point
        self._goal_point = self._pack_state(
            delta_rel=delta_eq[1:] - delta_eq[0],
            omega=torch.ones(n),
            Eqp=Eqp_eq,
            Efd=Efd_eq,
            Pm=Pm_eq,
            Pvalve=Pvalve_eq,
            Ppv=P_pv_eq,
            Qpv=Q_pv_eq
        ).unsqueeze(0)
        
        self._u_eq = torch.cat([P_pv_eq, Q_pv_eq]).unsqueeze(0)
        
        Pg_eq = (P_sg_eq + P_pv_eq)
        Qg_eq = (Q_sg_eq + Q_pv_eq)
        
        return (V_actual, theta_actual, Pg_eq, Qg_eq,
                P_sg_eq, Q_sg_eq, P_pv_eq, Q_pv_eq,
                delta_eq, Eqp_eq, Efd_eq, Pm_eq, Pvalve_eq, self.Pref0)

    def _pv_current(self, Ppv: torch.Tensor, Qpv: torch.Tensor,
                    V: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        PV injection as a current source: I = conj(S / Vc), where Vc = V * e^{jθ}.
        """
        V_safe = torch.clamp(V, min=1e-4)
        Vc_safe = V_safe * torch.exp(1j * theta)
        S = (Ppv + 1j * Qpv).to(Vc_safe.dtype)
        return torch.conj(S / Vc_safe)   # <-- critical: conj(S / Vc), not S/|V|


    def _kcl_residual(self, z: torch.Tensor, delta: torch.Tensor, Eqp: torch.Tensor, Ppv: torch.Tensor, Qpv: torch.Tensor) -> torch.Tensor:
        # force 1-D views
        delta = _as_1d(delta); Eqp = _as_1d(Eqp); Ppv = _as_1d(Ppv); Qpv = _as_1d(Qpv)
        n = self.n_gen
        theta_rel = z[:n-1]
        V = z[n-1:]
        theta = torch.cat([torch.zeros(1, device=z.device, dtype=z.dtype), theta_rel], dim=0)

        Vc = V * torch.exp(1j * theta)
        Eprime = Eqp * torch.exp(1j * delta)

        sg_mask = self._sg_mask()
        I_sg = torch.zeros(n, dtype=torch.complex64, device=z.device)
        if sg_mask.any():
            Xd_c = self.Xd_prime[sg_mask].to(z.device).cfloat()
            I_sg[sg_mask] = (Eprime[sg_mask] - Vc[sg_mask]) / (1j * Xd_c)

        I_net  = self.Y.to(z.device) @ Vc
        I_load = self._zip_currents(V, theta)
        I_pv   = self._pv_current(Ppv.to(z.device), Qpv.to(z.device), V, theta)

        k = I_net + I_load - I_sg - I_pv
        return torch.cat([k.real, k.imag], dim=0)  # (2n,)
    def _solve_kcl_newton(
        self, delta: torch.Tensor, Eqp: torch.Tensor, Ppv: torch.Tensor, Qpv: torch.Tensor,
        tol: float = 1e-8, itmax: int = 25
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Newton solve for KCL=0 with square Jacobian by dropping one redundant KCL row
        (angle-gauge). Unknowns: z = [theta_rel(=theta[1:]); V], size 2n-1.
        Residual F stacks real/imag KCL at all buses (size 2n); we drop one row so J is square.
        """
        # 1-D views (autograd-safe)
        delta = _as_1d(delta); Eqp = _as_1d(Eqp); Ppv = _as_1d(Ppv); Qpv = _as_1d(Qpv)
        device = Eqp.device
        n = self.n_gen

        # warm starts (not used for grads)
        V0 = getattr(self, "_last_V", self.Vset).to(device)
        th0 = getattr(self, "_last_theta", torch.zeros(n, device=device, dtype=V0.dtype))
        z = torch.cat([th0[1:], V0], dim=0).clone().requires_grad_(True)



        # --- inside _solve_kcl_newton, replace the LS loop with this square-Newton loop ---
        drop_kind, drop_bus = getattr(self, "kcl_row_drop", ("imag", 0))
        n = self.n_gen
        drop_idx = (drop_bus if drop_kind == "real" else n + drop_bus)

        mask = torch.ones(2*n, dtype=torch.bool, device=device)
        mask[drop_idx] = False  # remove exactly one redundant KCL row

        for it in range(itmax):
            F_full = self._kcl_residual(z, delta, Eqp, Ppv, Qpv)     # (2n,)
            F      = F_full[mask]                                     # (2n-1,)

            # Square Jacobian: mask rows after autograd
            J_full = torch.autograd.functional.jacobian(
                lambda zz: self._kcl_residual(zz, delta, Eqp, Ppv, Qpv), z, create_graph=True
            )                                                         # (2n, 2n-1)
            J = J_full[mask, :]                                       # (2n-1, 2n-1)

            # Newton step for the square system
            try:
                dz = torch.linalg.solve(J, -F)
            except RuntimeError:
                dz = torch.linalg.lstsq(J, -F, rcond=1e-12).solution

            # Armijo backtracking on the masked norm
            F0 = torch.norm(F)
            alpha = 1.0
            for _ in range(6):
                z_try   = z + alpha*dz
                V_try   = torch.clamp(z_try[n-1:], 0.35, 1.60)
                z_trial = torch.cat([z_try[:n-1], V_try], dim=0)

                F_trial_full = self._kcl_residual(z_trial, delta, Eqp, Ppv, Qpv)
                F_trial      = F_trial_full[mask]

                if torch.norm(F_trial) <= (1.0 - 1e-3*alpha) * F0:
                    z = z_trial
                    break
                alpha *= 0.5
            else:
                z = z + dz  # fall-back if line search fails

            if torch.norm(F_trial) < tol:
                break
        # --- end replacement ---
        print(f"[EQ-DEBUG][newton] drop=({drop_kind},{drop_bus}), "
            f"J_shape={J.shape}, ||F_masked||={float(F_trial.norm()):.3e}, "
            f"||F_full||={float(F_trial_full.norm()):.3e}")


        theta_rel = z[:n-1]
        V = torch.clamp(z[n-1:], 0.35, 1.60)
        theta = torch.cat([torch.zeros(1, device=device, dtype=V.dtype), theta_rel], dim=0)



        # Consistency sanity (remove once stable)
        I_pv_runtime = self._pv_current(Ppv, Qpv, V, theta)
        S_pv = (Ppv + 1j*Qpv).to(I_pv_runtime.dtype)
        Vc   = V * torch.exp(1j*theta)
        I_pv_audit = torch.conj(S_pv / Vc)
        print("[EQ-DEBUG] max|I_pv(runtime) - I_pv(audit)|:",
            float((I_pv_runtime - I_pv_audit).abs().max()))

        # cache warm starts (detach = hints only)
        self._last_V = V.detach()
        self._last_theta = theta.detach()
        return V, theta

    # ---------------- initialization: network, PF, ZIP calibration ----------------

    def _build_reduced_network_and_loads(self):
        """
        Build the full 39‑bus Y‑bus, reduce to the 10 generator buses by Ward/Kron,
        and assemble reduced loads (PL, QL) and generator targets.
        Returns: (Yred, G, B, PL, QL, Pg_target, Vset)
        """
        base_MVA = 100.0
        nbus = 39
        # Branch data: (from_bus, to_bus, R, X, Bsh, tap)
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

        # Build full Y‑bus
        Y = np.zeros((nbus, nbus), dtype=complex)
        for f, t, r, x, b, tap in branches:
            f -= 1; t -= 1
            y = 0.0 if (r == 0.0 and x == 0.0) else 1.0 / complex(r, x)
            bsh = 1j * (b / 2.0)
            a = 1.0 if tap == 0.0 else float(tap)
            Y[f, f] += (y + bsh) / (a * a); Y[t, t] += y + bsh
            Y[f, t] -= y / a; Y[t, f] -= y / a

        # Reduce to generator buses by Ward/Kron
        gen_bus_order = [31, 30, 32, 33, 34, 35, 36, 37, 38, 39]
        keep = [b - 1 for b in gen_bus_order]
        elim = [i for i in range(nbus) if i not in keep]
        Yaa, Ybb = Y[np.ix_(keep, keep)], Y[np.ix_(elim, elim)]
        Yab, Yba = Y[np.ix_(keep, elim)], Y[np.ix_(elim, keep)]
        Yred = Yaa - Yab @ np.linalg.solve(Ybb, Yba)

        # Aggregate loads onto kept buses
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

        # Generator P targets (MW -> pu on base_MVA) for non‑slack units
        gen_P_MW = {30: 250.0, 32: 650.0, 33: 632.0, 34: 508.0, 35: 650.0,
                    36: 560.0, 37: 540.0, 38: 830.0, 39: 1000.0}
        Pg_pu = np.zeros(10)
        for idx, bus in enumerate(gen_bus_order):
            if bus in gen_P_MW: Pg_pu[idx] = gen_P_MW[bus] / base_MVA

        return Yred, Yred.real, Yred.imag, PL, QL, Pg_pu, Vset

    def _newton_pf_numpy(self, Vset: np.ndarray, tol: float = 1e-7, itmax: int = 20):
        """Lightweight decoupled PF to get a reasonable (V*, theta*) and (Pg, Qg)."""
        n = self.n_gen
        P_spec = self.Pg_target_total.numpy() - self.PL_base.numpy()
        theta = np.zeros(n); V = Vset.copy()

        for _ in range(itmax):
            Vc = V * np.exp(1j * theta)
            Y = self.Y.numpy()
            S_inj = Vc * np.conj(Y @ Vc)  # power injection at kept buses
            P, Q = S_inj.real, S_inj.imag

            mis_P = P_spec - P
            mis_V = Vset**2 - V**2

            # crude decoupled step; good enough for initial guess
            dtheta = 0.1 * mis_P[1:]
            dV = 0.5 * mis_V / np.maximum(V, 1e-3)

            theta[1:] += dtheta
            V += dV
            if np.linalg.norm(np.concatenate([mis_P[1:], mis_V])) < tol:
                break

        Pg = P_spec.copy()
        Qg = Q + self.QL_base.numpy()
        return V, theta, Pg, Qg, 0.0
    def _calibrate_zip(self, V_ref, P_ref, Q_ref):
        V = V_ref.detach()
        denomP = (self.kZ_P*V**2 + self.kI_P*V + self.kP_P) * torch.clamp(self.PL_base, min=1e-8)
        denomQ = (self.kZ_Q*V**2 + self.kI_Q*V + self.kP_Q) * torch.clamp(self.QL_base.abs(), min=1e-8) * torch.sign(self.QL_base).clamp(min=1.0)
        sP = (P_ref / denomP).clamp(min=-10.0, max=10.0)
        sQ = (Q_ref / denomQ).clamp(min=-10.0, max=10.0)
        return sP, sQ

    @property
    def u_eq(self) -> torch.Tensor:
        ue = self._u_eq  # store as either (n_controls,) or (1, n_controls)
        return ue if ue.dim() == 2 else ue.unsqueeze(0)
        
    def audit_pf_at(self, V: torch.Tensor, theta: torch.Tensor,
                    delta: torch.Tensor, Eqp: torch.Tensor,
                    Ppv: torch.Tensor, Qpv: torch.Tensor):
        """
        Returns detailed KCL and complex-power mismatches at a specified (V,theta)
        without running Newton. All currents use the SAME laws as runtime.
        """
        ejth = torch.exp(1j*theta)
        Vc = V * ejth

        # Currents per runtime definitions
        I_net  = (self.Y.to(Vc.dtype) @ Vc)
        I_zip  = self._zip_currents(V, theta)                     # uses calibrated sP,sQ
        I_pv   = ((Ppv + 1j*Qpv)/torch.clamp(V,1e-6)).conj()*ejth
        Eprime = Eqp * torch.exp(1j*delta)
        I_sg   = torch.zeros_like(Vc, dtype=Vc.dtype)
        mask   = (self.sg_ratio > 0)
        if mask.any():
            Xd_c = self.Xd_prime[mask].to(Vc.real.dtype).to(Vc.dtype)
            I_sg[mask] = (Eprime[mask] - Vc[mask]) / (1j*Xd_c)

        # KCL residual (complex)
        k = I_net + I_zip - I_sg - I_pv

        # Complex-power mismatch per bus
        S_net  = Vc * I_net.conj()
        S_zip  = Vc * I_zip.conj()
        S_sg   = Vc * I_sg.conj()
        S_pv   = Vc * I_pv.conj()
        dS     = (S_net + S_zip) - (S_sg + S_pv)

        return {
            "k": k,                # complex KCL residual
            "k_norm": float(torch.linalg.norm(torch.view_as_real(k))),
            "dS": dS,              # complex power mismatch
            "dS_P_max": float(dS.real.abs().max()),
            "dS_Q_max": float(dS.imag.abs().max()),
        }
