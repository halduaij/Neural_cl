#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch

from neural_clbf.systems.control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario

logger = logging.getLogger(__name__)

class IEEE39HybridSystem(ControlAffineSystem):
    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[Dict] = None,
    ):
        self._initialize_system_constants()
        self._setup_parameters(nominal_params)
        self.goal_point_x = self._find_equilibrium()

        # Disable the parent's LQR linearization, which is not needed and was a source of errors
        super().__init__(
            nominal_params,
            dt=dt,
            controller_dt=controller_dt,
            scenarios=scenarios,
            use_linearized_controller=False,
        )
        # Manually define simple controller and Lyapunov matrices
        self.P = torch.eye(self.n_dims)
        self.K = torch.zeros(self.n_controls, self.n_dims)
        logger.info(f"System initialized with {self.n_dims} states and {self.n_controls} controls.")

    def _initialize_system_constants(self):
        self.n_gen = 10
        self.base_freq = 60.0
        self.omega_s = 2 * torch.pi * self.base_freq

        Y_np, G_np, B_np, PL_np, QL_np, Pg_target_np, Vset_np = self._build_network_np()
        self.Y = torch.from_numpy(Y_np).cfloat()
        self.G = torch.from_numpy(G_np).float()
        self.B = torch.from_numpy(B_np).float()
        self.PL_base = torch.from_numpy(PL_np).float()
        self.QL_base = torch.from_numpy(QL_np).float()
        self.Pg_target_total = torch.from_numpy(Pg_target_np).float()
        self.Vset = torch.from_numpy(Vset_np).float()

        self.H_base = torch.tensor([15.15, 21.0, 17.9, 14.3, 13.0, 17.4, 13.2, 12.15, 17.25, 250.0])
        self.D_base = torch.tensor([17.3, 11.8, 17.3, 17.3, 17.3, 17.3, 17.3, 17.3, 18.22, 18.22])
        self.Xd_prime = torch.tensor([0.0697, 0.0310, 0.0531, 0.0436, 0.1320, 0.0500, 0.0490, 0.0570, 0.0570, 0.0060])
        self.Td0_prime = torch.tensor([6.56, 10.2, 5.70, 5.69, 5.40, 7.30, 5.66, 6.70, 4.79, 7.00])
        self.Ka = torch.full((self.n_gen,), 50.0)
        self.Ta = torch.full((self.n_gen,), 0.001)
        self.R = torch.full((self.n_gen,), 0.05)
        self.Tg = torch.full((self.n_gen,), 0.05)
        self.Tt = torch.full((self.n_gen,), 2.1)

    def _setup_parameters(self, params: Scenario):
        device = self.H_base.device
        self.pv_ratio = params["pv_ratio"].float().to(device)
        self.sg_ratio = 1.0 - self.pv_ratio
        self.H = self.H_base.to(device) * self.sg_ratio.clamp(min=1e-6)
        self.D = self.D_base.to(device) * self.sg_ratio
        self.T_p_pv, self.T_q_pv = params["T_pv"][0].item(), params["T_pv"][1].item()
        self.tau_V, self.tau_theta = params["tau_network"][0].item(), params["tau_network"][1].item()
        n = self.n_gen
        self._n_dims = (n - 1) + 5 * n + 2 * n + n + (n - 1)
        self._n_controls = 2 * n
        self.idx_delta_rel, self.idx_omega, self.idx_Eq_prime, self.idx_Efd, self.idx_Pm, self.idx_Pvalve = \
            slice(0, n-1), slice(n-1, 2*n-1), slice(2*n-1, 3*n-1), slice(3*n-1, 4*n-1), slice(4*n-1, 5*n-1), slice(5*n-1, 6*n-1)
        self.idx_P_pv, self.idx_Q_pv, self.idx_V, self.idx_theta_rel = \
            slice(6*n-1, 7*n-1), slice(7*n-1, 8*n-1), slice(8*n-1, 9*n-1), slice(9*n-1, 10*n-2)
        self.idx_u_p, self.idx_u_q = slice(0, n), slice(n, 2 * n)

    def _find_equilibrium(self) -> torch.Tensor:
        logger.info("Solving for system equilibrium point...")
        V_eq_np, theta_eq_np, Pg_eq_np, Qg_eq_np, _ = self._newton_pf_np(self.Vset.numpy())
        
        V_eq, theta_eq = torch.from_numpy(V_eq_np).float(), torch.from_numpy(theta_eq_np).float()
        Pg_eq, Qg_eq = torch.from_numpy(Pg_eq_np).float(), torch.from_numpy(Qg_eq_np).float()
        Vc_eq = V_eq * torch.exp(1j * theta_eq)

        P_sg_eq, Q_sg_eq = Pg_eq * self.sg_ratio, Qg_eq * self.sg_ratio
        P_pv_eq, Q_pv_eq = Pg_eq * self.pv_ratio, Qg_eq * self.pv_ratio
        self.P_pv_eq, self.Q_pv_eq = P_pv_eq, Q_pv_eq

        Sg_eq = P_sg_eq + 1j * Q_sg_eq
        Igen_sg_eq = torch.zeros_like(Vc_eq)
        Igen_sg_eq[V_eq > 1e-6] = (Sg_eq[V_eq > 1e-6] / Vc_eq[V_eq > 1e-6]).conj()
        
        Eprime_eq_phasor = Vc_eq + 1j * self.Xd_prime * Igen_sg_eq
        delta_eq, Eq_prime_eq = torch.angle(Eprime_eq_phasor), torch.abs(Eprime_eq_phasor)
        Efd_eq = Eq_prime_eq.clone()
        self.Vref = V_eq + Efd_eq / self.Ka
        Pm_eq = P_sg_eq.clone()
        self.Pref0 = Pm_eq.clone()
        
        x_eq = torch.cat([
            delta_eq[1:]-delta_eq[0], torch.ones(self.n_gen), Eq_prime_eq, Efd_eq, Pm_eq, Pm_eq.clone(),
            P_pv_eq, Q_pv_eq, V_eq, theta_eq[1:]-theta_eq[0]
        ])
        return x_eq.unsqueeze(0)

    def _f(self, x: torch.Tensor, params: Scenario) -> torch.Tensor:
        self._setup_parameters(params)
        self.to(x.device)
        batch_size, n = x.shape[0], self.n_gen
        
        delta_rel, omega, Eq_prime, Efd, Pm, Pvalve, P_pv, Q_pv, V, theta_rel = \
            (x[:, s] for s in [self.idx_delta_rel, self.idx_omega, self.idx_Eq_prime, self.idx_Efd, self.idx_Pm, self.idx_Pvalve, self.idx_P_pv, self.idx_Q_pv, self.idx_V, self.idx_theta_rel])

        delta0 = torch.zeros(batch_size, 1, device=x.device); delta = torch.cat([delta0, delta_rel + delta0], dim=1)
        theta0 = torch.zeros(batch_size, 1, device=x.device); theta = torch.cat([theta0, theta_rel + theta0], dim=1)

        Vc = V.cfloat() * torch.exp(1j * theta.cfloat())
        Eprime_phasor = Eq_prime.cfloat() * torch.exp(1j * delta.cfloat())
        
        I_sg = torch.zeros_like(Vc)
        sg_mask = self.sg_ratio > 1e-6
        I_sg[:, sg_mask] = (Eprime_phasor[:, sg_mask] - Vc[:, sg_mask]) / (1j * self.Xd_prime[sg_mask])
        S_sg = Vc * I_sg.conj(); P_sg, Q_sg = S_sg.real, S_sg.imag

        # ** RESET TO SIMPLER, CONSISTENT CONSTANT POWER LOAD MODEL **
        P_load = self.PL_base.to(x.device)
        Q_load = self.QL_base.to(x.device)

        S_inj = (P_sg + P_pv - P_load) + 1j * (Q_sg + Q_pv - Q_load)
        I_net = (self.Y @ Vc.unsqueeze(-1)).squeeze(-1)
        I_mismatch = (S_inj / Vc).conj() - I_net
        
        f = torch.zeros(batch_size, self.n_dims, 1, device=x.device)
        f[:, self.idx_delta_rel, 0] = self.omega_s * (omega[:, 1:] - omega[:, 0].unsqueeze(1))
        f[:, self.idx_omega, 0] = (self.omega_s / (2 * self.H)) * (Pm - P_sg - self.D * (omega - 1.0))
        f[:, self.idx_Eq_prime, 0] = (Efd - Eq_prime) / self.Td0_prime
        f[:, self.idx_Efd, 0] = (self.Ka * (self.Vref - V) - Efd) / self.Ta
        f[:, self.idx_Pm, 0] = (Pvalve - Pm) / self.Tt
        Pref = self.Pref0 - (omega - 1.0) / self.R
        f[:, self.idx_Pvalve, 0] = (Pref - Pvalve) / self.Tg
        f[:, self.idx_P_pv, 0] = -P_pv / self.T_p_pv
        f[:, self.idx_Q_pv, 0] = -Q_pv / self.T_q_pv
        f[:, self.idx_V, 0] = I_mismatch.imag / self.tau_V
        f[:, self.idx_theta_rel, 0] = (I_mismatch.real[:, 1:] - I_mismatch.real[:, 0].unsqueeze(1)) / self.tau_theta
        
        return f

    def _g(self, x: torch.Tensor, params: Scenario) -> torch.Tensor:
        g = torch.zeros(x.shape[0], self.n_dims, self.n_controls, device=x.device)
        p_rows, p_cols = torch.arange(self.idx_P_pv.start, self.idx_P_pv.stop), torch.arange(self.idx_u_p.start, self.idx_u_p.stop)
        q_rows, q_cols = torch.arange(self.idx_Q_pv.start, self.idx_Q_pv.stop), torch.arange(self.idx_u_q.start, self.idx_u_q.stop)
        g[:, p_rows, p_cols] = 1.0 / self.T_p_pv
        g[:, q_rows, q_cols] = 1.0 / self.T_q_pv
        return g

    @property
    def n_dims(self) -> int: return self._n_dims
    @property
    def n_controls(self) -> int: return self._n_controls
    @property
    def goal_point(self): return self.goal_point_x.to(self.Y.device)
    @property
    def u_eq(self) -> torch.Tensor:
        u_eq = torch.cat([self.P_pv_eq, self.Q_pv_eq])
        return u_eq.unsqueeze(0).to(self.Y.device)
    def to(self, device):
        for name, val in self.__dict__.items():
            if isinstance(val, torch.Tensor): setattr(self, name, val.to(device))
        return self
    def validate_params(self, params: Scenario) -> bool: return True
    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]: return (torch.full((self.n_dims,), -torch.inf), torch.full((self.n_dims,), torch.inf))
    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]: return (torch.full((self.n_controls,), -torch.inf), torch.full((self.n_controls,), torch.inf))
    def safe_mask(self, x: torch.Tensor) -> torch.Tensor: return torch.ones_like(x[:, 0], dtype=torch.bool)
    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor: return torch.zeros_like(x[:, 0], dtype=torch.bool)

    def _build_network_np(self) -> Tuple:
        base_MVA, nbus = 100.0, 39
        branches = [ (1, 2, 0.0035, 0.0411, 0.6987, 0.0), (1, 39, 0.001, 0.025, 0.75, 0.0), (2, 3, 0.0013, 0.0151, 0.2572, 0.0), (2, 25, 0.007, 0.0086, 0.146, 0.0), (3, 4, 0.0013, 0.0213, 0.2214, 0.0), (3, 18, 0.0011, 0.0133, 0.2138, 0.0), (4, 5, 0.0008, 0.0128, 0.1342, 0.0), (4, 14, 0.0008, 0.0129, 0.1382, 0.0), (5, 6, 0.0002, 0.0026, 0.0434, 0.0), (5, 8, 0.0008, 0.0112, 0.1476, 0.0), (6, 7, 0.0006, 0.0092, 0.113, 0.0), (6, 11, 0.0007, 0.0082, 0.1389, 0.0), (7, 8, 0.0004, 0.0046, 0.078, 0.0), (8, 9, 0.0023, 0.0363, 0.3804, 0.0), (9, 39, 0.001, 0.025, 1.2, 0.0), (10, 11, 0.0004, 0.0043, 0.0729, 0.0), (10, 13, 0.0004, 0.0043, 0.0729, 0.0), (13, 14, 0.0009, 0.0101, 0.1723, 0.0), (14, 15, 0.0018, 0.0217, 0.366, 0.0), (15, 16, 0.0009, 0.0094, 0.171, 0.0), (16, 17, 0.0007, 0.0089, 0.1342, 0.0), (16, 19, 0.0016, 0.0195, 0.304, 0.0), (16, 21, 0.0008, 0.0135, 0.2548, 0.0), (16, 24, 0.0003, 0.0059, 0.068, 0.0), (17, 18, 0.0007, 0.0082, 0.1319, 0.0), (17, 27, 0.0013, 0.0173, 0.3216, 0.0), (21, 22, 0.0008, 0.014, 0.2565, 0.0), (22, 23, 0.0006, 0.0096, 0.1846, 0.0), (23, 24, 0.0022, 0.035, 0.361, 0.0), (25, 26, 0.0032, 0.0323, 0.513, 0.0), (26, 27, 0.0014, 0.0147, 0.2396, 0.0), (26, 28, 0.0043, 0.0474, 0.7802, 0.0), (26, 29, 0.0057, 0.0625, 1.029, 0.0), (28, 29, 0.0014, 0.0151, 0.249, 0.0), (12, 11, 0.0016, 0.0435, 0.0, 1.006), (12, 13, 0.0016, 0.0435, 0.0, 1.006), (6, 31, 0.0, 0.025, 0.0, 1.07), (10, 32, 0.0, 0.02, 0.0, 1.07), (19, 33, 0.0007, 0.0142, 0.0, 1.07), (20, 34, 0.0009, 0.018, 0.0, 1.009), (22, 35, 0.0, 0.0143, 0.0, 1.025), (23, 36, 0.0005, 0.0272, 0.0, 1.0), (25, 37, 0.0006, 0.0232, 0.0, 1.025), (2, 30, 0.0, 0.0181, 0.0, 1.025), (29, 38, 0.0008, 0.0156, 0.0, 1.025), (19, 20, 0.0007, 0.0138, 0.0, 1.06), ]
        Y = np.zeros((nbus, nbus), dtype=complex)
        for f, t, r, x, b, tap in branches:
            f -= 1; t -= 1; y = 1.0/complex(r,x) if (r!=0 or x!=0) else 0.0
            bsh = 1j*(b/2.0); a = 1.0 if tap==0.0 else float(tap)
            Y[f,f] += (y+bsh)/(a*a); Y[t,t] += y+bsh; Y[f,t]-=y/a; Y[t,f]-=y/a
        gen_bus_order=[31,30,32,33,34,35,36,37,38,39]; keep=[b-1 for b in gen_bus_order]
        elim=[i for i in range(nbus) if i not in keep]
        Yaa,Ybb,Yab,Yba = Y[np.ix_(keep,keep)], Y[np.ix_(elim,elim)], Y[np.ix_(keep,elim)], Y[np.ix_(elim,keep)]
        Yred = Yaa - Yab @ np.linalg.solve(Ybb, Yba)
        P_MW={3:322.0,4:500.0,7:233.8,8:522.0,12:7.5,15:320.0,16:329.0,18:158.0,20:628.0,21:274.0,23:247.5,24:308.6,25:224.0,26:139.0,27:281.0,28:206.0,29:283.5,31:9.2,39:1104.0}
        Q_MVAr={3:2.4,4:184.0,7:84.0,8:176.0,12:88.0,15:153.0,16:32.3,18:30.0,20:103.0,21:115.0,23:84.6,24:-92.0,25:47.2,26:17.0,27:75.5,28:27.6,29:26.9,31:4.6,39:250.0}
        P,Q=np.zeros(nbus),np.zeros(nbus)
        for k,v in P_MW.items(): P[k-1]=v/base_MVA
        for k,v in Q_MVAr.items(): Q[k-1]=v/base_MVA
        Vset=np.array([0.982,1.0475,0.9831,0.9972,1.0123,1.0493,1.0635,1.0278,1.0265,1.03])
        I_b=(P[elim]-1j*Q[elim]); I_eq=-Yab@np.linalg.solve(Ybb,I_b); S_eq=Vset*np.conj(I_eq)
        S_dir=P[keep]+1j*Q[keep]; S_tot=S_eq+S_dir; PL,QL=S_tot.real,S_tot.imag
        gen_P_MW={30:250.0,32:650.0,33:632.0,34:508.0,35:650.0,36:560.0,37:540.0,38:830.0,39:1000.0}
        Pg_pu=np.zeros(10)
        for i,b in enumerate(gen_bus_order):
            if b in gen_P_MW: Pg_pu[i]=gen_P_MW[b]/base_MVA
        return Yred, Yred.real, Yred.imag, PL, QL, Pg_pu, Vset

    def _newton_pf_np(self, Vset, slack=0, tol=1e-7, itmax=20):
        n=self.n_gen; P_spec=self.Pg_target_total.numpy()-self.PL_base.numpy()
        theta,V=np.zeros(n),Vset.copy(); pvpq_idx=list(range(1,n))
        B_reduced = -self.B.numpy()[np.ix_(pvpq_idx, pvpq_idx)]
        if np.linalg.det(B_reduced) < 1e-12: logger.error("Jacobian is singular."); return V,theta,P_spec+self.PL_base.numpy(),np.zeros(n),None
        for it in range(itmax):
            Vc=V*np.exp(1j*theta); S_inj=Vc*np.conj(self.Y.numpy()@Vc); P,Q=S_inj.real,S_inj.imag
            mis_P=P_spec[pvpq_idx]-P[pvpq_idx]; mis_V=Vset[pvpq_idx]**2-V[pvpq_idx]**2
            d_theta=np.linalg.solve(B_reduced,mis_P/V[pvpq_idx])
            d_V=np.linalg.solve(B_reduced,mis_V/V[pvpq_idx])
            theta[pvpq_idx]+=d_theta; V[pvpq_idx]+=d_V
            if max(np.abs(d_theta).max(),np.abs(d_V).max())<tol: break
        Vc=V*np.exp(1j*theta); S_inj=Vc*np.conj(self.Y.numpy()@Vc)
        Pg_used=S_inj.real+self.PL_base.numpy(); Qg_used=S_inj.imag+self.QL_base.numpy()
        return V,theta,Pg_used,Qg_used,None

# --- Self-Contained Verification Script ---
if __name__ == '__main__':
    from neural_clbf.systems.fixed_power_dae_ieee39_bestpractice3 import FixedPowerDAESystem

    # This self-test will now run if you execute `python SwingEquationSystem.py`
    
    # 1. Compare Foundations
    print("--- 1. Comparing Foundational Parameters ---")
    dae_system_test = FixedPowerDAESystem()
    nominal_params_test = {
        "pv_ratio": torch.zeros(10), # No PV for direct comparison
        "T_pv": torch.tensor([0.05, 0.05]),
        "tau_network": torch.tensor([0.01, 0.01]),
    }
    ode_system_test = IEEE39HybridSystem(nominal_params=nominal_params_test)
    
    diff_Y = np.linalg.norm(dae_system_test.Y - ode_system_test.Y.numpy())
    diff_PL = np.linalg.norm(dae_system_test.PL - ode_system_test.PL_base.numpy())
    if diff_Y < 1e-9 and diff_PL < 1e-9:
        print("✅ SUCCESS: Foundational parameters in _build_network_np match the DAE model.")
    else:
        print("❌ FAILED: Foundational parameters DO NOT match.")
        exit()

    # 2. Verify Equilibrium
    print("\n--- 2. Verifying Equilibrium Point ---")
    x_eq_test = ode_system_test.goal_point
    u_eq_test = ode_system_test.u_nominal(x_eq_test)
    x_dot_test = ode_system_test.closed_loop_dynamics(x_eq_test, u_eq_test)
    norm_test = torch.norm(x_dot_test)

    print(f"Norm of dynamics at equilibrium: {norm_test.item():.4e}")
    if norm_test.item() < 1e-6:
        print("✅ SUCCESS: The equilibrium point is correct.")
    else:
        print("❌ FAILED: The equilibrium point is incorrect.")