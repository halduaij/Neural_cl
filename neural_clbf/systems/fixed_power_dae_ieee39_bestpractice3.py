#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FixedPowerDAESystem — IEEE-39 (New England) network integrated
==============================================================

This module embeds the user's DAE (swing + AVR + governor/turbine with algebraic
PF constraints, robust algebraic solve, analytic Jacobian, ZIP + C¹-limited CPL,
and a physics-consistent pseudo-transient field relaxer), and replaces the synthetic
ring network with the *actual* IEEE-39 network reduced to the 10 generator buses
via Kron/Ward reduction.

Key features
------------
1) Network:
   • Build the full 39×39 Y-bus from lines/transformers (R, X, B, tap).
   • Kron-reduce to kept buses [31, 30, 32, 33, 34, 35, 36, 37, 38, 39] with
     bus 31 (Gen 2) placed first so index 0 is the slack in this reduced model.
   • Ward-equivalent mapping of eliminated PQ loads onto the kept buses at
     nominal PV magnitudes (Vset from the IEEE-39 tables). This yields
     per-generator-bus (PL, QL) used by the algebraic KCL solver.

2) Machines and control:
   • IEEE-39-style machine parameters (H, Xd', Xq, Td0') are provided (see arrays).
   • Governor droop is present (1/R) with Tg, Tt lags; mechanical damping D_i can
     be zero by default or user-specified.
   • Optional system-level frequency-sensitive load damping (D_load) that scales
     the aggregate P load by COI frequency deviation.

3) Compatibility:
   • The public API and numerical methods match the original user code so that
     CPF, static-margin, simulation, and plotting work unchanged.

Usage
-----
Run as a script:
    python fixed_power_dae_ieee39.py

Or import the class:
    from fixed_power_dae_ieee39 import FixedPowerDAESystem

Notes
-----
- Per-unit base is 100 MVA and 60 Hz. Line charging B is split 50/50 to ends.
- Transformer taps are modeled with magnitude only (no phase-shifting taps
  in this dataset). Tap angles are 0.
- The Ward equivalent is constructed at nominal PV magnitudes and flat angles,
  which is standard for building a generator-bus equivalent. Your ZIP/CPL model
  then provides voltage sensitivity during dynamics.

"""

import time
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


class FixedPowerDAESystem:
    """
    Multi-machine swing + AVR + governor/turbine with algebraic PF constraints.
    Robust algebraic solve with scale-aware ZIP + C¹-smooth current-limited CPL.
    Includes a *physics-consistent pseudo-transient field relaxer* to ensure
    KCL solvability immediately after large P/Q changes.
    """

    # ---------------- Configuration knobs ----------------
    CACHE_TOL_T = 1e-10          # s: consider two RHS callbacks "same t"
    CACHE_TOL_Y = 1e-10          # state proximity for cache reuse (δ, E')
    CONT_THRESHOLD = 0.02       # trigger continuation once ΔP ~2%
    ALG_SOLVE_TIME_BUDGET = 1.0 # s: per-call algebraic solve CPU budget
    RHS_SAME_T_CLUSTER_MAX = 500
    V_LO, V_HI = 0.35, 1.60     # bounds for voltages in algebraic solve
    TH_LO, TH_HI = -5.0, 5.0    # bounds for angles (rad) for buses 2..n

    # Feasibility rescue if algebraic target is infeasible at current (δ,E′)
    BACKOFF_ENABLED = True
    BACKOFF_MIN_V = 0.5        # engineering acceptability threshold
    BACKOFF_TOL_RN = 1e-6
    BACKOFF_MAX_ITERS = 16

    # Optional AVR output limiter (disabled by default)
    AVR_LIMITS = True
    AVR_EFD_MIN_MARGIN = 0.6
    AVR_EFD_MAX_MARGIN = 1.8


    # === Best-practice PF/TDS options ===
    PF_ENFORCE_Q_LIMITS = True   # Enforce Q-limits (PV→PQ) in PF by default
    FULL_NETWORK_PF = True        # Use full 39-bus PF to initialize (recommended)
    PF_SLACK_BUS = 31             # <-- 1-based bus number for full-network slack (set to 31 as you prefer)

    PQBRAK_VBREAK = 0.70          # PSS®E-like: reduce CPL below ~0.7 pu (see set_pqbrak_like)
    # ---------------- Load-model controls ----------------
    zip_enabled = True


    cpl_limit_enabled = False
    CPL_Imax_gamma = 1.05   # Imax = gamma*|S_P|/V_nom; sat starts near V≈1/gamma
    V_nom = 1.0

    # ZIP mixing coefficients (sum to 1.0 each for P and Q)
    kP_P, kI_P, kZ_P = 0.10, 0.10, 0.80  # 80% of active load is constant impedance
    kP_Q, kI_Q, kZ_Q = 0.10, 0.10, 0.80  # 80% of reactive load is constant impedance
    # --- NEW: smooth CPL limiter band (fractional width above 1.0) ---
    CPL_SMOOTH_EPS = 0.08   # C¹ blending for |I|/Imax in [1, 1+eps]; try 0.08–0.15

    # --- NEW: optional frequency-sensitive aggregate load damping ---
    LOAD_FREQ_DAMP = False     # if True, scale P_load by COI frequency
    D_load = 0.0               # pu(P)/pu(ω) on system base for aggregate load damping

    # ---------------- NEW: pseudo-transient field relaxer ----------------
    # When KCL is infeasible at (δ, E′), advance (Efd, E′) only, with their *true*
    # ODEs for a small pseudo-physical time and retry the algebraic solve.
    PTC_ENABLED = False
    PTC_DT0 = 2e-3           # s: initial pseudo time step for Efd/E′ micro-advance
    PTC_DT_MAX = 5e-2        # s: max pseudo time step
    PTC_MAX_PHYS = 0.25      # s: max total pseudo physical time per call
    PTC_ACCEPT_RN = 1e-4     # accept when ||KCL||_2 below this (after PTC)
    PTC_MIN_V = 0.85         # require min(V) above this
    PTC_MAX_NEWTON = 300     # max Newton evals per inner attempt
    PTC_VERBOSE = False      # set True to print a one-line summary when engaged

    # -----------------------------------------------------

    def __init__(self, seed: int = 7):
        # -----------------------------
        # System size and constants
        # -----------------------------
        self.n_gen = 10
        self.base_freq = 60.0
        self.omega_s = 2 * np.pi * self.base_freq  # rad/s
        self.disturbance = None
        self.alter_events = []  # Simple list for ANDES-style load changes
        self._step_fired = False
        self._force_continuation = False

        # -----------------------------
        # Machine parameters (pu) — IEEE-39 mapped to gen_bus_order = [31,30,32,33,34,35,36,37,38,39]
        # The arrays below are provided in the kept-bus order above; they are consistent with
        # a 100 MVA, 60 Hz base. D is zero by default (you can set it later).
        # -----------------------------
        # Inertia constants H (s) - Updated to match ANDES values
        self.H = np.array([15.15, 21.0, 17.9, 14.3, 13.0, 17.4, 13.2, 12.15, 17.25, 250.0], dtype=float)
        # Mechanical damping (instantaneous torque proportional to speed deviation) - Updated to match ANDES values
        self.D = np.array([17.3, 11.8, 17.3, 17.3, 17.3,
              17.3, 17.3, 17.3, 18.22, 18.22], dtype=float)
        # Electrical parameters used by this model
        self.Xd_prime = np.array([0.0697, 0.0310, 0.0531, 0.0436, 0.1320, 0.0500, 0.0490, 0.0570, 0.0570, 0.0060], dtype=float)
        self.Xq       = np.array([0.2820, 0.0690, 0.2370, 0.2580, 0.6200, 0.2410, 0.2920, 0.2800, 0.2050, 0.0190], dtype=float)
        self.Td0_prime = np.array([6.56, 10.2, 5.70, 5.69, 5.40, 7.30, 5.66, 6.70, 4.79, 7.00], dtype=float)  # Swapped first two values to match ANDES

        # AVR & governor/turbine - Updated to match ANDES ESST3A/TGOV1 values
        self.Ka = np.array([50.0] * self.n_gen)   # Updated to match ANDES KA
        self.Ta = np.array([0.001] * self.n_gen) # AVR time constant (s)
        self.R  = np.array([0.05] * self.n_gen)  # droop (pu/pu) → 1/R is the slope (matches ANDES)
        self.Tg = np.array([0.05] * self.n_gen)  # governor time constant (s)
        self.Tt = np.array([2.1] * self.n_gen)   # turbine time constant (s) - Updated to match ANDES T3

        # -----------------------------
        # Build IEEE-39 reduced network and data (fills: Y,G,B,Vset,PL,QL,Pg_target,Qg_min,Qg_max)
        # -----------------------------
        print("=" * 80)
        print("IEEE-39 (New England) — DAE with robust algebraic solver (+ field relaxer + C¹ CPL limiter)")
        print("=" * 80)

        self.build_network()

        print(f"  Total generation *scheduled* (excl. slack) = {np.sum(self.Pg_target):.3f} pu")
        print(f"  Total load from Ward eq: P={self.PL.sum():.3f} pu, Q={self.QL.sum():.3f} pu")

        # Initialize from PF / phasors and verify
        self.find_equilibrium_pf_and_initialize(seed=seed)
        self._reset_step_state()
        self.verify_equilibrium()
        self._set_efd_limits_from_Q_caps(margin=1.15)
        self.LDC_ENABLED = True
        self.Rc = np.zeros(self.n_gen)        # start at 0
        self.Xc = 0.08 * np.ones(self.n_gen)  # 0.05–0.12 typical; try 0.08
        # Per-bus static load scaling factors for alter_events
        self.bus_P_scale = np.ones(self.n_gen)
        self.bus_Q_scale = np.ones(self.n_gen)
        self.set_pqbrak_like(vbreak=0.85, smooth_eps=0.06)

    # =========================================================================
    # Network
    # =========================================================================
    def build_network(self):
        """
        Build the IEEE-39 (New England) network reduced to the 10 generator buses.

        Steps:
          1) Assemble full 39x39 Ybus from line/transformer data (R, X, B, tap).
          2) Kron-reduce to kept buses: [31, 30, 32, 33, 34, 35, 36, 37, 38, 39]
             with bus 31 placed first so index 0 is the slack (Gen 2) in the reduced model.
          3) Form a Ward-equivalent of eliminated PQ loads at kept buses, at nominal PV magnitudes.
          4) Set:
               self.Y, self.G, self.B
               self.Vset (PV magnitudes from the IEEE-39 table)
               self.PL, self.QL  (per-unit on 100 MVA)
               self.Pg_target    (per-unit schedules; slack value left 0.0)
               self.Qg_min, self.Qg_max
        """
        base_MVA = 100.0
        nbus = 39

        # ------------------------------
        # 1) Raw branch data (from→to, R, X, B, tap)
        # Tap magnitude 0.0 means "no off-nominal tap" (treated as 1.0).
        # ------------------------------
        branches = [
            # Lines (tap = 0 → treated as 1)
            (1,  2, 0.0035, 0.0411, 0.6987, 0.000),
            (1, 39, 0.0010, 0.0250, 0.7500, 0.000),
            (2,  3, 0.0013, 0.0151, 0.2572, 0.000),
            (2, 25, 0.0070, 0.0086, 0.1460, 0.000),
            (3,  4, 0.0013, 0.0213, 0.2214, 0.000),
            (3, 18, 0.0011, 0.0133, 0.2138, 0.000),
            (4,  5, 0.0008, 0.0128, 0.1342, 0.000),
            (4, 14, 0.0008, 0.0129, 0.1382, 0.000),
            (5,  6, 0.0002, 0.0026, 0.0434, 0.000),
            (5,  8, 0.0008, 0.0112, 0.1476, 0.000),
            (6,  7, 0.0006, 0.0092, 0.1130, 0.000),
            (6, 11, 0.0007, 0.0082, 0.1389, 0.000),
            (7,  8, 0.0004, 0.0046, 0.0780, 0.000),
            (8,  9, 0.0023, 0.0363, 0.3804, 0.000),
            (9, 39, 0.0010, 0.0250, 1.2000, 0.000),
            (10,11, 0.0004, 0.0043, 0.0729, 0.000),
            (10,13, 0.0004, 0.0043, 0.0729, 0.000),
            (13,14, 0.0009, 0.0101, 0.1723, 0.000),
            (14,15, 0.0018, 0.0217, 0.3660, 0.000),
            (15,16, 0.0009, 0.0094, 0.1710, 0.000),
            (16,17, 0.0007, 0.0089, 0.1342, 0.000),
            (16,19, 0.0016, 0.0195, 0.3040, 0.000),
            (16,21, 0.0008, 0.0135, 0.2548, 0.000),
            (16,24, 0.0003, 0.0059, 0.0680, 0.000),
            (17,18, 0.0007, 0.0082, 0.1319, 0.000),
            (17,27, 0.0013, 0.0173, 0.3216, 0.000),
            (21,22, 0.0008, 0.0140, 0.2565, 0.000),
            (22,23, 0.0006, 0.0096, 0.1846, 0.000),
            (23,24, 0.0022, 0.0350, 0.3610, 0.000),
            (25,26, 0.0032, 0.0323, 0.5130, 0.000),
            (26,27, 0.0014, 0.0147, 0.2396, 0.000),
            (26,28, 0.0043, 0.0474, 0.7802, 0.000),
            (26,29, 0.0057, 0.0625, 1.0290, 0.000),
            (28,29, 0.0014, 0.0151, 0.2490, 0.000),
            # Transformers (non-unity taps shown)
            (12,11, 0.0016, 0.0435, 0.0, 1.006),
            (12,13, 0.0016, 0.0435, 0.0, 1.006),
            (6, 31, 0.0000, 0.0250, 0.0, 1.070),
            (10,32, 0.0000, 0.0200, 0.0, 1.070),
            (19,33, 0.0007, 0.0142, 0.0, 1.070),
            (20,34, 0.0009, 0.0180, 0.0, 1.009),
            (22,35, 0.0000, 0.0143, 0.0, 1.025),
            (23,36, 0.0005, 0.0272, 0.0, 1.000),
            (25,37, 0.0006, 0.0232, 0.0, 1.025),
            (2, 30, 0.0000, 0.0181, 0.0, 1.025),
            (29,38, 0.0008, 0.0156, 0.0, 1.025),
            (19,20, 0.0007, 0.0138, 0.0, 1.060),
        ]

        Y = np.zeros((nbus, nbus), dtype=complex)
        for f, t, r, x, b, tap in branches:
            f -= 1; t -= 1
            z = complex(r, x)
            y = 0.0 if (r == 0.0 and x == 0.0) else 1.0 / z
            bsh = 1j * (b / 2.0)
            a = 1.0 if (tap is None or tap == 0.0) else float(tap)  # tap magnitude, angle = 0
            # Ybus stamps with off-nominal tap on "from" side:
            Y[f, f] += (y + bsh) / (a * a)
            Y[t, t] += (y + bsh)
            Y[f, t] += -y / a
            Y[t, f] += -y / a

        # Keep a copy of full-network Y for best-practice PF/TDS alignment
# --- Save full network admittance AFTER all branch and shunt updates ---
        self.nbus = Y.shape[0]
        self.Y_full = Y.copy()
        self.G_full = self.Y_full.real
        self.B_full = self.Y_full.imag


        # ------------------------------
        # 2) Kron reduction to 10 generator buses with slack first
        # ------------------------------
        gen_bus_order = [31, 30, 32, 33, 34, 35, 36, 37, 38, 39]  # bus indices as in IEEE-39
        self.gen_bus_order = gen_bus_order[:]
        self.keep_idx = [b-1 for b in gen_bus_order]

        keep = [b - 1 for b in gen_bus_order]
        elim = [i for i in range(nbus) if i not in keep]
        Yaa = Y[np.ix_(keep, keep)]
        Ybb = Y[np.ix_(elim, elim)]
        Yab = Y[np.ix_(keep, elim)]
        Yba = Y[np.ix_(elim, keep)]
        # Solve with Ybb (assumed nonsingular for this dataset)
        Yred = Yaa - Yab @ np.linalg.solve(Ybb, Yba)

        self.Y = Yred
        self.G = self.Y.real
        self.B = self.Y.imag

        # ------------------------------
        # 3) Ward equivalent of eliminated PQ loads at kept buses
        #    (flat angles, PV magnitudes used)
        # ------------------------------
        # Bus loads from the IEEE-39 case (MW and MVAr). Only non-zeros listed.
        P_MW = {3:322.0, 4:500.0, 7:233.8, 8:522.0, 12:7.5, 15:320.0, 16:329.0,
                18:158.0, 20:628.0, 21:274.0, 23:247.5, 24:308.6, 25:224.0,
                26:139.0, 27:281.0, 28:206.0, 29:283.5, 31:9.2, 39:1104.0}  # Reverted to original
        Q_MVAr = {3:2.4, 4:184.0, 7:84.0, 8:176.0, 12:88.0, 15:153.0, 16:32.3,
                  18:30.0, 20:103.0, 21:115.0, 23:84.6, 24:-92.0, 25:47.2,
                  26:17.0, 27:75.5, 28:27.6, 29:26.9, 31:4.6, 39:250.0}    # Reverted to original

        P = np.zeros(nbus); Q = np.zeros(nbus)
        for k, v in P_MW.items(): P[k - 1] = v / base_MVA
        for k, v in Q_MVAr.items(): Q[k - 1] = v / base_MVA

        # Save full-network loads for full PF
        self.P_full = P.copy()
        self.Q_full = Q.copy()

        # PV magnitudes from the IEEE-39 PV table, in the kept-bus order
        Vset = np.array([0.9820, 1.0475, 0.9831, 0.9972, 1.0123,
                         1.0493, 1.0635, 1.0278, 1.0265, 1.0300], dtype=float)  # Reverted to original

        # Current injections at eliminated buses for constant-power loads at V ≈ 1∠0
        I_b = (P[elim] - 1j * Q[elim])  # conj(S)/V*, taking V=1∠0 for the eliminated buses
        I_eq = -Yab @ np.linalg.solve(Ybb, I_b)  # equivalent currents at kept buses

        # Equivalent complex powers at the kept buses at nominal PV magnitudes:
        S_eq = Vset * np.conj(I_eq)   # pu on 100 MVA
        # Add direct loads sitting at generator buses (31 and 39 in this dataset)
        S_dir = (P[keep] + 1j * Q[keep])
        S_tot = S_eq + S_dir

        self.PL = S_tot.real.copy()
        self.QL = S_tot.imag.copy()

        # ------------------------------
        # 4) Generator real power schedules (per-unit), PV magnitudes, Q limits
        # ------------------------------
        # Known PV injections in MW (slack bus 31 not fixed here; PF will determine its P)
        gen_P_MW = {
            30: 250.0, 32: 650.0, 33: 632.0, 34: 508.0, 35: 650.0,
            36: 560.0, 37: 540.0, 38: 830.0, 39: 1000.0  # Reverted to original working values
        }
        # order = [31, 30, 32, 33, 34, 35, 36, 37, 38, 39]
        Pg_pu = np.zeros(10, dtype=float)
        for idx, bus in enumerate(gen_bus_order):
            if bus in gen_P_MW:
                Pg_pu[idx] = gen_P_MW[bus] / base_MVA
            else:
                Pg_pu[idx] = 0.0  # slack; value unused by PF
        self.Pg_target = Pg_pu
        self.Vset = Vset

        # Reactive power limits (broad, proportional to scheduled P, in pu)
        self.Qg_min = -2.8 * np.maximum(self.Pg_target, 1e-3)  # Reverted to original simple formula
        self.Qg_max = +2.8 * np.maximum(self.Pg_target, 1e-3)

        print(f"  IEEE-39 reduction: cond(B_imag) = {np.linalg.cond(self.B):.2e}")
        print(f"  Σ load P,Q (pu) = {self.PL.sum():.3f}, {self.QL.sum():.3f}  "
              f"(includes eliminated-network losses via Ward eq.)")


    # =========================================================================
    # Best-practice: full-network power flow (PF) for initialization
    # =========================================================================
    def _calc_PQ_injections_full(self, Vmag, theta):
        """Compute P,Q injections using full-network Y_full."""
        n = self.nbus
        P = np.zeros(n)
        Q = np.zeros(n)
        for i in range(n):
            for j in range(n):
                gij = self.G_full[i, j]
                bij = self.B_full[i, j]
                ang = theta[i] - theta[j]
                P[i] += Vmag[i] * Vmag[j] * (gij * np.cos(ang) + bij * np.sin(ang))
                Q[i] += Vmag[i] * Vmag[j] * (gij * np.sin(ang) - bij * np.cos(ang))
        return P, Q
    def _newton_pf_full_network(self, Vset_full, slack_full=None, tol=1e-6, itmax=40):
        """
        Newton–Raphson PF on the full 39-bus network with *per-iteration* PV→PQ enforcement.
        Returns Vmag, theta, Pg_used, Qg_used, bus_type.
        """
        if slack_full is None:
            slack_full = int(self.PF_SLACK_BUS) - 1

        n = self.nbus

        # ---- Bus types: 0=slack, 1=PV, 2=PQ
        bus_type = np.full(n, 2, dtype=int)
        bus_type[slack_full] = 0
        gen_buses = set([b - 1 for b in self.gen_bus_order])
        for gb in gen_buses:
            if gb != slack_full:
                bus_type[gb] = 1

        # ---- Schedules and specifications
        base_MVA = 100.0
        gen_P_MW = {30: 250.0, 32: 650.0, 33: 632.0, 34: 508.0, 35: 650.0,
                    36: 560.0, 37: 540.0, 38: 830.0, 39: 1000.0}
        Pg_sched = np.zeros(n)
        for k, mw in gen_P_MW.items():
            Pg_sched[k - 1] = mw / base_MVA
        P_spec = Pg_sched - self.P_full                   # injection spec for P
        Qg = np.zeros(n)                                   # generator Q (for PV calc and PQ caps)
        Qg_min = -0.8 * np.maximum(Pg_sched, 1e-3)         # simple broad limits for PF stage
        Qg_max = +0.8 * np.maximum(Pg_sched, 1e-3)

        # ---- Initial guess
        theta = np.zeros(n)
        Vmag  = np.ones(n)
        Vset_map = {b - 1: Vset_full[b - 1] for b in self.gen_bus_order}  # correct mapping
        for i in range(n):
            if bus_type[i] == 1:
                Vmag[i] = Vset_map.get(i, 1.0)
        Vmag[slack_full] = Vset_map.get(slack_full, 1.0)

        # ---- NR iterations
        for it in range(itmax):

            # Power injections from current (V,θ)
            P, Q = self._calc_PQ_injections_full(Vmag, theta)

            # ---- PV→PQ ENFORCEMENT (every iteration) ----
            switched = False
            if self.PF_ENFORCE_Q_LIMITS:
                pv_now = [i for i in range(n) if bus_type[i] == 1]
                for i in pv_now:
                    qgi = Q[i] + self.Q_full[i]   # generator Q needed at bus i
                    Qg[i] = qgi
                    if qgi < Qg_min[i] - 1e-8:
                        Qg[i] = Qg_min[i]; bus_type[i] = 2; switched = True
                    elif qgi > Qg_max[i] + 1e-8:
                        Qg[i] = Qg_max[i]; bus_type[i] = 2; switched = True
            if switched:
                # restart the iteration with updated bus types
                continue

            # ---- Build mismatch in the standard order: [ΔP for PV+PQ; ΔQ for PQ]
            pv_idx = [i for i in range(n) if bus_type[i] == 1]
            pq_idx = [i for i in range(n) if bus_type[i] == 2]
            mis_P = [P_spec[i] - P[i] for i in (pv_idx + pq_idx)]
            mis_Q = [(-self.Q_full[i] + Qg[i]) - Q[i] for i in pq_idx]
            rhs   = np.concatenate([mis_P, mis_Q]) if len(mis_Q) else np.array(mis_P)

            # Convergence test
            if np.max(np.abs(rhs)) < tol:
                break

            # ---- Assemble Jacobian blocks (columns: θ (excluding slack), then V of PQ)
            angle_vars = [i for i in range(n) if i != slack_full]
            V_vars     = pq_idx
            H = np.zeros((len(pv_idx) + len(pq_idx), len(angle_vars)))
            N = np.zeros((len(pv_idx) + len(pq_idx), len(V_vars)))
            J = np.zeros((len(pq_idx), len(angle_vars)))
            L = np.zeros((len(pq_idx), len(V_vars)))

            angle_col = {bus: k for k, bus in enumerate(angle_vars)}
            V_col     = {bus: k for k, bus in enumerate(V_vars)}
            P_rows    = pv_idx + pq_idx
            Q_rows    = pq_idx

            for r, i in enumerate(P_rows):
                Vi, Pi, Qi = Vmag[i], P[i], Q[i]
                for k in angle_vars:
                    if k == i:
                        H[r, angle_col[k]] = -Qi - self.B_full[i, i] * Vi * Vi
                    else:
                        Vk = Vmag[k]; ang = theta[i] - theta[k]
                        # CORRECT off-diagonal sign:
                        H[r, angle_col[k]] = Vi * Vk * (self.G_full[i, k] * np.sin(ang) - self.B_full[i, k] * np.cos(ang))
                for k in V_vars:
                    if k == i:
                        N[r, V_col[k]] = Pi / Vi + self.G_full[i, i] * Vi
                    else:
                        ang = theta[i] - theta[k]
                        N[r, V_col[k]] = Vi * (self.G_full[i, k] * np.cos(ang) + self.B_full[i, k] * np.sin(ang))

            for r, i in enumerate(Q_rows):
                Vi, Pi, Qi = Vmag[i], P[i], Q[i]
                for k in angle_vars:
                    if k == i:
                        J[r, angle_col[k]] = Pi - self.G_full[i, i] * Vi * Vi
                    else:
                        Vk = Vmag[k]; ang = theta[i] - theta[k]
                        J[r, angle_col[k]] = -Vi * Vk * (self.G_full[i, k] * np.cos(ang) + self.B_full[i, k] * np.sin(ang))
                for k in V_vars:
                    if k == i:
                        L[r, V_col[k]] = Qi / Vi - self.B_full[i, i] * Vi
                    else:
                        ang = theta[i] - theta[k]
                        L[r, V_col[k]] = Vi * (self.G_full[i, k] * np.sin(ang) - self.B_full[i, k] * np.cos(ang))

            # ---- Newton step:  J * step = [ΔP; ΔQ]
            if len(Q_rows) == 0:
                Jmat = H
            else:
                Jmat = np.block([[H, N],
                                [J, L]])
            step = np.linalg.solve(Jmat, rhs)

            # Partition step to (θ,V_PQ)
            dtheta = step[:len(angle_vars)]
            dV     = step[len(angle_vars):] if len(V_vars) else np.array([])

            # ---- Backtracking if residual increases
            def residual_norm(Vtry, thtry):
                P2, Q2 = self._calc_PQ_injections_full(Vtry, thtry)
                pv_i = [i for i in range(n) if bus_type[i] == 1]
                pq_i = [i for i in range(n) if bus_type[i] == 2]
                rP = [P_spec[i] - P2[i] for i in (pv_i + pq_i)]
                rQ = [(-self.Q_full[i] + Qg[i]) - Q2[i] for i in pq_i]
                if rQ:
                    return max(np.max(np.abs(rP)), np.max(np.abs(rQ)))
                return np.max(np.abs(rP))

            res0  = np.max(np.abs(rhs))
            alpha = 1.0
            while alpha > 1/64:
                th_try = theta.copy(); V_try = Vmag.copy()
                for idx, bus in enumerate(angle_vars): th_try[bus] += alpha * dtheta[idx]
                for idx, bus in enumerate(V_vars):     V_try[bus]  += alpha * dV[idx]
                if residual_norm(V_try, th_try) < res0:
                    theta, Vmag = th_try, V_try
                    break
                alpha *= 0.5
            else:
                # accept tiny step
                for idx, bus in enumerate(angle_vars): theta[bus] += alpha * dtheta[idx]
                for idx, bus in enumerate(V_vars):     Vmag[bus]  += alpha * dV[idx]

            if it % 5 == 0 or it == itmax - 1:
                print(f"  [PF(full)] it={it:02d}  max|Δ|={res0:.3e}")

        if it == itmax - 1:
            print(f"  [PF(full)] Warning: Power flow did not fully converge in {itmax} iterations.")

        # ---- Finalize Pg, Qg from last state (and PVs within limits)
        P, Q = self._calc_PQ_injections_full(Vmag, theta)
        for i in range(n):
            if bus_type[i] != 2:              # PV or slack
                Qg[i] = Q[i] + self.Q_full[i]
        Pg_used = P + self.P_full
        Qg_used = Qg.copy()
        print(f"Full PF slack bus (1-based): {self.PF_SLACK_BUS}")
        return Vmag, theta, Pg_used, Qg_used, bus_type

    # =========================================================================
    # Power flow core (PF with PV buses and Q-limits)
    # =========================================================================

    def _set_efd_limits_from_Q_caps(self, margin=1.15):
        Vc0 = self.V_0 * np.exp(1j * self.theta_0)
        ang = np.exp(-1j * self.delta_0)
        alpha = (Vc0 * 1j / self.Xd_prime) * ang                    # coefficient of E'
        beta  = (Vc0 * 1j / self.Xd_prime) * (-np.conj(Vc0))        # constant term
        A = np.imag(alpha)
        B = np.imag(beta)

        # Guard: if |A| is tiny (pathological), fall back to broad limits
        tiny = 1e-6
        E_lo = np.empty(self.n_gen); E_hi = np.empty(self.n_gen)
        for i in range(self.n_gen):
            if abs(A[i]) < tiny:
                E_lo[i] = 0.5 * self.Efd_0[i]
                E_hi[i] = 2.5 * self.Efd_0[i]
                continue
            e_min = (self.Qg_min[i] - B[i]) / A[i]
            e_max = (self.Qg_max[i] - B[i]) / A[i]
            lo, hi = (min(e_min, e_max), max(e_min, e_max))
            # modest safety margin and nonnegativity
            E_lo[i] = max(0.2 * self.Efd_0[i], lo / margin)
            E_hi[i] = max(E_lo[i] + 1e-3, hi * margin)

        self.Efd_min = E_lo*1.6
        self.Efd_max = E_hi*1.6
        print("Efd limits from Q-caps set.")
    def _calc_PQ_injections(self, Vmag, theta):
        n = len(Vmag)
        P = np.zeros(n)
        Q = np.zeros(n)
        for i in range(n):
            for j in range(n):
                gij = self.G[i, j]
                bij = self.B[i, j]
                ang = theta[i] - theta[j]
                P[i] += Vmag[i] * Vmag[j] * (gij * np.cos(ang) + bij * np.sin(ang))
                Q[i] += Vmag[i] * Vmag[j] * (gij * np.sin(ang) - bij * np.cos(ang))
        return P, Q

    def _newton_pf(self, Vset, slack=0, tol=1e-6, itmax=40):
        n = self.n_gen
        bus_type = np.ones(n, dtype=int)  # 0=slack, 1=PV, 2=PQ
        bus_type[slack] = 0

        P_spec = self.Pg_target - self.PL
        theta = np.zeros(n)
        Vmag = Vset.copy()
        Qg = np.zeros(n)
        for it in range(itmax):
            P, Q = self._calc_PQ_injections(Vmag, theta)

            # -- PV→PQ at each iteration
            switched = False
            pv_now = [i for i in range(n) if bus_type[i] == 1]
            for i in pv_now:
                qgi = Q[i] + self.QL[i]
                Qg[i] = qgi
                if qgi < self.Qg_min[i] - 1e-8:
                    Qg[i] = self.Qg_min[i]; bus_type[i] = 2; switched = True
                elif qgi > self.Qg_max[i] + 1e-8:
                    Qg[i] = self.Qg_max[i]; bus_type[i] = 2; switched = True
            if switched:
                continue  # restart with new types

            pv_idx = [i for i in range(n) if bus_type[i] == 1]
            pq_idx = [i for i in range(n) if bus_type[i] == 2]

            mis_P = [P_spec[i] - P[i] for i in (pv_idx + pq_idx)]
            mis_Q = [(-self.QL[i] + Qg[i]) - Q[i] for i in pq_idx]
            rhs   = np.concatenate([mis_P, mis_Q]) if len(mis_Q) else np.array(mis_P)
            if np.max(np.abs(rhs)) < tol:
                break

            angle_vars = [i for i in range(n) if i != slack]
            V_vars     = pq_idx[:]
            H = np.zeros((len(pv_idx)+len(pq_idx), len(angle_vars)))
            N = np.zeros((len(pv_idx)+len(pq_idx), len(V_vars)))
            J = np.zeros((len(pq_idx), len(angle_vars)))
            L = np.zeros((len(pq_idx), len(V_vars)))

            angle_col = {bus:k for k,bus in enumerate(angle_vars)}
            V_col     = {bus:k for k,bus in enumerate(V_vars)}
            P_rows    = pv_idx + pq_idx
            Q_rows    = pq_idx

            for r,i in enumerate(P_rows):
                Vi, Pi, Qi = Vmag[i], P[i], Q[i]
                for k in angle_vars:
                    if k == i:
                        H[r, angle_col[k]] = -Qi - self.B[i,i]*Vi*Vi
                    else:
                        Vk = Vmag[k]; ang = theta[i] - theta[k]
                        H[r, angle_col[k]] = Vi*Vk*(self.G[i,k]*np.sin(ang) - self.B[i,k]*np.cos(ang))
                for k in V_vars:
                    if k == i:
                        N[r, V_col[k]] = Pi/Vi + self.G[i,i]*Vi
                    else:
                        ang = theta[i] - theta[k]
                        N[r, V_col[k]] = Vi*(self.G[i,k]*np.cos(ang) + self.B[i,k]*np.sin(ang))

            for r,i in enumerate(Q_rows):
                Vi, Pi, Qi = Vmag[i], P[i], Q[i]
                for k in angle_vars:
                    if k == i:
                        J[r, angle_col[k]] = Pi - self.G[i,i]*Vi*Vi
                    else:
                        Vk = Vmag[k]; ang = theta[i] - theta[k]
                        J[r, angle_col[k]] = -Vi*Vk*(self.G[i,k]*np.cos(ang) + self.B[i,k]*np.sin(ang))
                for k in V_vars:
                    if k == i:
                        L[r, V_col[k]] = Qi/Vi - self.B[i,i]*Vi
                    else:
                        ang = theta[i] - theta[k]
                        L[r, V_col[k]] = Vi*(self.G[i,k]*np.sin(ang) - self.B[i,k]*np.cos(ang))

            if len(Q_rows) == 0:
                Jmat = H
            else:
                Jmat = np.block([[H, N],
                                [J, L]])
            step = np.linalg.solve(Jmat, rhs)

            dtheta = step[:len(angle_vars)]
            dV     = step[len(angle_vars):] if len(V_vars) else np.array([])
            for idx,bus in enumerate(angle_vars): theta[bus] += dtheta[idx]
            for idx,bus in enumerate(V_vars):     Vmag[bus]  += dV[idx]


        if it == itmax - 1:
            print(f"  [PF] Warning: Power flow did not fully converge in {itmax} iterations.")

        P, Q = self._calc_PQ_injections(Vmag, theta)
        for i in range(n):
            if bus_type[i] != 2:  # PV or slack
                Qg[i] = Q[i] + self.QL[i]

        Pg_used = P + self.PL
        Qg_used = Qg.copy()
        return Vmag, theta, Pg_used, Qg_used, bus_type

    # =========================================================================
    # Equilibrium via PF + phasors (reconciled)
    # =========================================================================
    def find_equilibrium_pf_and_initialize(self, seed=7):
        n = self.n_gen
        if getattr(self, "FULL_NETWORK_PF", False):
            # Build a full-network Vset (1.0 everywhere except generator buses)
            Vset_full = np.ones(self.nbus, dtype=float)
            vmap = dict(zip([b-1 for b in self.gen_bus_order], self.Vset))
            for i in range(self.nbus):
                if i in vmap:
                    Vset_full[i] = vmap[i]

            slack_full = int(self.PF_SLACK_BUS) - 1   # bus 31 ⇒ index 30
            V_full, th_full, Pg_full, Qg_full, _ = self._newton_pf_full_network(
                Vset_full, slack_full=slack_full
            )

            # Project to kept generator buses (kept order) for DAE initialization
            sel = [b-1 for b in self.gen_bus_order]
            Vmag  = V_full[sel].copy()
            theta = th_full[sel].copy()
            Pg_used = Pg_full[sel].copy()
            Qg_used = Qg_full[sel].copy()
        else:
            Vmag, theta, Pg_used, Qg_used, _ = self._newton_pf(self.Vset, slack=0)

        # === NEW: remember the Ward PF loads (for reporting/CPF) ===
        self.PL_pf = self.PL.copy()
        self.QL_pf = self.QL.copy()

        # === NEW: calibrate ZIP so that at (Vmag,theta) it equals the Ward loads ===
        if self.zip_enabled:
            sP = 1.0 / (self.kZ_P * Vmag**2 + self.kI_P * Vmag + self.kP_P)
            sQ = 1.0 / (self.kZ_Q * Vmag**2 + self.kI_Q * Vmag + self.kP_Q)
            self.PL_zip = self.PL_pf * sP
            self.QL_zip = self.QL_pf * sQ
        else:
            self.PL_zip = self.PL_pf.copy()
            self.QL_zip = self.QL_pf.copy()
        Vc    = Vmag * np.exp(1j * theta)
        Inet  = self.Y @ Vc
        S_inj = Vc * np.conj(Inet)
        S_load = self.PL + 1j * self.QL
        Sg_cons = S_inj + S_load  # generator powers consistent with Y and loads

        try:
            Sg_pf = Pg_used + 1j * Qg_used
            mis = Sg_pf - Sg_cons
            print(f"  pre-reconcile max|S_mismatch| = {np.max(np.abs(mis)):.3e}")
        except Exception:
            pass

        Igen   = np.conj(Sg_cons / Vc)
        Eprime = Vc + 1j * self.Xd_prime * Igen
        delta  = np.angle(Eprime)
        Eqp    = np.abs(Eprime)

        self.V_0         = Vmag.copy()
        self.theta_0     = theta.copy()
        self.delta_0     = delta.copy()
        self.Eq_prime_0  = Eqp.copy()
        self.Efd_0       = Eqp.copy()
        self.Vref        = self.V_0 + self.Efd_0 / self.Ka
        self.omega_0     = np.ones(n)
        self.Pm_0        = Sg_cons.real.copy()

        cap_floor = 0.6   # pu on 100 MVA base (gives the slack a nonzero base)
        cap_base  = np.maximum(self.Pm_0, cap_floor)

        self.Qg_min = -0.8 * cap_base
        self.Qg_max = +0.8 * cap_base
        print("Qg limits set from Pm_0: ",np.round(self.Qg_min,3), np.round(self.Qg_max,3))
        self.Pvalve_0    = self.Pm_0.copy()
        self.Pref_0      = self.Pm_0.copy()

        if self.AVR_LIMITS:
            self._estimate_efd_limits()

        Pg_chk, _, _, _ = self.compute_network_power(self.V_0, self.theta_0, self.delta_0, self.Eq_prime_0)
        self.Pe_0 = Pg_chk.copy()

        print("\nEquilibrium (PF-backed, *generator powers reconciled from Y*):")
        print(f"  V range: {self.V_0.min():.4f} .. {self.V_0.max():.4f} pu")
        print(f"  ΣPm_0 = {self.Pm_0.sum():.4f} pu,  ΣPe_chk = {self.Pe_0.sum():.4f} pu  (should match)")
        PL_chk = self.PL_zip if getattr(self, "zip_enabled", False) and hasattr(self, "PL_zip") else self.PL
        QL_chk = self.QL_zip if getattr(self, "zip_enabled", False) and hasattr(self, "QL_zip") else self.QL

        PL0 = self.PL_zip if (getattr(self, "zip_enabled", False) and hasattr(self, "PL_zip")) else self.PL
        QL0 = self.QL_zip if (getattr(self, "zip_enabled", False) and hasattr(self, "QL_zip")) else self.QL
        rn0, _ = self.kcl_residual_norm(self.V_0, self.theta_0, self.delta_0, self.Eq_prime_0, PL0, QL0)

        print(f"  ||KCL||_2 at nominal (after reconcile) = {rn0:.2e}")

    def _estimate_efd_limits(self):
        n = self.n_gen
        Efd_min = 0.0 * self.Efd_0
        Efd_max = 0.0 * self.Efd_0
        for i in range(n):
            Efd_min[i] = self.AVR_EFD_MIN_MARGIN * self.Efd_0[i]
            Efd_max[i] = self.AVR_EFD_MAX_MARGIN * self.Efd_0[i]
        self.Efd_min = Efd_min
        self.Efd_max = Efd_max

        print("\nEstimated Efd limits from Q-limits (heuristic):")
        for i in range(n):
            print(f"  G{i+1}: Efd_min={self.Efd_min[i]:.3f}, Efd_0={self.Efd_0[i]:.3f}, Efd_max={self.Efd_max[i]:.3f}")


    # =========================================================================
    # Best-practice helper: emulate PSS®E-like PQBRAK behavior via CPL limiter
    # =========================================================================
    def set_pqbrak_like(self, vbreak: float = 0.85, smooth_eps: float = 0.06, gamma: float = None):
        """
        Configure a smooth, C¹ current limiter for the constant-power portion of static loads.
        If |V| >= vbreak: no effect. If |V| < vbreak: |I_cp| is limited to |S_cp|/vbreak.
        smooth_eps sets the width of the blending band (in units of r = |I|/Imax).
        """
        self.cpl_limit_enabled = True
        self.PQBRAK_VBREAK = float(vbreak)              # for printing/diagnostics
        self.CPL_SMOOTH_EPS = float(smooth_eps)         # <-- what _compute_load_currents uses
        self.CPL_Imax_gamma = (1.0 / self.PQBRAK_VBREAK) if gamma is None else float(gamma)  # <-- used
    def _saturate_cpl_current(self, I_cp: np.ndarray, S_cp: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Limit the constant-power current magnitude smoothly.
        Imax = |S_cp| / vbreak  -> saturation kicks in only when |V| < vbreak.
        """
        if not getattr(self, "cpl_limit_enabled", False):
            return I_cp

        eps = float(getattr(self, "pqbrak_smooth_eps", 0.06))
        vbreak = float(getattr(self, "pqbrak_vbreak", 0.85))

        absS = np.abs(S_cp)
        Imax = absS / max(vbreak, 1e-12)              # per-bus cap
        mag  = np.abs(I_cp)
        r = np.zeros_like(mag)
        mask = Imax > 0.0
        r[mask] = mag[mask] / Imax[mask]              # r <= 1: no cap; r > 1: cap

        scale = np.ones_like(mag)
        hard = r >= (1.0 + eps)
        mid  = (r > 1.0) & (~hard)

        # Hard cap region: scale to 1/r
        scale[hard] = 1.0 / r[hard]

        # Smooth C¹ blend on 1 < r < 1+eps (cubic Hermite)
        if np.any(mid):
            s = (r[mid] - 1.0) / eps                  # s in (0,1)
            w = s*s*(3.0 - 2.0*s)                     # 0→1 with C¹ endpoints
            scale[mid] = (1.0 - w) + w * (1.0 / r[mid])

        return I_cp * scale

    def _load_currents_zip_cpl(self, V: np.ndarray, PL_eff: np.ndarray, QL_eff: np.ndarray) -> np.ndarray:
        """
        Compute the *drawn* load current (phasor) for each kept bus:
        I_L = I_Z + I_I + I_CP_limited
        where Z/I/P weights use (kP_*, kI_*, kZ_*) in your code’s (kP, kI, kZ) order.
        Sign convention: returns the *drawn* current; if your KCL is in terms of *injected* current,
        subtract this from the network/generator injection.
        """
        n = V.size
        Vmag = np.abs(V)
        Vmag = np.where(Vmag > 1e-12, Vmag, 1e-12)      # avoid divide-by-zero
        Vdir = V / Vmag                                  # unit phasor in the voltage direction

        # ZIP split at the *current time*, including your global/per-bus scaling
        SP = self.kP_P * PL_eff + 1j * self.kP_Q * QL_eff
        SI = self.kI_P * PL_eff + 1j * self.kI_Q * QL_eff
        SZ = self.kZ_P * PL_eff + 1j * self.kZ_Q * QL_eff

        # Constant-impedance part: Y_Z = conj(SZ_nom)/V_nom^2, I_Z = Y_Z * V
        Vnom = np.abs(getattr(self, "V_0", np.ones(n)))   # use nominal magnitudes
        Vnom = np.where(Vnom > 1e-12, Vnom, 1.0)
        Yz = np.conj(SZ) / (Vnom**2)                      # admittance equivalent at nominal
        I_Z = Yz * V

        # Constant-current part: I magnitude fixed to nominal, aligned with V
        I_I_nom = np.conj(SI) / Vnom                      # |I| set by nominal
        I_I = I_I_nom * Vdir

        # Constant-power part: I_CP = conj(SP)/conj(V)  (draw current), then limit it
        I_CP = np.conj(SP) / np.conj(V)
        I_CP = self._saturate_cpl_current(I_CP, SP, V)    # PQBRAK-like limiter

        # Total *drawn* current
        return I_Z + I_I + I_CP

    # =========================================================================
    # Generator terminal powers (classical Norton)
    # =========================================================================
    def compute_network_power(self, V, theta, delta, Eq_prime):
        Vc      = V * np.exp(1j * theta)
        Eprime  = Eq_prime * np.exp(1j * delta)
        Yg      = -1j / self.Xd_prime

        Ig   = Yg * (Eprime - Vc)
        Sg   = Vc * np.conj(Ig)
        Pg   = Sg.real
        Qg   = Sg.imag

        Inet = self.Y @ Vc
        Snet = Vc * np.conj(Inet)
        P_net = Snet.real
        Q_net = Snet.imag
        return Pg, Qg, P_net, Q_net

    # =========================================================================
    # ZIP + current-limited CPL: currents and Jacobian pieces (C¹ limiter)
    # =========================================================================
    def _compute_load_currents(self, V, theta, PL_eff, QL_eff, need_jac=False):
        """
        SCALE-AWARE ZIP + C¹-smooth current-limited CPL on the constant-power portion.

        P(V) = PL_eff * (kZ_P V^2 + kI_P V + kP_P)
        Q(V) = QL_eff * (kZ_Q V^2 + kI_Q V + kP_Q)

        Currents per bus k:
          I_Z = (Pz - jQz) * V * e^{jθ}
          I_I = (Pi - jQi) * e^{jθ}
          I_P_unsat = ((Pp - jQp) * e^{jθ}) / V
          |I_P_unsat| limited smoothly to Imax = gamma * |S_P| / V_nom
          with a C¹ Hermite blend in |I|/Imax ∈ [1, 1+ε].

        Returns:
          Iload : complex(n,)
          dIdV  : complex(n,)  (diag ∂I_k/∂V_k)    [if need_jac]
          dIdth : complex(n,)  (diag ∂I_k/∂θ_k)    [if need_jac]
        """
        n = len(V)
        ejth = np.exp(1j * theta)

        Iload = np.zeros(n, dtype=complex)
        dIdV  = np.zeros(n, dtype=complex) if need_jac else None
        dIdth = np.zeros(n, dtype=complex) if need_jac else None

        eps_band = float(getattr(self, "CPL_SMOOTH_EPS", 0.12))
        gamma    = float(self.CPL_Imax_gamma)
        Vnom     = float(self.V_nom)
        tiny     = 1e-12

        for k in range(n):
            Vk = max(V[k], 1e-9)       # guard
            ej = ejth[k]

            # ZIP partitions
            Pp = PL_eff[k] * self.kP_P; Qp = QL_eff[k] * self.kP_Q
            Pi = PL_eff[k] * self.kI_P; Qi = QL_eff[k] * self.kI_Q
            Pz = PL_eff[k] * self.kZ_P; Qz = QL_eff[k] * self.kZ_Q

            # I and Z parts
            I_I = (Pi - 1j * Qi) * ej
            I_Z = (Pz - 1j * Qz) * Vk * ej

            # Constant-power portion with C¹ limiter
            SP_mag = np.hypot(Pp, Qp)  # |S_P|
            if SP_mag < tiny:
                I_P = 0.0 + 0.0j
                dI_P_dV = 0.0 + 0.0j
            else:
                C = (Pp - 1j * Qp)              # conj(S_P) at V=1 pu
                I_unsat = (C * ej) / Vk        # CPL current seed
                if not getattr(self, "cpl_limit_enabled", True):
                    I_P = I_unsat
                    dI_P_dV = -I_unsat / Vk
                else:
                    Imax = gamma * (SP_mag / Vnom)
                    x = np.abs(I_unsat) / max(Imax, tiny)   # ratio |I|/Imax
                    u = (C / SP_mag) * ej                   # unit PF vector (constant wrt V)

                    if x <= 1.0:
                        # unsaturated region
                        I_P = I_unsat
                        dI_P_dV = -I_unsat / Vk
                    elif x >= 1.0 + eps_band:
                        # fully saturated, constant magnitude Imax
                        I_P = Imax * u
                        dI_P_dV = 0.0 + 0.0j
                    else:
                        # C¹ Hermite blending across the band
                        s = (x - 1.0) / eps_band          # s in [0,1]
                        h = s * s * (3.0 - 2.0 * s)       # smoothstep h(s)
                        dhds = 6.0 * s * (1.0 - s)
                        # x = const / V  -> dx/dV = -x / V
                        dhdV = dhds * (1.0 / eps_band) * (-x / Vk)

                        z  = I_unsat                # depends on V
                        rU = Imax * u               # constant vs V

                        # I(V) = (1-h) z + h rU
                        # dI/dV = (1-h) dz/dV + (dh/dV) (rU - z)
                        dzz = -z / Vk               # dz/dV
                        I_P = (1.0 - h) * z + h * rU
                        dI_P_dV = (1.0 - h) * dzz + dhdV * (rU - z)

            Ik = I_I + I_Z + I_P
            Iload[k] = Ik

            if need_jac:
                # ∂I/∂V_k: I_Z contributes (Pz - jQz)*ej; I_I none; I_P as above
                dIdV[k]  = (Pz - 1j * Qz) * ej + dI_P_dV
                # ∂I/∂θ_k = j * I_k (all terms share e^{jθ_k})
                dIdth[k] = 1j * Ik

        return Iload, dIdV, dIdth

    # =========================================================================
    # Algebraic solver (analytic Jacobian, bounds, memoization, budgets)
    # =========================================================================

    ### NEW: pseudo-transient relaxer for (Efd, Eq′) only
    def _ptc_algebraic_recovery(self, t, y_diff, V_init, th_init,
                                PL_target, QL_target, pack, unpack, lower, upper, x_scale):
        """
        Physics-consistent micro-advance of Efd and Eq′ to re-establish KCL solvability.
        Keeps δ, ω, Pm, Pvalve *fixed* during this inner process (they will update
        normally through Radau). Uses your AVR and Eq′ ODEs with pseudo time steps.
        """
        if not self.PTC_ENABLED:
            return None

        n = self.n_gen
        delta    = y_diff[0:n]
        Eq_prime = y_diff[2*n:3*n].copy()
        Efd      = y_diff[3*n:4*n].copy()

        V = V_init.copy()
        theta = th_init.copy()
        x = pack(V, theta)

        # local closures that capture *current* Eq_prime (updated each loop)
        def build_residuals(Eq_local):
            Eprime = Eq_local * np.exp(1j * delta)
            Yg = -1j / self.Xd_prime

            def residual_only(z):
                Vv, th = unpack(z)
                ejth = np.exp(1j * th)
                Vc   = Vv * ejth
                Inet = self.Y @ Vc
                if self.zip_enabled:
                    Iload, _, _ = self._compute_load_currents(Vv, th, PL_target, QL_target, need_jac=False)
                else:
                    Iload = (PL_target - 1j * QL_target) * ejth / Vv
                Igen  = Yg * (Eprime - Vc)
                k = Inet + Iload - Igen
                return np.concatenate([k.real, k.imag[1:]])

            def jacobian_only(z):
                Vv, th = unpack(z)
                ejth = np.exp(1j * th)
                Vc   = Vv * ejth

                if self.zip_enabled:
                    Iload, dI_dV, dI_dth = self._compute_load_currents(Vv, th, PL_target, QL_target, need_jac=True)
                else:
                    Iload = (PL_target - 1j * QL_target) * ejth / Vv
                    dI_dV = np.zeros(n, dtype=complex)
                    for kk in range(n):
                        dI_dV[kk] = -Iload[kk] / max(Vv[kk], 1e-9)
                    dI_dth = 1j * Iload

                J = np.zeros((2*n - 1, 2*n - 1), dtype=float)

                def set_col(col_idx, dkdvar):
                    J[0:n, col_idx]         = dkdvar.real
                    J[n:(2*n - 1), col_idx] = dkdvar.imag[1:]

                for k in range(n):
                    dInet_dVk  = self.Y[:, k] * ejth[k]
                    dInet_dthk = self.Y[:, k] * (1j * Vc[k])

                    dIload_dVk  = np.zeros(n, dtype=complex); dIload_dVk[k]  = dI_dV[k]
                    dIload_dthk = np.zeros(n, dtype=complex); dIload_dthk[k] = dI_dth[k]

                    dnegIgen_dVk  = np.zeros(n, dtype=complex)
                    dnegIgen_dthk = np.zeros(n, dtype=complex)
                    dnegIgen_dVk[k]  =  (-1j / self.Xd_prime[k]) * ejth[k]
                    dnegIgen_dthk[k] =  (-1j / self.Xd_prime[k]) * (1j * Vc[k])

                    dk_dVk  = dInet_dVk  + dIload_dVk  + dnegIgen_dVk
                    dk_dthk = dInet_dthk + dIload_dthk + dnegIgen_dthk

                    set_col(k, dk_dVk)
                    if k >= 1:
                        set_col(n - 1 + k, dk_dthk)

                return J

            return residual_only, jacobian_only

        # PTC loop
        dt = float(self.PTC_DT0)
        phys_accum = 0.0
        success = False
        ls = None

        while phys_accum <= self.PTC_MAX_PHYS:
            residual_only, jacobian_only = build_residuals(Eq_prime)
            ls = least_squares(
                residual_only, x, jac=jacobian_only, method="trf",
                bounds=(lower, upper),
                ftol=1e-8, xtol=1e-8, gtol=1e-12, max_nfev=self.PTC_MAX_NEWTON,
                x_scale=x_scale
            )
            x = ls.x
            V, theta = unpack(x)
            k = residual_only(x)
            rn = float(np.linalg.norm(k))
            if rn < self.PTC_ACCEPT_RN and (np.min(V) >= self.PTC_MIN_V):
                success = True
                break

            # micro-advance AVR and field with *true* ODEs, using current V
            Vr = self.Ka * (self.Vref - V)
            if self.AVR_LIMITS:
                lo = getattr(self, "Efd_min", self.Efd_0 * 0.5)
                hi = getattr(self, "Efd_max", self.Efd_0 * 1.5)
                Vr = np.clip(Vr, lo, hi)
            Efd += dt * (Vr - Efd) / self.Ta
            Eq_prime += dt * (Efd - Eq_prime) / self.Td0_prime

            phys_accum += dt
            dt = min(self.PTC_DT_MAX, dt * 1.5)

        if self.PTC_VERBOSE and ls is not None:
            print(f"[PTC] t={t:.6f}s status={'OK' if success else 'FAIL'}  "
                  f"phys≈{phys_accum:.4f}s  rn≈{(float(np.linalg.norm(ls.fun)) if hasattr(ls,'fun') else np.nan):.2e}  "
                  f"Vmin={np.min(V):.3f}")

        if success:
            return {"V": V.copy(), "theta": theta.copy(),
                    "Eq_prime_eff": Eq_prime.copy(),
                    "Efd_eff": Efd.copy()}
        return None

    def _feasibility_backoff(self, x_seed, PL_prev, QL_prev, PL_tgt, QL_tgt,
                             pack, unpack, residual_only, jacobian_only,
                             lower, upper, x_scale):
        n = self.n_gen
        alpha_lo, alpha_hi = 0.0, 1.0
        x = x_seed.copy()
        best = None

        for _ in range(self.BACKOFF_MAX_ITERS):
            alpha = 0.5 * (alpha_lo + alpha_hi)
            PL = PL_prev*(1.0 - alpha) + PL_tgt*alpha
            QL = QL_prev*(1.0 - alpha) + QL_tgt*alpha

            ls = least_squares(
                lambda z: residual_only(z, PL, QL),
                x, jac=lambda z: jacobian_only(z, PL, QL),
                method="trf", bounds=(lower, upper),
                ftol=1e-8, xtol=1e-8, gtol=1e-12, max_nfev=300, x_scale=x_scale
            )
            x = ls.x
            V, theta = unpack(x)

            k = residual_only(x, PL, QL)
            rn = float(np.linalg.norm(k))
            at_bound = (np.min(V) <= self.V_LO + 5e-3) or (np.max(V) >= self.V_HI - 5e-3)

            if (rn < self.BACKOFF_TOL_RN) and (not at_bound) and (np.min(V) > self.BACKOFF_MIN_V):
                best = (V.copy(), theta.copy(), alpha)
                alpha_lo = alpha
            else:
                alpha_hi = alpha

            if (alpha_hi - alpha_lo) < 1e-3:
                break

        if best is not None and best[2] < 0.999:
            base = float(self.PL.sum())
            deltaP = float((PL_tgt - PL_prev).sum())
            scale_used = 1.0 + best[2] * (deltaP / max(base, 1e-12))
            print(f"[ALG] Feasibility back‑off: α={best[2]:.3f} ⇒ effective load scale ≈ {scale_used:.4f}; "
                  f"try slower ramp or enable ZIP.")
        return best if best is not None else (None, None, None)

    def solve_algebraic_equations(self, t, y_diff, return_jacobian=False):
                
        n = self.n_gen
        delta    = y_diff[0:n]
        Eq_prime = y_diff[2*n:3*n]

        # (optional) clear any effective Eq' override for RHS if you carry that state
        self._Eq_prime_eff_for_rhs = None

        # Detect if a load change is active now
        sP_now = 1.0
        if getattr(self, "disturbance", None) is not None:
            try:
                sP_now = self._load_scale(float(t))
            except Exception:
                sP_now = 1.0

        bus_scaled = (
            np.max(np.abs(getattr(self, "bus_P_scale", np.ones(self.n_gen)) - 1.0)) > 1e-12 or
            np.max(np.abs(getattr(self, "bus_Q_scale", np.ones(self.n_gen)) - 1.0)) > 1e-12
        )
        load_change_active = (abs(sP_now - 1.0) > 1e-12) or bus_scaled

        # “exactly nominal” fast path ONLY if nothing changed
        delta_change = float(np.max(np.abs(delta - self.delta_0)))
        Eq_change    = float(np.max(np.abs(Eq_prime - self.Eq_prime_0)))

        at_equilibrium = (
            (self.disturbance is None or t < float(self.disturbance.get("t_start", np.inf)) - 1e-12)
            and delta_change < 1e-8 and Eq_change < 1e-8
        )
        if at_equilibrium and (not load_change_active):
            # return cached nominal V,θ (set _alg_cache accordingly)
            self._last_V     = self.V_0.copy()
            self._last_theta = self.theta_0.copy()
            self._last_PL    = self.PL.copy()
            self._last_QL    = self.QL.copy()
            self._alg_cache = {
                "t": float(t), "delta": delta.copy(), "Eq": Eq_prime.copy(),
                "PL": self.PL.copy(), "QL": self.QL.copy(),
                "V": self.V_0.copy(), "theta": self.theta_0.copy()
            }
            if return_jacobian:
                return self.V_0.copy(), self.theta_0.copy(), None
            else:
                return self.V_0.copy(), self.theta_0.copy()

        # Otherwise, form targets via global × per-bus × ZIP
        PL_target, QL_target = self._effective_loads(float(t))
        
        
        # --- NEW: aggregate frequency-sensitive load damping (optional) ---
        if getattr(self, "LOAD_FREQ_DAMP", False):
            # compute COI frequency
            omega = y_diff[n:2*n]
            Mi = 2.0 * self.H / self.omega_s
            omega_coi = float(np.dot(Mi, omega) / np.sum(Mi))
            s_f = 1.0 + float(self.D_load) * (omega_coi - 1.0)
            # clamp to keep realism for large events
            s_f = min(max(s_f, 0.90), 1.10)
            PL_target = PL_target * s_f

        if hasattr(self, "_alg_cache"):
            same_time = abs(float(t) - self._alg_cache["t"]) < self.CACHE_TOL_T
            if same_time:
                if (np.max(np.abs(delta    - self._alg_cache["delta"])) < self.CACHE_TOL_Y and
                    np.max(np.abs(Eq_prime - self._alg_cache["Eq"]))    < self.CACHE_TOL_Y and
                    np.max(np.abs(PL_target - self._alg_cache["PL"]))   < 1e-16 and
                    np.max(np.abs(QL_target - self._alg_cache["QL"]))   < 1e-16):
                    if return_jacobian:
                        return self._alg_cache["V"].copy(), self._alg_cache["theta"].copy(), None
                    else:
                        return self._alg_cache["V"].copy(), self._alg_cache["theta"].copy()

        if not hasattr(self, "_alg_calls_total"):
            self._alg_calls_total = 0; self._alg_cont_runs = 0; self._alg_time_sec = 0.0
        t0 = time.perf_counter(); self._alg_calls_total += 1

        Eprime = Eq_prime * np.exp(1j * delta)
        Yg     = -1j / self.Xd_prime
        theta_ref = float(self.theta_0[0])

        # Initial guess
        if hasattr(self, "_last_V") and hasattr(self, "_last_theta"):
            V_init  = self._last_V.copy(); th_init = self._last_theta.copy()
        else:
            V_init, th_init = self.V_0.copy(), self.theta_0.copy()

        def pack(V, theta):
            return np.concatenate([V, theta[1:]])

        def unpack(x):
            V = x[:n]
            theta = np.empty(n)
            theta[0] = theta_ref
            theta[1:] = x[n:]
            return V, theta

        # Residual and analytic Jacobian
        def residual_only(x, PL, QL):
            V, theta = unpack(x)
            ejth = np.exp(1j * theta)
            Vc   = V * ejth
            Inet = self.Y @ Vc
            if getattr(self, "zip_enabled", False):
                Iload, _, _ = self._compute_load_currents(V, theta, PL, QL, need_jac=False)
            else:
                Iload = (PL - 1j * QL) * ejth / V
            Igen  = Yg * (Eprime - Vc)
            k = Inet + Iload - Igen
            if not np.all(np.isfinite(k)):
                k = np.nan_to_num(k, nan=1e6, posinf=1e6, neginf=-1e6)
            return np.concatenate([k.real, k.imag[1:]])

        def jacobian_only(x, PL, QL):
            V, theta = unpack(x)
            ejth = np.exp(1j * theta)
            Vc   = V * ejth

            if getattr(self, "zip_enabled", False):
                Iload, dI_dV, dI_dth = self._compute_load_currents(V, theta, PL, QL, need_jac=True)
            else:
                Iload = (PL - 1j * QL) * ejth / V
                dI_dV = np.zeros(n, dtype=complex)
                for k in range(n):
                    dI_dV[k] = -Iload[k] / max(V[k], 1e-9)
                dI_dth = 1j * Iload

            J = np.zeros((2*n - 1, 2*n - 1), dtype=float)

            def set_col_from_complex(col_idx, dkdvar):
                J[0:n, col_idx]         = dkdvar.real
                J[n:(2*n - 1), col_idx] = dkdvar.imag[1:]

            for k in range(n):
                dInet_dVk  = self.Y[:, k] * ejth[k]
                dInet_dthk = self.Y[:, k] * (1j * Vc[k])

                dIload_dVk  = np.zeros(n, dtype=complex); dIload_dVk[k]  = dI_dV[k]
                dIload_dthk = np.zeros(n, dtype=complex); dIload_dthk[k] = dI_dth[k]

                dnegIgen_dVk  = np.zeros(n, dtype=complex)
                dnegIgen_dthk = np.zeros(n, dtype=complex)
                dnegIgen_dVk[k]  =  Yg[k] * ejth[k]
                dnegIgen_dthk[k] =  Yg[k] * (1j * Vc[k])

                dk_dVk  = dInet_dVk  + dIload_dVk  + dnegIgen_dVk
                dk_dthk = dInet_dthk + dIload_dthk + dnegIgen_dthk

                set_col_from_complex(k, dk_dVk)
                if k >= 1:
                    set_col_from_complex(n - 1 + k, dk_dthk)

            return J

        # Bounds & scaling
        lower = np.concatenate([np.full(n, self.V_LO), np.full(n-1, self.TH_LO)])
        upper = np.concatenate([np.full(n, self.V_HI), np.full(n-1, self.TH_HI)])
        x_scale = np.concatenate([np.ones(n), 0.3*np.ones(n-1)])

        # Decide on continuation
        need_continuation = False
        if hasattr(self, "_last_PL"):
            load_change_1norm = float(np.sum(np.abs(PL_target - self._last_PL)))
            base = float(self.PL.sum())
            need_continuation = (load_change_1norm >= self.CONT_THRESHOLD * base) or bool(self._force_continuation)

        # --- ensure continuation on the FIRST post-step solve ---
        if getattr(self, "_step_fired", False):
            need_continuation = True
        V, theta = V_init.copy(), th_init.copy()
        x0 = pack(V, theta)
        x = x0.copy()

        def elapsed_exceeded():
            return (time.perf_counter() - t0) > self.ALG_SOLVE_TIME_BUDGET

        # Quick attempt first (bounded TRF)
        if not need_continuation:
            res = lambda z: residual_only(z, PL_target, QL_target)
            jac = lambda z: jacobian_only(z, PL_target, QL_target)
            ls = least_squares(
                res, x0, jac=jac, method="trf",
                bounds=(lower, upper),
                ftol=1e-8, xtol=1e-8, gtol=1e-12,
                max_nfev=80, x_scale=x_scale
            )
            x = ls.x
            V, theta = unpack(x)
            kq = res(x)
            if (np.linalg.norm(kq) < self.BACKOFF_TOL_RN) and (np.min(V) < self.BACKOFF_MIN_V) and not elapsed_exceeded():
                need_continuation = True
            elif ls.cost >= 1e-3 and not elapsed_exceeded():
                need_continuation = True

        # Continuation (homotopy) if needed or forced
        if need_continuation and not elapsed_exceeded():
            self._alg_cont_runs += 1
            PL_start = getattr(self, "_last_PL", self.PL.copy())
            QL_start = getattr(self, "_last_QL", self.QL.copy())

            base = float(self.PL.sum())
            rel = float(np.sum(np.abs(PL_target - PL_start))) / max(base, 1e-12)
            n_steps = max(4, min(16, int(np.ceil(6 * max(rel, 0.05)))))

            x = x0.copy()
            for i, alpha in enumerate(np.linspace(0.0, 1.0, n_steps + 1)[1:]):
                if elapsed_exceeded():
                    print(f"[ALG] budget exceeded @ t={t:.6f} (cont i={i}/{n_steps})")
                    break
                PL_cont = PL_start + alpha * (PL_target - PL_start)
                QL_cont = QL_start + alpha * (QL_target - QL_start)
                res = lambda z, PLc=PL_cont, QLc=QL_cont: residual_only(z, PLc, QLc)
                jac = lambda z, PLc=PL_cont, QLc=QL_cont: jacobian_only(z, PLc, QLc)

                ls = least_squares(
                    res, x, jac=jac, method="trf",
                    bounds=(lower, upper),
                    ftol=1e-7 if i < n_steps-1 else 1e-9,
                    xtol=1e-7 if i < n_steps-1 else 1e-9,
                    gtol=1e-12, max_nfev=300,
                    x_scale=x_scale
                )
                x = ls.x
            V, theta = unpack(x)

        # Quality gate / feasibility rescue
        res_target = lambda z: residual_only(z, PL_target, QL_target)
        k_target = res_target(pack(V, theta))
        rn_target = float(np.linalg.norm(k_target))
        hit_bound = (np.min(V) <= self.V_LO + 5e-3) or (np.max(V) >= self.V_HI - 5e-3)
        low_voltage = (np.min(V) < self.BACKOFF_MIN_V)

        # --- NEW: slightly looser gate on the first post-step continuation
        self_tol = self.BACKOFF_TOL_RN
        if getattr(self, "_step_fired", False) and (getattr(self, "_alg_cont_runs", 0) < 2):
            self_tol = max(self_tol, 5e-1)

        if (rn_target > self_tol or hit_bound or low_voltage):
            # --- try back-off first
            V_bk, th_bk, alpha = self._feasibility_backoff(
                pack(V, theta),
                getattr(self, "_last_PL", self.PL.copy()),
                getattr(self, "_last_QL", self.QL.copy()),
                PL_target, QL_target,
                pack, unpack, residual_only, jacobian_only,
                lower, upper, x_scale
            )
            if V_bk is not None:
                V, theta = V_bk, th_bk
            else:
                # --- physics-based pseudo-transient field relaxer
                rec = self._ptc_algebraic_recovery(
                    t, y_diff, V, theta, PL_target, QL_target,
                    pack, unpack, lower, upper, x_scale
                )
                if rec is not None:
                    V = rec["V"]; theta = rec["theta"]
                    # remember effective Eq′ to use for Pe in this RHS
                    self._Eq_prime_eff_for_rhs = rec["Eq_prime_eff"].copy()  # ### NEW
                else:
                    # fall back silently; still return best found V,θ (rare)
                    print(f"[ALG] Infeasible for current (δ,E′) even with back-off and PTC "
                          f"(rn={rn_target:.2e}, minV={np.min(V):.3f}). "
                          f"Consider slower ramp or stronger ZIP/CPL limits.")

        # Cache & instrumentation
        self._last_V     = V.copy()
        self._last_theta = theta.copy()
        self._last_PL    = PL_target.copy()
        self._last_QL    = QL_target.copy()

        self._alg_cache = {
            "t": float(t), "delta": y_diff[0:n].copy(), "Eq": y_diff[2*n:3*n].copy(),
            "PL": PL_target.copy(), "QL": QL_target.copy(),
            "V": V.copy(), "theta": theta.copy()
        }

        self._alg_time_sec += (time.perf_counter() - t0)

        if return_jacobian:
            try:
                J_alg = jacobian_only(pack(V, theta), PL_target, QL_target)
            except Exception:
                J_alg = None
            return V, theta, J_alg
        else:
            return V, theta



    def _same_state_as_cache(self, t: float, delta: np.ndarray, Eq_prime: np.ndarray) -> bool:
        C = getattr(self, "_alg_cache", None)
        if C is None or C.get("t", None) != float(t):
            return False
        # Require *very* tight equality of differential states at same t
        if delta.shape != C["delta"].shape or Eq_prime.shape != C["Eq"].shape:
            return False
        if np.max(np.abs(delta   - C["delta"]))   > 1e-13:  # tighter than solver eps
            return False
        if np.max(np.abs(Eq_prime - C["Eq"]))     > 1e-13:
            return False
        return True

    # =========================================================================
    # DAE RHS
    # =========================================================================
    def dae_rhs(self, t, y):
        # Step detection: force continuation once at the step time
        if getattr(self, "disturbance", None):
            fired = getattr(self, "_step_fired", False)
            t_start = float(self.disturbance["t_start"])
            if (not fired) and (t >= t_start - 1e-12):
                self._step_fired = True
                self._force_continuation = True

        # Instrument RHS call patterns (detect integrator hovering)
        if not hasattr(self, "_rhs_calls_total"):
            self._rhs_calls_total = 0
            self._rhs_same_t_cluster = 0
            self._last_rhs_t = None
            self._last_cluster_log_t = None
            self._rhs_time_sec = 0.0
        rhs_t0 = time.perf_counter()
        self._rhs_calls_total += 1

        is_stalling = False
        if self._last_rhs_t is None or abs(t - self._last_rhs_t) > self.CACHE_TOL_T:
            self._rhs_same_t_cluster = 0
        else:
            self._rhs_same_t_cluster += 1
            if self._rhs_same_t_cluster >= self.RHS_SAME_T_CLUSTER_MAX:
                is_stalling = True
                if (self._last_cluster_log_t is None) or abs(t - self._last_cluster_log_t) > self.CACHE_TOL_T:
                    print(f"[RHS] Warning: same‑t cluster @{t:.6f} s: {self._rhs_same_t_cluster} callbacks. Running algebraic Jacobian SVD next.")
                    self._last_cluster_log_t = t
        self._last_rhs_t = t

        n = self.n_gen
        idx = 0
        delta = y[idx:idx+n]; idx += n
        omega = y[idx:idx+n]; idx += n
        Eq_prime = y[idx:idx+n]; idx += n
        Efd = y[idx:idx+n]; idx += n
        Pm = y[idx:idx+n]; idx += n
        Pvalve = y[idx:idx+n]; idx += n

        # Algebraic solve
        if is_stalling:
            V, theta, J_alg = self.solve_algebraic_equations(t, y, return_jacobian=True)
        else:
            V, theta = self.solve_algebraic_equations(t, y, return_jacobian=False)
            J_alg = None

        # Use effective Eq′ if PTC provided one for this RHS; otherwise the state value
        Eq_prime_for_power = getattr(self, "_Eq_prime_eff_for_rhs", None)  # ### NEW
        if Eq_prime_for_power is None:
            Eq_prime_for_power = Eq_prime

        # Electrical power
        Pg, Qg, _, _ = self.compute_network_power(V, theta, delta, Eq_prime_for_power)
        Pe = Pg

        # ODE
        dydt = np.zeros(6*n)
        dydt[0:n] = self.omega_s * (omega - 1.0)  # dδ/dt
        for i in range(n):
            dydt[n+i] = (self.omega_s / (2.0*self.H[i])) * (Pm[i] - Pe[i] - self.D[i]*(omega[i]-1.0))  # dω/dt
        dydt[2*n:3*n] = (Efd - Eq_prime) / self.Td0_prime  # dE'q/dt

        # AVR (with optional limiter)
        for i in range(n):
            Vr = self.Ka[i] * (self.Vref[i] - V[i])
            if self.AVR_LIMITS:
                Efd_next = Efd[i] + (Vr - Efd[i]) / self.Ta[i]
                lo = getattr(self, "Efd_min", self.Efd_0 * 0.5)[i]
                hi = getattr(self, "Efd_max", self.Efd_0 * 1.5)[i]
                if (Efd[i] <= lo and Vr < Efd[i]) or (Efd[i] >= hi and Vr > Efd[i]):
                    dydt[3*n+i] = 0.0
                else:
                    dydt[3*n+i] = (Vr - Efd[i]) / self.Ta[i]
            else:
                dydt[3*n+i] = (Vr - Efd[i]) / self.Ta[i]

        dydt[4*n:5*n] = (Pvalve - Pm) / self.Tt  # dPm/dt
        for i in range(n):
            Pref = self.Pref_0[i] + (1.0 - omega[i]) / self.R[i]
            Pref = np.clip(Pref, 0.5*self.Pref_0[i], 1.5*self.Pref_0[i])
            dydt[5*n+i] = (Pref - Pvalve[i]) / self.Tg[i]  # dPvalve/dt

        self._rhs_time_sec += (time.perf_counter() - rhs_t0)
# --- Sanitize RHS for SciPy's num_jac perturbations ---
        dydt = np.asarray(dydt, dtype=float)
        if dydt.ndim != 1:
            dydt = dydt.reshape(-1)
        if not np.all(np.isfinite(dydt)):
            # one shot rescue: force a fresh algebraic re-solve and rebuild RHS
            _was = getattr(self, "_force_continuation", False)
            self._force_continuation = True
            try:
                V_safe, th_safe = self.solve_algebraic_equations(t, y)
                # compose RHS again with the freshly solved algebraics
                n = self.n_gen
                delta    = y[0:n]
                omega    = y[n:2*n]
                Eq_prime = y[2*n:3*n]
                Efd      = y[3*n:4*n]
                Pm       = y[4*n:5*n]
                Pvalve   = y[5*n:6*n]

                Eq_for_power = Eq_prime if getattr(self, "_Eq_prime_eff_for_rhs", None) is None else self._Eq_prime_eff_for_rhs
                Pg, Qg, _, _ = self.compute_network_power(V_safe, th_safe, delta, Eq_for_power)
                Pe = Pg

                dydt = np.zeros(6*n, dtype=float)
                dydt[0:n]   = self.omega_s * (omega - 1.0)
                for i in range(n):
                    dydt[n+i] = (Pm[i] - Pe[i] - self.D[i]*(omega[i]-1.0)) / self.M[i]
                # AVR & GOV blocks (same formulas as above)
                for i in range(n):
                    Vr = self.Ka[i] * (self.Vref_0[i] - V_safe[i]) - self.Kf[i] * self.ifd[i]
                    dydt[3*n+i] = (Vr - Efd[i]) / self.Ta[i]
                dydt[4*n:5*n] = (Pvalve - Pm) / self.Tt
                for i in range(n):
                    Pref = self.Pref_0[i] + (1.0 - omega[i]) / self.R[i]
                    Pref = np.clip(Pref, 0.5*self.Pref_0[i], 1.5*self.Pref_0[i])
                    dydt[5*n+i] = (Pref - Pvalve[i]) / self.Tg[i]
            finally:
                self._force_continuation = _was
            if not np.all(np.isfinite(dydt)):
                raise RuntimeError("dae_rhs produced non-finite values during Jacobian estimation.")


        return dydt

    # =========================================================================
    # Helper functions
    # =========================================================================
    def _reset_step_state(self):
        self._step_scale = 1.0
        self._step_fired = False
        self._force_continuation = False

        # clear caches/instrumentation that can make the solver hover
        for attr in ["_last_V", "_last_theta", "_last_PL", "_last_QL", "_alg_cache",
                    "_last_rhs_t", "_rhs_calls_total", "_rhs_same_t_cluster",
                    "_rhs_time_sec", "_alg_calls_total", "_alg_cont_runs", "_alg_time_sec",
                    "_last_cluster_log_t", "_Eq_prime_eff_for_rhs"]:
            if hasattr(self, attr):
                delattr(self, attr)


    def _load_scale(self, t: float) -> float:
        if getattr(self, "disturbance", None) is None:
            return 1.0
        t0 = float(self.disturbance["t_start"])
        dP = float(self.disturbance["delta_P"])
        base = float(self.PL.sum())
        if t <= t0 - 1e-12:
            return 1.0
        # if ramp keys are absent or zero → TRUE STEP
        ramp_tau = float(self.disturbance.get("ramp_tau", 0.0))
        ramp_T   = float(self.disturbance.get("ramp_T", 0.0))
        if ramp_tau > 0.0:
            return 1.0 + (dP/base) * (1.0 - np.exp(-(t - t0)/ramp_tau))
        if ramp_T > 0.0:
            a = (t - t0) / ramp_T
            a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
            return 1.0 + (dP/base) * a
        return 1.0 + dP/base
    def _effective_loads(self, t: float):
        sP = self._load_scale(t)
        eta = 0.0                     # if you want Q to follow P 1:1, keep 0
        sQ = 1.0 + eta * (sP - 1.0)

        # Base static loads: ZIP-calibrated if enabled (preserves nominal PF)
        if getattr(self, "zip_enabled", False) and hasattr(self, "PL_zip"):
            PLb = self.PL_zip.copy()
            QLb = self.QL_zip.copy()
        else:
            PLb = self.PL.copy()
            QLb = self.QL.copy()

        # Per-bus multipliers; created in simulate()
        Pscale = getattr(self, "bus_P_scale", np.ones_like(PLb))
        Qscale = getattr(self, "bus_Q_scale", np.ones_like(QLb))

        PL_eff = PLb * sP * Pscale
        QL_eff = QLb * sQ * Qscale
        return PL_eff, QL_eff

    # ---- NEW: helper to set mechanical damping to hit a target primary slope β (MW/Hz)
    def set_mechanical_damping_from_beta(self, beta_MW_per_Hz: float, share: str = "H", D_load: float = 0.0):
        """
        Compute and assign the mechanical damping array self.D so that the
        total primary slope β (MW/Hz) is approximated by:
            β ≈ sum_i (1/R_i) + D_load + sum_i D_i     (converted to pu/pu internally)
        where D_load is the aggregate load-damping coefficient (pu(P)/pu(ω)).

        Parameters
        ----------
        beta_MW_per_Hz : float
            Target primary slope in MW/Hz on 100 MVA base.
        share : {"H","Pg"}
            How to distribute the required mechanical damping across units.
            "H" → proportional to inertia H_i; "Pg" → proportional to scheduled Pg_target.
        D_load : float
            Aggregate load-damping coefficient in pu(P)/pu(ω) to account for in β.

        Notes
        -----
        The conversion to pu/pu uses f_base = 60 Hz and S_base = 100 MVA.
        """
        f_base = self.base_freq            # Hz
        S_base = 100.0                     # MVA (IEEE-39 base)
        beta_pu = (beta_MW_per_Hz * f_base) / S_base
        droop_dc = float(np.sum(1.0 / self.R))
        D_sys_mech = max(0.0, beta_pu - D_load - droop_dc)

        if share == "H":
            w = self.H / np.sum(self.H)
        elif share == "Pg":
            total = max(np.sum(self.Pg_target), 1e-9)
            w = self.Pg_target / total
        else:
            raise ValueError("share must be 'H' or 'Pg'")
        self.D = D_sys_mech * w

    # =========================================================================
    # Diagnostics / verification helpers
    # =========================================================================
    def kcl_residual_norm(self, V, theta, delta, Eq_prime, PL, QL):
        Vc     = V * np.exp(1j * theta)
        Eprime = Eq_prime * np.exp(1j * delta)
        Yg     = -1j / self.Xd_prime
        Inet   = self.Y @ Vc
        if getattr(self, "zip_enabled", False):
            Iload, _, _ = self._compute_load_currents(V, theta, PL, QL, need_jac=False)
        else:
            Iload  = (PL - 1j*QL) * np.exp(1j*theta) / V
        Igen   = Yg * (Eprime - Vc)
        k      = Inet + Iload - Igen
        return float(np.linalg.norm(k)), k

    def check_post_step_feasibility(self, scale=1.05,
                                    delta=None, Eq_prime=None,
                                    V_init=None, theta_init=None,
                                    n_steps=12, tol_final=1e-8, verbose=True):
        n = self.n_gen
        if delta is None:     delta      = self.delta_0.copy()
        if Eq_prime is None: Eq_prime = self.Eq_prime_0.copy()
        if V_init is None:   V_init   = self.V_0.copy()
        if theta_init is None: theta_init = self.theta_0.copy()
        PL_base = self.PL_zip if (getattr(self, "zip_enabled", False) and hasattr(self, "PL_zip")) else self.PL
        QL_base = self.QL_zip if (getattr(self, "zip_enabled", False) and hasattr(self, "QL_zip")) else self.QL

        PL_tgt = PL_base * scale
        QL_tgt = QL_base * scale

        Eprime = Eq_prime * np.exp(1j * delta)
        Yg     = -1j / self.Xd_prime
        theta_ref = float(self.theta_0[0])

        def pack(V, th):   return np.concatenate([V, th[1:]])
        def unpack(x):
            V = x[:n]; th = np.empty(n); th[0]=theta_ref; th[1:] = x[n:]; return V, th

        def residual(x, PL, QL):
            V, th = unpack(x); ejth = np.exp(1j*th); Vc = V*ejth
            if getattr(self, "zip_enabled", False):
                Iload, _, _ = self._compute_load_currents(V, th, PL, QL, need_jac=False)
            else:
                Iload = (PL - 1j*QL)*ejth/V
            k = (self.Y @ Vc) + Iload - Yg*(Eprime - Vc)
            return np.concatenate([k.real, k.imag[1:]])

        def jacobian(x, PL, QL):
            V, th = unpack(x); ejth = np.exp(1j*th); Vc = V*ejth
            if getattr(self, "zip_enabled", False):
                Iload, dI_dV, dI_dth = self._compute_load_currents(V, th, PL, QL, need_jac=True)
            else:
                Iload = (PL - 1j*QL)*ejth/V
                dI_dV = np.zeros(n, dtype=complex)
                for k in range(n):
                    dI_dV[k] = -Iload[k] / max(V[k], 1e-9)
                dI_dth = 1j * Iload

            J = np.zeros((2*n-1, 2*n-1), float)
            def put(c, dv): J[0:n, c]=dv.real; J[n:(2*n-1), c]=dv.imag[1:]

            for k in range(n):
                dInet_dVk  = self.Y[:,k]*ejth[k]
                dInet_dthk = self.Y[:,k]*(1j*Vc[k])
                dIload_dVk  = np.zeros(n, complex); dIload_dVk[k]  = dI_dV[k]
                dIload_dthk = np.zeros(n, complex); dIload_dthk[k] = dI_dth[k]
                dnegIgen_dVk  = np.zeros(n, complex); dnegIgen_dVk[k]  =  (-1j/self.Xd_prime[k])*ejth[k]
                dnegIgen_dthk = np.zeros(n, complex); dnegIgen_dthk[k] =  (-1j/self.Xd_prime[k])*(1j*Vc[k])
                put(k, dInet_dVk + dIload_dVk + dnegIgen_dVk)
                if k>=1:
                    put(n-1+k, dInet_dthk + dIload_dthk + dnegIgen_dthk)
            return J

        V_lo, V_hi = self.V_LO, self.V_HI
        TH_lo, TH_hi = self.TH_LO, self.TH_HI
        lower = np.concatenate([np.full(n, V_lo), np.full(n-1, TH_lo)])
        upper = np.concatenate([np.full(n, V_hi), np.full(n-1, TH_hi)])
        x_scale = np.concatenate([np.ones(n), 0.3*np.ones(n-1)])

        x = pack(V_init, theta_init)
        alphas = np.linspace(0, 1, n_steps+1)[1:]
        for i, a in enumerate(alphas):
            PL = PL_base*(1-a) + PL_tgt*a
            QL = QL_base*(1-a) + QL_tgt*a
            ls = least_squares(lambda z: residual(z, PL, QL),
                               x, jac=lambda z: jacobian(z, PL, QL),
                               method="trf",
                               bounds=(lower, upper),
                               ftol=1e-7 if i < n_steps-1 else tol_final,
                               xtol=1e-7 if i < n_steps-1 else tol_final,
                               gtol=1e-12, max_nfev=600, x_scale=x_scale)
            x = ls.x
        V_fin, th_fin = unpack(x)

        Jfin = jacobian(x, PL_tgt, QL_tgt)
        svals = np.linalg.svd(Jfin, compute_uv=False)
        smin = float(np.min(svals)); smax = float(np.max(svals))
        cond = float(smax/smin) if smin>0 else np.inf

        rn, _ = self.kcl_residual_norm(V_fin, th_fin, delta, Eq_prime, PL_tgt, QL_tgt)
        ok = rn < 1e-6

        info = {
            "feasible": bool(ok),
            "residual_norm": rn,
            "min_voltage": float(np.min(V_fin)),
            "jac_smin": smin,
            "jac_cond": cond,
            "V": V_fin, "theta": th_fin
        }
        if verbose:
            print(f"[post-step snapshot] feasible={ok}, ||KCL||_2={rn:.2e}, "
                  f"min(V)={info['min_voltage']:.3f}, smin(J)={smin:.2e}, cond(J)={cond:.2e}")
        return info

    def find_static_margin(self, s_lo=1.00, s_hi=1.10, tol=1e-3, verbose=True):
        def test(scale):
            info = self.check_post_step_feasibility(scale=scale, verbose=False)
            return info["feasible"], info

        fe_lo, info_lo = test(s_lo)
        fe_hi, info_hi = test(s_hi)

        if verbose:
            print(f"[margin] s_lo={s_lo:.4f} feasible={fe_lo}, ||KCL||={info_lo['residual_norm']:.2e}")
            print(f"[margin] s_hi={s_hi:.4f} feasible={fe_hi}, ||KCL||={info_hi['residual_norm']:.2e}")

        if not fe_lo:
            return {"s_crit": None, "note": "Baseline not solvable", "info_lo": info_lo, "info_hi": info_hi}
        if fe_hi:
            return {"s_crit": s_hi, "note": "Upper bound still solvable; increase s_hi", "info_lo": info_lo, "info_hi": info_hi}

        left, right = s_lo, s_hi
        info_left, info_right = info_lo, info_hi
        while right - left > tol:
            mid = 0.5 * (left + right)
            fe_mid, info_mid = test(mid)
            if verbose:
                print(f"[margin] s_mid={mid:.4f} feasible={fe_mid}, ||KCL||={info_mid['residual_norm']:.2e}")
            if fe_mid:
                left, info_left = mid, info_mid
            else:
                right, info_right = mid, info_mid

        return {"s_crit": left, "info_lo": info_left, "info_hi": info_right}

    def cpf_power_flow_margin(self, s_lo=1.00, s_hi=1.20, ds=0.01, slack=0, tol=1e-7, itmax=40):
        n = self.n_gen
        PL0 = self.PL.copy()
        QL0 = self.QL.copy()
        Vset = self.Vset.copy()

        V, th, Pg, Qg, bus_type = self._newton_pf(Vset, slack=slack, tol=tol, itmax=itmax)
        events = []
        hist = []

        def pf_jacobian(Vmag, theta, bus_type):
            angle_vars = [i for i in range(n) if i != slack]
            pv_idx = [i for i in range(n) if bus_type[i] == 1]
            pq_idx = [i for i in range(n) if bus_type[i] == 2]
            P, Q = self._calc_PQ_injections(Vmag, theta)
            H = np.zeros((len(pv_idx)+len(pq_idx), len(angle_vars)))
            N = np.zeros((len(pv_idx)+len(pq_idx), len(pq_idx)))
            J = np.zeros((len(pq_idx), len(angle_vars)))
            L = np.zeros((len(pq_idx), len(pq_idx)))
            angle_col = {bus:k for k,bus in enumerate(angle_vars)}
            V_col = {bus:k for k,bus in enumerate(pq_idx)}
            for r,i in enumerate(pv_idx + pq_idx):
                Vi, Pi, Qi = Vmag[i], P[i], Q[i]
                for k in angle_vars:
                    if k == i:
                        H[r, angle_col[k]] = -Qi - self.B[i,i]*Vi*Vi
                    else:
                        Vk = Vmag[k]; ang = theta[i]-theta[k]
                        H[r, angle_col[k]] = Vi*Vk*(self.G[i,k]*np.sin(ang) - self.B[i,k]*np.cos(ang))
                for k in pq_idx:
                    if k == i:
                        N[r, V_col[k]] = Pi/Vi + self.G[i,i]*Vi
                    else:
                        ang = theta[i]-theta[k]
                        N[r, V_col[k]] = Vi*(self.G[i,k]*np.cos(ang) + self.B[i,k]*np.sin(ang))
            for r,i in enumerate(pq_idx):
                Vi, Pi, Qi = Vmag[i], P[i], Q[i]
                for k in angle_vars:
                    if k == i:
                        J[r, angle_col[k]] = Pi - self.G[i,i]*Vi*Vi
                    else:
                        Vk = Vmag[k]; ang = theta[i]-theta[k]
                        J[r, angle_col[k]] = -Vi*Vk*(self.G[i,k]*np.cos(ang) + self.B[i,k]*np.sin(ang))
                for k in pq_idx:
                    if k == i:
                        L[r, V_col[k]] = Qi/Vi - self.B[i,i]*Vi
                    else:
                        ang = theta[i]-theta[k]
                        L[r, V_col[k]] = Vi*(self.G[i,k]*np.sin(ang) - self.B[i,k]*np.cos(ang))
            if len(pq_idx) == 0:
                Jmat = H
            else:
                Jmat = np.block([[H, N],[J, L]])
            return Jmat, pv_idx, pq_idx

        s = s_lo
        s_star = s_lo
        try:
            while s <= s_hi + 1e-12:
                self.PL = PL0 * s
                self.QL = QL0 * s

                V_prev, th_prev = V.copy(), th.copy()
                V, th, Pg, Qg, bus_type = self._newton_pf(Vset, slack=slack, tol=tol, itmax=itmax)

                q_hits = [i for i in range(n) if bus_type[i] == 2]
                if q_hits:
                    events.append((s, "PV→PQ", q_hits))

                Jpf, pv_idx, pq_idx = pf_jacobian(V, th, bus_type)
                try:
                    svals = np.linalg.svd(Jpf, compute_uv=False)
                    smin = float(np.min(svals)); smax = float(np.max(svals))
                    cond = float(smax/smin) if smin > 0 else np.inf
                except Exception:
                    smin, cond = 0.0, np.inf

                vmin = float(np.min(V))
                hist.append({"s": s, "vmin": vmin, "sminJ": smin, "condJ": cond,
                             "pv_count": len(pv_idx), "pq_count": len(pq_idx)})
                if not np.isfinite(cond) or smin < 1e-5 or vmin < 0.85:
                    break

                s_star = s
                s += ds

        finally:
            self.PL = PL0; self.QL = QL0

        print("\n== CPF summary ==")
        for rec in hist:
            print(f"  s={rec['s']:.4f}  vmin={rec['vmin']:.3f}  smin(J)={rec['sminJ']:.2e}  cond(J)={rec['condJ']:.2e}  PV={rec['pv_count']} PQ={rec['pq_count']}")
        for ev in events:
            print(f"  @ s={ev[0]:.4f}: {ev[1]} buses {ev[2]}")
        print(f"  --> CPF margin s* ≈ {s_star:.4f}")
        return {"s_star": s_star, "history": hist, "events": events, "V": V, "theta": th, "Qg": Qg, "bus_type": bus_type}

    # =========================================================================
    # Verification
    # =========================================================================
    def verify_equilibrium(self):
        print("\nVerifying equilibrium derivatives at t=0 …")
        n = self.n_gen
        y0 = np.concatenate([
            self.delta_0, self.omega_0, self.Eq_prime_0,
            self.Efd_0, self.Pm_0, self.Pvalve_0
        ])

        self._last_V = self.V_0.copy()
        self._last_theta = self.theta_0.copy()

        dydt = self.dae_rhs(0.0, y0)

        names = ["δ", "ω", "E'q", "Efd", "Pm", "Pvalve"]
        tol = 5e-6
        ok = True
        for k in range(6):
            seg = dydt[k*n:(k+1)*n]
            m = np.max(np.abs(seg))
            print(f"  |d{names[k]}/dt|_max = {m:.2e} {'✓' if m < tol else '✗'}")
            ok &= (m < tol)
        # --- choose the same static load model used by the algebraic solver at nominal ---
        PL0 = self.PL_zip if (getattr(self, "zip_enabled", False) and hasattr(self, "PL_zip")) else self.PL
        QL0 = self.QL_zip if (getattr(self, "zip_enabled", False) and hasattr(self, "QL_zip")) else self.QL

        rn0, _ = self.kcl_residual_norm(self.V_0, self.theta_0, self.delta_0, self.Eq_prime_0,
                                        PL0, QL0)
        print(f"  max|Pe - Pm| = {np.max(np.abs(self.Pe_0 - self.Pm_0)):.2e} "
            f"{'✓' if np.max(np.abs(self.Pe_0 - self.Pm_0)) < 5e-7 else '✗'}")
        print(f"  ||KCL||_2 at nominal = {rn0:.2e} {'✓' if rn0 < 1e-8 else '✗'}")

        if ok and rn0 < 1e-8:
            print("  ✓ TRUE EQUILIBRIUM ACHIEVED")
        else:
            print("  ⚠ Small residuals remain; check network scaling if needed.")
        return ok

    # =========================================================================
    # Simulation (piecewise across step + post-processing algebraics)
    # =========================================================================

    def _post_process_solution(self, t_eval, y_eval):
        V_hist, Theta_hist, Pe_hist = [], [], []
        # prime cache
        if len(t_eval) > 0:
            self.solve_algebraic_equations(t_eval[0], y_eval[:, 0])
        for t_i, y_i in zip(t_eval, y_eval.T):
            V_i, Theta_i = self.solve_algebraic_equations(t_i, y_i)
            n = self.n_gen
            delta_i = y_i[0:n]
            Eq_prime_i = y_i[2*n:3*n]
            # SAFE fallback: use state Eq′ unless an override array exists
            Eq_for_power = Eq_prime_i if getattr(self, "_Eq_prime_eff_for_rhs", None) is None else self._Eq_prime_eff_for_rhs
            Pe_i, _, _, _ = self.compute_network_power(V_i, Theta_i, delta_i, Eq_for_power)
            V_hist.append(V_i)
            Theta_hist.append(Theta_i)
            Pe_hist.append(Pe_i)
        return np.array(V_hist).T, np.array(Theta_hist).T, np.array(Pe_hist).T
    def simulate(self, t_span=(0.0, 20.0), disturbance=None, split_at_step=True, n_eval_points=500):
        """
        Multi-interval simulator with optional global disturbance and per-bus alter events.
        - If split_at_step: the global step time is a breakpoint.
        - Per-bus Alter events are applied at their exact times.
        - At each event boundary, do ONE warm algebraic solve and clear one-shot flags.
        """
        # Make disturbance visible to dae_rhs/_load_scale
        self.disturbance = disturbance

        # Reset caches/state and per-bus scales
        self._reset_step_state()
        self.bus_P_scale = np.ones(self.n_gen)
        self.bus_Q_scale = np.ones(self.n_gen)
        for ev in self.alter_events:
            ev["applied"] = False

        # Initial state vector
        y0 = np.concatenate([
            self.delta_0, self.omega_0, self.Eq_prime_0,
            self.Efd_0, self.Pm_0, self.Pvalve_0
        ])
        t0, tf = float(t_span[0]), float(t_span[1])

        # -------- Breakpoints
        bps = [t0, tf]
        if split_at_step and (disturbance is not None):
            t_step = float(disturbance.get("t_start", np.inf))
            if t0 < t_step < tf:
                bps.append(t_step)
        for ev in self.alter_events:
            te = float(ev["t"])
            if t0 < te < tf:
                bps.append(te)

        # Sort, unique, and drop near-duplicates
        bps = sorted(set(bps))
        bps2 = [bps[0]]
        for x in bps[1:]:
            if x - bps2[-1] > 1e-12:
                bps2.append(x)
        bps = bps2

        # -------- Inner segment runner
        def run_segment(a, b, y_init, first_seg=False):
            total_T = tf - t0
            seg_T = b - a
            seg_pts = max(3, int(round(n_eval_points * seg_T / max(total_T, 1e-9))))
            if first_seg and len(bps) == 2:     # single-segment case
                seg_pts = n_eval_points
            t_eval = np.linspace(a, b, seg_pts)
            sol = solve_ivp(self.dae_rhs, (a, b), y_init,
                            method="Radau", rtol=1e-6, atol=1e-8,
                            max_step=0.05, t_eval=t_eval)
            return sol

        # -------- March segments and apply events at boundaries
        sols = []
        y_init = y0.copy()
        for k in range(len(bps) - 1):
            a, b = bps[k], bps[k+1]
            solk = run_segment(a, b, y_init, first_seg=(k == 0))
            if not solk.success:
                print(f"  ✗ Failure in segment [{a}, {b}]: {solk.message}")
                return solk
            sols.append(solk)
            y_init = solk.y[:, -1].copy()

            # Apply events at boundary 'b' if not the final boundary
            if k < len(bps) - 2:
                fired_global = (
                    split_at_step and (disturbance is not None) and
                    abs(b - float(disturbance.get("t_start", np.inf))) < 1e-12
                )
                events_here = []
                for ev in self.alter_events:
                    if (not ev.get("applied", False)) and abs(float(ev["t"]) - b) < 1e-12:
                        idx = int(ev["bus_idx"]); amt = float(ev["amount"])
                        self.bus_P_scale[idx] = amt
                        self.bus_Q_scale[idx] = amt
                        ev["applied"] = True
                        events_here.append((idx, amt))
                for idx, amt in events_here:
                    print(f"[event] Applied alter_event at t={b:.4f}: bus {idx+1} scale→{amt:.3f}")

                if fired_global or events_here:
                    # one-shot continuation for the very next algebraic solve
                    self._force_continuation = True
                    self._step_fired = True
                    self._last_rhs_t = None
                    self._rhs_same_t_cluster = 0

                    # Warm algebraic cache at t = b+ (using state at the boundary)
                    _ = self.solve_algebraic_equations(b + 1e-9, y_init)

                    # CLEAR flags so continuation/PTC cannot re-engage later
                    self._force_continuation = False
                    self._step_fired = False

        # -------- Stitch segments
        t_all = sols[0].t.copy()
        y_all = sols[0].y.copy()
        for solk in sols[1:]:
            t_all = np.concatenate([t_all, solk.t[1:]])
            y_all = np.concatenate([y_all, solk.y[:, 1:]], axis=1)
        sol = SimpleNamespace(t=t_all, y=y_all, success=True)

        # -------- Algebraic post-processing along the trajectory
        V, Theta, Pe = self._post_process_solution(sol.t, sol.y)
        sol.V = V; sol.theta = Theta; sol.Pe = Pe
        return sol

    # =========================================================================
    # Plotting
    # =========================================================================
    def plot_results(self, sol, title="Simulation Results"):
        if not getattr(sol, "success", False):
            print("Cannot plot: Simulation was not successful or data is missing.")
            return

        n = self.n_gen
        t = sol.t
        delta = sol.y[0:n, :]
        omega = sol.y[n:2*n, :]
        Eq_prime = sol.y[2*n:3*n, :]
        Efd = sol.y[3*n:4*n, :]
        Pm = sol.y[4*n:5*n, :]
        Pvalve = sol.y[5*n:6*n, :]
        V = sol.V
        Pe = sol.Pe

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        ax = axes[0,0]
        for i in range(min(5, n)):
            ax.plot(t, (omega[i,:] - 1.0) * self.base_freq, label=f"G{i+1}", alpha=0.85)
        ax.axhline(0, color="k", ls="--", alpha=0.3); ax.grid(True, alpha=0.3)
        ax.set_title("Frequency Deviation (Hz)"); ax.set_xlabel("t (s)"); ax.set_ylabel("Δf (Hz)"); ax.legend(fontsize=8)

        ax = axes[0,1]
        for i in range(min(5, n)):
            ax.plot(t, V[i,:], label=f"Bus {i+1}", alpha=0.85)
        ax.axhline(1.0, color="k", ls="--", alpha=0.3); ax.grid(True, alpha=0.3)
        ax.set_title("Voltages (pu)"); ax.set_xlabel("t (s)"); ax.set_ylabel("V (pu)"); ax.legend(fontsize=8)

        ax = axes[0,2]
        for i in range(min(3, n)):
            ax.plot(t, Pm[i,:] - Pe[i,:], label=f"G{i+1}", alpha=0.85)
        ax.axhline(0, color="k", ls="--", alpha=0.3); ax.grid(True, alpha=0.3)
        ax.set_title("Power Imbalance Pm - Pe (pu)"); ax.set_xlabel("t (s)"); ax.set_ylabel("pu"); ax.legend(fontsize=8)

        ax = axes[1,0]
        for i in range(1, min(5, n)):
            ax.plot(t, np.degrees(delta[i,:] - delta[0,:]), label=f"G{i+1}", alpha=0.85)
        ax.grid(True, alpha=0.3)
        ax.set_title("Rotor Angles (deg rel. G1)"); ax.set_xlabel("t (s)"); ax.set_ylabel("deg"); ax.legend(fontsize=8)

        ax = axes[1,1]
        for i in range(min(5, n)):
            ax.plot(t, Pe[i,:], label=f"G{i+1}", alpha=0.85)
        ax.grid(True, alpha=0.3)
        ax.set_title("Electrical Power Pe (pu)"); ax.set_xlabel("t (s)"); ax.set_ylabel("pu"); ax.legend(fontsize=8)

        ax = axes[1,2]
        for i in range(min(5, n)):
            ax.plot(t, Pm[i,:], label=f"G{i+1}", alpha=0.85)
        ax.grid(True, alpha=0.3)
        ax.set_title("Mechanical Power Pm (pu)"); ax.set_xlabel("t (s)"); ax.set_ylabel("pu"); ax.legend(fontsize=8)

        ax = axes[2,0]
        for i in range(min(5, n)):
            ax.plot(t, Efd[i,:], label=f"G{i+1}", alpha=0.85)
        ax.grid(True, alpha=0.3)
        ax.set_title("Field Voltage Efd (pu)"); ax.set_xlabel("t (s)"); ax.set_ylabel("pu"); ax.legend(fontsize=8)

        ax = axes[2,1]
        for i in range(min(5, n)):
            ax.plot(t, (omega[i,:] - 1.0) * self.base_freq * 1000.0, label=f"G{i+1}", alpha=0.85)
        ax.axhline(0, color="k", ls="--", alpha=0.3); ax.grid(True, alpha=0.3)
        ax.set_title("Speed Deviation (mHz)"); ax.set_xlabel("t (s)"); ax.set_ylabel("mHz"); ax.legend(fontsize=8)

        ax = axes[2,2]
        ax.plot(t, np.sum(Pm, axis=0), label="ΣPm", lw=2)
        ax.plot(t, np.sum(Pe, axis=0), label="ΣPe", lw=2, ls="--")
        ax.grid(True, alpha=0.3)
        ax.set_title("Total Power"); ax.set_xlabel("t (s)"); ax.set_ylabel("pu"); ax.legend()

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        return fig
    def add_alter_event(self, bus_idx, t, amount):
        """Per-bus static load change at time t.
        bus_idx: 0..n-1 in reduced kept-bus order.
        amount: absolute multiplier on both P and Q (e.g., 1.10 for +10%)."""
        self.alter_events.append({
            "bus_idx": int(bus_idx),
            "t": float(t),
            "amount": float(amount),
            "applied": False
        })


# -------------------------
# Main execution (self-test)
# -------------------------
if __name__ == "__main__":
    system = FixedPowerDAESystem()


    # ---- Frozen (δ0,E'0) static margin ----
    m = system.find_static_margin(s_lo=1.00, s_hi=1.10, tol=5e-4)
    scrit = m["s_crit"] if m["s_crit"] is not None else 1.0

    # ---- CPF margin (PV buses & Q limits) ----
    cpf = system.cpf_power_flow_margin(s_lo=1.00, s_hi=1.15, ds=0.01)
    print(f"CPF margin s* ≈ {cpf['s_star']:.4f}")

    # ---- Quick transient-feasibility snapshot at +5% (frozen δ,E′) ----
    _ = system.check_post_step_feasibility(scale=1.05, n_steps=20)
    print("ZIP:", system.zip_enabled,
        "| CPL limiter:", system.cpl_limit_enabled,
        "| vbreak:", getattr(system, "PQBRAK_VBREAK", None),
        "| eps:", getattr(system, "CPL_SMOOTH_EPS", None),
        "| gamma:", getattr(system, "CPL_Imax_gamma", None))

    # ---- Test 1: No disturbance ----
    sol1 = system.simulate(t_span=(0, 5), disturbance=None, split_at_step=True, n_eval_points=300)
    if getattr(sol1, "success", False):
        system.plot_results(sol1, "IEEE-39 Reduced — No Disturbance")
        plt.show()
        max_df = np.max(np.abs(sol1.y[system.n_gen:2*system.n_gen, :] - 1.0))
        max_dV = np.max(np.abs(sol1.V - system.V_0[:, None]))
        print("\nEquilibrium check:")
        print(f"  Max |Δω|: {max_df:.2e} pu")
        print(f"  Max |ΔV|: {max_dV:.2e} pu")


   # 1. IMPORTANT: Disable the global disturbance
    disturbance = None

    # 2. Clear any alter events from previous runs
    system.alter_events = []

    # 3. Add a per-bus load increase event
    # Let's increase the load on the original Bus 39 (index 9) by 15% at t=2.0s
    target_bus_idx = 6  # Corresponds to original Bus 39
    change_time = 2.0
    change_amount = 1.05 # 15% increase
    
    system.add_alter_event(bus_idx=target_bus_idx, t=change_time, amount=change_amount)
    print(f"Scheduled Event: Load at Bus index {target_bus_idx} to be scaled by {change_amount} at t={change_time}s.")

    # You can add multiple events
    # For example, let's also decrease the load on Bus 34 (index 4) by 10% at t=3.0s
    # system.add_alter_event(bus_idx=4, t=3.0, amount=0.90)

    # Now, run the simulation with these new settings
    sol = system.simulate(t_span=(0, 6), disturbance=disturbance, split_at_step=True, n_eval_points=600)

    if getattr(sol, "success", False):
        title = f"IEEE-39 — Per-Bus Load Change (+{(change_amount-1.0):.0%} at Bus {target_bus_idx+1})"
        system.plot_results(sol, title)
        plt.show()
