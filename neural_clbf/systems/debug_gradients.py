import torch
from neural_clbf.systems.IEEE39ControlAffineDAE import IEEE39ControlAffineDAE

def ensure_batch(t: torch.Tensor) -> torch.Tensor:
    return t if t.dim() == 2 else t.unsqueeze(0)

def run_equilibrium_and_gradient_debug():
    torch.manual_seed(0)

    sys = IEEE39ControlAffineDAE(
        nominal_params={"pv_ratio": [0]*10, "T_pv": (0.05, 0.05)},
        dt=0.001,
    )



    # in debug script, right after sys creation
    delta_rel, omega, Eqp, Efd, Pm, Pvalve, Ppv, Qpv = sys._unpack_state(sys.goal_point[0])
    delta = sys._angle_reconstruct(delta_rel)
    V, th = sys._V_goal, sys._theta_goal            # cached in projection

    rep_star = sys.audit_pf_at(V, th, delta, Eqp, Ppv, Qpv)
    print("PF audit @ projection cache:", rep_star)

    # Now build f at V*,θ* (you can add a variant of _f that accepts (V,θ) to avoid calling Newton)

    # ------------------ A) EQUILIBRIUM VERIFICATION ------------------
    print("\n=== EQUILIBRIUM VERIFICATION ===")
    x_eq = ensure_batch(sys.goal_point).clone().requires_grad_(True)   # (1, D)
    u_eq = ensure_batch(sys.u_eq).clone().requires_grad_(True)         # (1, C)

    # Compute f, g and build xdot explicitly (avoid base closed_loop_dynamics)
    f, g = sys.control_affine_dynamics(x_eq, params=None)              # f: (1,D), g: (1,D,C)
    xdot_eq = f + torch.bmm(g, u_eq.unsqueeze(-1)).squeeze(-1)         # (1, D)

    D = xdot_eq.shape[1]
    n = sys.n_gen
    idx_delta_rel = slice(0, n - 1)
    idx_omega     = slice(n - 1, 2*n - 1)
    idx_Eq_prime  = slice(2*n - 1, 3*n - 1)
    idx_Efd       = slice(3*n - 1, 4*n - 1)
    idx_Pm        = slice(4*n - 1, 5*n - 1)
    idx_Pvalve    = slice(5*n - 1, 6*n - 1)
    idx_P_pv      = slice(6*n - 1, 7*n - 1)
    idx_Q_pv      = slice(7*n - 1, 8*n - 1)

    def bn(slc): return torch.norm(xdot_eq[0, slc]).item()

    print(f"||xdot|| at eq:         {torch.norm(xdot_eq).item():.4e}")
    print(f"  d(delta_rel)/dt norm: {bn(idx_delta_rel):.4e}")
    print(f"  d(omega)/dt norm:     {bn(idx_omega):.4e}")
    print(f"  d(Eq')/dt norm:       {bn(idx_Eq_prime):.4e}")
    print(f"  d(Efd)/dt norm:       {bn(idx_Efd):.4e}")
    print(f"  d(Pm)/dt norm:        {bn(idx_Pm):.4e}")
    print(f"  d(Pvalve)/dt norm:    {bn(idx_Pvalve):.4e}")
    print(f"  d(P_pv)/dt norm:      {bn(idx_P_pv):.4e}")
    print(f"  d(Q_pv)/dt norm:      {bn(idx_Q_pv):.4e}")

    # Algebraic KCL residual at equilibrium (differentiable)
    delta_rel, omega, Eqp, Efd, Pm, Pvalve, Ppv, Qpv = sys._unpack_state(x_eq[0])
    delta = sys._angle_reconstruct(delta_rel)
    V, theta = sys._solve_kcl_newton(delta, Eqp, Ppv, Qpv)             # keep graph
    z = torch.cat([theta[1:], V], dim=0)
    F = sys._kcl_residual(z, delta, Eqp, Ppv, Qpv)
    print(f"||KCL residual|| at eq: {torch.norm(F).item():.4e}")

    eq_ok = (torch.norm(xdot_eq) < 1e-6) and (torch.norm(F) < 1e-6)
    print("EQUILIBRIUM:", "✅ OK" if eq_ok else "❌ needs attention")

    # ------------------ B) GRADIENT DEBUGGING ------------------
    print("\n=== GRADIENT DEBUGGING ===")
    x = x_eq.clone().detach().requires_grad_(True)
    u = u_eq.clone().detach().requires_grad_(True)
    x.data += 1e-4 * torch.randn_like(x)
    u.data += 1e-4 * torch.randn_like(u)

    f, g = sys.control_affine_dynamics(x, params=None)
    xdot = f + torch.bmm(g, u.unsqueeze(-1)).squeeze(-1)
    J = (xdot**2).sum() + 1e-6*(x**2).sum() + 1e-6*(u**2).sum()
    J.backward()

    print(f"Loss J:          {float(J):.6e}")
    print(f"||∂J/∂x||:       {float(x.grad.norm()):.4e}")
    print(f"||∂J/∂u||:       {float(u.grad.norm()):.4e}")
    print(f"||∂J/∂x_Ppv||:   {float(x.grad[0, idx_P_pv].norm()):.4e}")
    print(f"||∂J/∂x_Qpv||:   {float(x.grad[0, idx_Q_pv].norm()):.4e}")

    # Quick FD vs BP sanity check (float32 -> loose)
    eps = 1e-5
    i_x = (sys.n_gen - 1) + 0  # first omega slot
    i_u = 0                    # first PV-P control
    def loss_from(x_, u_):
        # freeze warm-starts so both FD probes evaluate the same function
        sys._last_V = sys._V_goal.clone().detach()
        sys._last_theta = sys._theta_goal.clone().detach()
        f_, g_ = sys.control_affine_dynamics(x_, params=None)
        xdot_ = f_ + torch.bmm(g_, u_.unsqueeze(-1)).squeeze(-1)
        return (xdot_**2).sum()


    eps_x = 1e-5
    eps_u = 1e-4   # controls: use a bit larger step


    x_p = x.clone().detach().requires_grad_(True)
    u_h = u.clone().detach().requires_grad_(True)
    x_plus  = x_p.clone().detach().requires_grad_(True); x_plus.data[0, i_x] += eps
    x_minus = x_p.clone().detach().requires_grad_(True); x_minus.data[0, i_x] -= eps
    fd_x = (loss_from(x_plus, u_h) - loss_from(x_minus, u_h)) / (2*eps)

    x_h = x.clone().detach().requires_grad_(True)
    u_p = u.clone().detach().requires_grad_(True)
    u_plus  = u_p.clone().detach().requires_grad_(True); u_plus.data[0, i_u] += eps
    u_minus = u_p.clone().detach().requires_grad_(True); u_minus.data[0, i_u] -= eps
    fd_u = (loss_from(x_h, u_plus) - loss_from(x_h, u_minus)) / (2*eps)

    print(f"FD dJ/dx[{i_x}]: {float(fd_x):.4e}")
    print(f"BP dJ/dx[{i_x}]: {float(x.grad[0, i_x]):.4e}")
    print(f"FD dJ/du[{i_u}]: {float(fd_u):.4e}")
    print(f"BP dJ/du[{i_u}]: {float(u.grad[0, i_u]):.4e}")
    # Masked vs full residual (square-Newton uses masked)
    # After you have (delta_rel, Eqp, Ppv, Qpv) for the equilibrium state:
    delta = sys._angle_reconstruct(delta_rel)
    n = sys.n_gen

    # Rebuild z from the Newton output so we can recompute F
    V, theta = sys._last_V, sys._last_theta
    z = torch.cat([theta[1:], V], dim=0)

    # Full residual
    F_full = sys._kcl_residual(z, delta, Eqp, Ppv, Qpv)

    # Mask matching the solver's dropped row
    drop_kind, drop_bus = getattr(sys, "kcl_row_drop", ("imag", 0))
    drop_idx = (drop_bus if drop_kind == "real" else n + drop_bus)
    mask = torch.ones(2*n, dtype=torch.bool, device=F_full.device); mask[drop_idx] = False

    F_masked = F_full[mask]
    print(f"||F_masked|| at eq: {float(F_masked.norm()):.6e}")
    print(f"||F_full||   at eq: {float(F_full.norm()):.6e}")

# Check AVR consistency at eq
    with torch.no_grad():
# unpack goal state
        delta_rel, omega, Eqp, Efd, Pm, Pvalve, Ppv, Qpv = sys._unpack_state(sys.goal_point[0])
        delta = sys._angle_reconstruct(delta_rel)

        # choose V*, θ* EXACTLY as used in equilibrium builder
        V_star     = sys.Vset.clone()
        theta_star = torch.zeros_like(V_star)

        rep = sys.audit_pf_at(V_star, theta_star, delta, Eqp, Ppv, Qpv)
        print("PF audit @ goal (no Newton):")
        print("  ||k||:", rep["k_norm"], "  max|ΔP|:", rep["dS_P_max"], "  max|ΔQ|:", rep["dS_Q_max"])


        V_hat, theta_hat = sys._solve_kcl_newton(delta, Eqp, Ppv, Qpv)
        rep2 = sys.audit_pf_at(V_hat, theta_hat, delta, Eqp, Ppv, Qpv)
        print("PF audit @ Newton solution:")
        print("  ||k||:", rep2["k_norm"], "  max|ΔP|:", rep2["dS_P_max"], "  max|ΔQ|:", rep2["dS_Q_max"])
        rep_goal = sys.audit_pf_at(sys._V_goal, sys._theta_goal, delta, Eqp, Ppv, Qpv)
        print("PF audit @ cached projection point:",
            "||k||:", rep_goal["k_norm"], "max|ΔP|:", rep_goal["dS_P_max"], "max|ΔQ|:", rep_goal["dS_Q_max"])

if __name__ == "__main__":
    run_equilibrium_and_gradient_debug()
