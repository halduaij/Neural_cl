import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

from neural_clbf.systems import SwingEquationSystem
from neural_clbf.systems.utils import Scenario, ScenarioList
from neural_clbf.controllers import CLFController, BCController
from neural_clbf.experiments import ExperimentSuite

def load_bc_controller(
    system: SwingEquationSystem,
    scenarios: ScenarioList,
    experiment_suite: ExperimentSuite,
    model_path: str = "bc_controller.pt",
) -> BCController:
    """Load a trained BC controller.

    args:
        system: the swing equation system
        scenarios: list of scenarios
        experiment_suite: the experiment suite
        model_path: path to the saved model
    returns:
        bc_controller: the loaded BC controller
    """
    # Create BC controller
    bc_controller = BCController(
        dynamics_model=system,
        scenarios=scenarios,
        experiment_suite=experiment_suite,
        hidden_layers=[64, 64],
        learning_rate=1e-3,
        batch_size=32,
        controller_period=0.01,
    )
    
    # Load state dict
    bc_controller.load_state_dict(torch.load(model_path))
    
    return bc_controller

def verify_stability(
    system: SwingEquationSystem,
    bc_controller: BCController,
    clf_controller: CLFController,
    n_samples: int = 1000,
    n_levels: int = 10,
) -> Tuple[float, float, Dict[str, torch.Tensor]]:
    """Verify the stability of the BC controller using the pre-BC Lyapunov function.

    args:
        system: the swing equation system
        bc_controller: the BC controller
        clf_controller: the CLF controller (for Lyapunov function)
        n_samples: number of samples to evaluate
        n_levels: number of level sets to evaluate
    returns:
        violation_rate: rate of Lyapunov condition violations
        avg_violation: average violation magnitude
        metrics: dictionary of additional stability metrics
    """
    # Sample random states
    lower_lim, upper_lim = system.state_limits
    states = torch.rand(n_samples, system.n_dims) * (upper_lim - lower_lim) + lower_lim
    
    # Compute Lyapunov function values and derivatives
    V = clf_controller.V(states)
    Lf_V, Lg_V = clf_controller.V_lie_derivatives(states)
    
    # Get BC controller actions
    u_bc = bc_controller.u(states)
    
    # Compute Vdot for BC controller
    Vdot = Lf_V[:, 0, :] + torch.bmm(
        Lg_V[:, 0, :].unsqueeze(1),
        u_bc.reshape(-1, system.n_controls, 1),
    ).squeeze()
    
    # Check Lyapunov condition: Vdot + lambda*V <= 0
    violations = torch.relu(Vdot + clf_controller.clf_lambda * V)
    violation_rate = (violations > 1e-5).float().mean().item()
    avg_violation = violations.mean().item()
    
    # Additional metrics
    metrics = {
        'V': V,
        'Vdot': Vdot,
        'violations': violations,
        'u_bc': u_bc,
    }
    
    # Analyze stability at different level sets
    V_sorted, _ = torch.sort(V)
    level_indices = torch.linspace(0, n_samples-1, n_levels).long()
    V_levels = V_sorted[level_indices]
    
    level_metrics = []
    for level in V_levels:
        mask = (V <= level)
        if mask.any():
            level_violations = violations[mask]
            level_rate = (level_violations > 1e-5).float().mean().item()
            level_avg = level_violations.mean().item()
            level_metrics.append({
                'level': level.item(),
                'violation_rate': level_rate,
                'avg_violation': level_avg,
            })
    
    metrics['level_metrics'] = level_metrics
    
    return violation_rate, avg_violation, metrics

def plot_stability_analysis(
    metrics: Dict[str, torch.Tensor],
    save_path: str = 'stability_analysis.png',
) -> None:
    """Plot stability analysis results.

    args:
        metrics: dictionary of stability metrics
        save_path: path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Plot V vs Vdot
    plt.subplot(2, 2, 1)
    plt.scatter(metrics['V'].numpy(), metrics['Vdot'].numpy(), alpha=0.5)
    plt.xlabel('V(x)')
    plt.ylabel('Vdot(x)')
    plt.title('Lyapunov Function vs. Derivative')
    
    # Plot violation distribution
    plt.subplot(2, 2, 2)
    plt.hist(metrics['violations'].numpy(), bins=50)
    plt.xlabel('Violation Magnitude')
    plt.ylabel('Count')
    plt.title('Distribution of Lyapunov Violations')
    
    # Plot control inputs
    plt.subplot(2, 2, 3)
    for i in range(metrics['u_bc'].shape[1]):
        plt.hist(metrics['u_bc'][:, i].numpy(), bins=50, alpha=0.5, label=f'u_{i}')
    plt.xlabel('Control Input')
    plt.ylabel('Count')
    plt.title('Distribution of Control Inputs')
    plt.legend()
    
    # Plot violation rate vs level set
    plt.subplot(2, 2, 4)
    levels = [m['level'] for m in metrics['level_metrics']]
    rates = [m['violation_rate'] for m in metrics['level_metrics']]
    plt.plot(levels, rates, 'b-')
    plt.xlabel('V(x) Level')
    plt.ylabel('Violation Rate')
    plt.title('Violation Rate vs. Level Set')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_trajectories(
    system: SwingEquationSystem,
    bc_controller: BCController,
    clf_controller: CLFController,
    n_trajectories: int = 5,
    t_final: float = 10.0,
    save_path: str = 'trajectories.png',
) -> None:
    """Plot trajectories of the system under BC and CLF controllers.

    args:
        system: the swing equation system
        bc_controller: the BC controller
        clf_controller: the CLF controller
        n_trajectories: number of trajectories to plot
        t_final: final time for simulation
        save_path: path to save the plot
    """
    # Sample random initial states
    lower_lim, upper_lim = system.state_limits
    x0 = torch.rand(n_trajectories, system.n_dims) * (upper_lim - lower_lim) + lower_lim
    
    # Simulate trajectories
    t = torch.linspace(0, t_final, int(t_final / system.dt) + 1)
    x_bc = torch.zeros(len(t), n_trajectories, system.n_dims)
    x_clf = torch.zeros(len(t), n_trajectories, system.n_dims)
    V_bc = torch.zeros(len(t), n_trajectories)
    V_clf = torch.zeros(len(t), n_trajectories)
    
    x_bc[0] = x0
    x_clf[0] = x0
    V_bc[0] = clf_controller.V(x0)
    V_clf[0] = clf_controller.V(x0)
    
    for i in range(1, len(t)):
        # BC controller
        u_bc = bc_controller.u(x_bc[i-1])
        xdot_bc = system.closed_loop_dynamics(x_bc[i-1], u_bc)
        x_bc[i] = x_bc[i-1] + system.dt * xdot_bc
        V_bc[i] = clf_controller.V(x_bc[i])
        
        # CLF controller
        u_clf = clf_controller.u(x_clf[i-1])
        xdot_clf = system.closed_loop_dynamics(x_clf[i-1], u_clf)
        x_clf[i] = x_clf[i-1] + system.dt * xdot_clf
        V_clf[i] = clf_controller.V(x_clf[i])
    
    # Plot trajectories
    plt.figure(figsize=(15, 10))
    
    # Plot angles
    plt.subplot(3, 1, 1)
    for j in range(n_trajectories):
        plt.plot(t, x_bc[:, j, :system.N_NODES-1].numpy(), 'b-', alpha=0.3)
        plt.plot(t, x_clf[:, j, :system.N_NODES-1].numpy(), 'r--', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Angles')
    plt.title('Angle Trajectories')
    plt.legend(['BC Controller', 'CLF Controller'])
    
    # Plot frequencies
    plt.subplot(3, 1, 2)
    for j in range(n_trajectories):
        plt.plot(t, x_bc[:, j, system.N_NODES-1:].numpy(), 'b-', alpha=0.3)
        plt.plot(t, x_clf[:, j, system.N_NODES-1:].numpy(), 'r--', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequencies')
    plt.title('Frequency Trajectories')
    plt.legend(['BC Controller', 'CLF Controller'])
    
    # Plot Lyapunov function values
    plt.subplot(3, 1, 3)
    for j in range(n_trajectories):
        plt.plot(t, V_bc[:, j].numpy(), 'b-', alpha=0.3)
        plt.plot(t, V_clf[:, j].numpy(), 'r--', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('V(x)')
    plt.title('Lyapunov Function Values')
    plt.legend(['BC Controller', 'CLF Controller'])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Create system and scenarios
    system, scenarios = create_swing_system(n_nodes=3)
    
    # Create experiment suite
    experiment_suite = create_experiment_suite()
    
    # Create CLF controller (for Lyapunov function)
    clf_controller = CLFController(
        dynamics_model=system,
        scenarios=scenarios,
        experiment_suite=experiment_suite,
        clf_lambda=1.0,
        clf_relaxation_penalty=50.0,
        controller_period=0.01,
    )
    
    # Load BC controller
    bc_controller = load_bc_controller(system, scenarios, experiment_suite)
    
    # Verify stability
    violation_rate, avg_violation, metrics = verify_stability(
        system, bc_controller, clf_controller, n_samples=1000, n_levels=10
    )
    print(f"Lyapunov condition violation rate: {violation_rate:.2%}")
    print(f"Average violation magnitude: {avg_violation:.6f}")
    
    # Plot stability analysis
    plot_stability_analysis(metrics)
    print("Stability analysis saved to stability_analysis.png")
    
    # Plot trajectories
    plot_trajectories(system, bc_controller, clf_controller)
    print("Trajectories saved to trajectories.png")

if __name__ == "__main__":
    main() 