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

def create_swing_system(n_nodes: int = 3, max_rocof: float = 1.0) -> Tuple[SwingEquationSystem, ScenarioList]:
    """Create a swing equation system with n nodes.

    args:
        n_nodes: number of nodes in the system
        max_rocof: maximum allowed Rate of Change of Frequency (Hz/s)
    returns:
        system: the swing equation system
        scenarios: list of scenarios
    """
    # Create nominal parameters
    nominal_params = {
        "M": torch.ones(n_nodes),  # inertia constants
        "D": torch.ones(n_nodes),  # damping coefficients
        "P": torch.zeros(n_nodes),  # mechanical power inputs
        "K": torch.ones((n_nodes, n_nodes)),  # coupling strength matrix
    }
    
    # Create scenarios
    scenarios = [Scenario(nominal_params)]
    
    # Create system with RoCoF constraints
    system = SwingEquationSystem(
        nominal_params=nominal_params,
        dt=0.01,
        controller_dt=0.01,
        scenarios=scenarios,
        max_rocof=max_rocof
    )
    
    return system, scenarios

def create_experiment_suite() -> ExperimentSuite:
    """Create an experiment suite for evaluation.

    returns:
        experiment_suite: the experiment suite
    """
    experiment_suite = ExperimentSuite()
    return experiment_suite

def verify_stability(
    system: SwingEquationSystem,
    bc_controller: BCController,
    clf_controller: CLFController,
    n_samples: int = 1000,
    n_levels: int = 10,
) -> Tuple[float, float, Dict]:
    """Verify the stability of the BC controller.

    args:
        system: the swing equation system
        bc_controller: the BC controller
        clf_controller: the CLF controller (for Lyapunov function)
        n_samples: number of samples to check
        n_levels: number of Lyapunov level sets to check
    returns:
        violation_rate: percentage of samples with Lyapunov violations
        avg_violation: average magnitude of Lyapunov violations
        metrics: dictionary with detailed metrics
    """
    # Sample states randomly
    lower_lim, upper_lim = system.state_limits
    states = torch.rand(n_samples, system.n_dims) * (upper_lim - lower_lim) + lower_lim
    
    # Get Lyapunov values and find range
    V = clf_controller.V(states)
    v_min, v_max = V.min().item(), V.max().item()
    
    # Define level sets
    level_values = torch.linspace(v_min, v_max, n_levels)
    metrics = {
        "level_values": level_values.numpy(),
        "violation_rates": [],
        "violation_magnitudes": []
    }
    
    # Check violations at each level set
    for level in level_values:
        # Find states close to this level set
        mask = (V <= level * 1.1) & (V >= level * 0.9)
        if not mask.any():
            metrics["violation_rates"].append(0.0)
            metrics["violation_magnitudes"].append(0.0)
            continue
            
        level_states = states[mask]
        if len(level_states) == 0:
            metrics["violation_rates"].append(0.0)
            metrics["violation_magnitudes"].append(0.0)
            continue
            
        # Get BC controller actions
        bc_actions = bc_controller.u(level_states)
        
        # Compute Lyapunov derivative
        level_states.requires_grad_(True)
        V_level = clf_controller.V(level_states)
        grad_V = torch.autograd.grad(V_level.sum(), level_states, create_graph=True)[0]
        
        # Compute dynamics
        dynamics = system.closed_loop_dynamics(level_states, bc_actions)
        
        # Compute Vdot = ∇V·ẋ
        Vdot = torch.sum(grad_V * dynamics, dim=1)
        
        # Compute violations: Vdot + λV > 0
        violations = (Vdot + clf_controller.clf_lambda * V_level) > 0
        violation_rate = violations.float().mean().item()
        
        # Compute magnitudes of violations
        if violations.any():
            violation_magnitudes = (Vdot + clf_controller.clf_lambda * V_level)[violations]
            avg_magnitude = violation_magnitudes.mean().item()
        else:
            avg_magnitude = 0.0
            
        metrics["violation_rates"].append(violation_rate)
        metrics["violation_magnitudes"].append(avg_magnitude)
    
    # Compute overall violation rate and magnitude
    overall_violations = torch.zeros(n_samples, dtype=torch.bool)
    overall_magnitudes = torch.zeros(n_samples)
    
    states.requires_grad_(True)
    V = clf_controller.V(states)
    grad_V = torch.autograd.grad(V.sum(), states, create_graph=True)[0]
    
    bc_actions = bc_controller.u(states)
    dynamics = system.closed_loop_dynamics(states, bc_actions)
    
    Vdot = torch.sum(grad_V * dynamics, dim=1)
    violations = (Vdot + clf_controller.clf_lambda * V) > 0
    violation_rate = violations.float().mean().item()
    
    if violations.any():
        violation_magnitudes = (Vdot + clf_controller.clf_lambda * V)[violations]
        avg_violation = violation_magnitudes.mean().item()
    else:
        avg_violation = 0.0
    
    return violation_rate, avg_violation, metrics

def plot_stability_analysis(metrics: Dict, save_path: str = 'stability_analysis.png') -> None:
    """Plot stability analysis results.

    args:
        metrics: dictionary with stability metrics
        save_path: path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(metrics["level_values"], metrics["violation_rates"], 'b-o')
    plt.xlabel('Lyapunov Level Set Value')
    plt.ylabel('Violation Rate')
    plt.title('Lyapunov Condition Violation Rate by Level Set')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(metrics["level_values"], metrics["violation_magnitudes"], 'r-o')
    plt.xlabel('Lyapunov Level Set Value')
    plt.ylabel('Average Violation Magnitude')
    plt.title('Lyapunov Condition Violation Magnitude by Level Set')
    plt.grid(True)
    
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
    u_bc = torch.zeros(len(t), n_trajectories, system.n_controls)
    u_clf = torch.zeros(len(t), n_trajectories, system.n_controls)
    V_bc = torch.zeros(len(t), n_trajectories)
    V_clf = torch.zeros(len(t), n_trajectories)
    
    # Calculate RoCoF for each step
    rocof_bc = torch.zeros(len(t), n_trajectories, system.N_NODES)
    rocof_clf = torch.zeros(len(t), n_trajectories, system.N_NODES)
    
    x_bc[0] = x0
    x_clf[0] = x0
    V_bc[0] = clf_controller.V(x0)
    V_clf[0] = clf_controller.V(x0)
    
    # Initial control inputs
    u_bc[0] = bc_controller.u(x0)
    u_clf[0] = clf_controller.u(x0)
    
    # Calculate initial RoCoF
    rocof_bc[0] = system.compute_rocof(x0, u_bc[0])
    rocof_clf[0] = system.compute_rocof(x0, u_clf[0])
    
    for i in range(1, len(t)):
        # BC controller
        u_bc[i] = bc_controller.u(x_bc[i-1])
        xdot_bc = system.closed_loop_dynamics(x_bc[i-1], u_bc[i])
        x_bc[i] = x_bc[i-1] + system.dt * xdot_bc
        V_bc[i] = clf_controller.V(x_bc[i])
        rocof_bc[i] = system.compute_rocof(x_bc[i], u_bc[i])
        
        # CLF controller
        u_clf[i] = clf_controller.u(x_clf[i-1])
        xdot_clf = system.closed_loop_dynamics(x_clf[i-1], u_clf[i])
        x_clf[i] = x_clf[i-1] + system.dt * xdot_clf
        V_clf[i] = clf_controller.V(x_clf[i])
        rocof_clf[i] = system.compute_rocof(x_clf[i], u_clf[i])
    
    # Plot trajectories
    plt.figure(figsize=(15, 15))
    
    # Plot angles
    plt.subplot(4, 1, 1)
    for j in range(n_trajectories):
        plt.plot(t, x_bc[:, j, :system.N_NODES-1].numpy(), 'b-', alpha=0.3)
        plt.plot(t, x_clf[:, j, :system.N_NODES-1].numpy(), 'r--', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Angles')
    plt.title('Angle Trajectories')
    plt.legend(['BC Controller', 'CLF Controller'])
    
    # Plot frequencies
    plt.subplot(4, 1, 2)
    for j in range(n_trajectories):
        plt.plot(t, x_bc[:, j, system.N_NODES-1:].numpy(), 'b-', alpha=0.3)
        plt.plot(t, x_clf[:, j, system.N_NODES-1:].numpy(), 'r--', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequencies')
    plt.title('Frequency Trajectories')
    plt.legend(['BC Controller', 'CLF Controller'])
    
    # Plot Lyapunov function values
    plt.subplot(4, 1, 3)
    for j in range(n_trajectories):
        plt.plot(t, V_bc[:, j].numpy(), 'b-', alpha=0.3)
        plt.plot(t, V_clf[:, j].numpy(), 'r--', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('V(x)')
    plt.title('Lyapunov Function Values')
    plt.legend(['BC Controller', 'CLF Controller'])
    
    # Plot RoCoF values with safety limits
    plt.subplot(4, 1, 4)
    max_rocof = system.max_rocof
    for j in range(n_trajectories):
        for k in range(system.N_NODES):
            plt.plot(t, rocof_bc[:, j, k].numpy(), 'b-', alpha=0.3)
            plt.plot(t, rocof_clf[:, j, k].numpy(), 'r--', alpha=0.3)
    # Add RoCoF safety limits
    plt.axhline(y=max_rocof, color='g', linestyle='--', label='RoCoF Limit')
    plt.axhline(y=-max_rocof, color='g', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('RoCoF (Hz/s)')
    plt.title('Rate of Change of Frequency (RoCoF)')
    plt.legend(['BC Controller', 'CLF Controller', 'Safety Limit'])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_rocof(
    system: SwingEquationSystem,
    bc_controller: BCController,
    clf_controller: CLFController,
    n_samples: int = 1000,
    save_path: str = 'rocof_analysis.png'
) -> Dict:
    """Analyze RoCoF violations for the controllers.

    args:
        system: the swing equation system
        bc_controller: the BC controller
        clf_controller: the CLF controller
        n_samples: number of samples to check
        save_path: path to save the plot
    returns:
        results: dictionary with RoCoF analysis results
    """
    # Sample states randomly
    lower_lim, upper_lim = system.state_limits
    states = torch.rand(n_samples, system.n_dims) * (upper_lim - lower_lim) + lower_lim
    
    # Get control inputs
    bc_actions = bc_controller.u(states)
    clf_actions = clf_controller.u(states)
    
    # Compute RoCoF for both controllers
    rocof_bc = system.compute_rocof(states, bc_actions)
    rocof_clf = system.compute_rocof(states, clf_actions)
    
    # Check for violations
    bc_violations = torch.abs(rocof_bc) > system.max_rocof
    clf_violations = torch.abs(rocof_clf) > system.max_rocof
    
    # Compute violation rates (per node and overall)
    bc_node_violation_rates = bc_violations.float().mean(dim=0)
    clf_node_violation_rates = clf_violations.float().mean(dim=0)
    
    bc_overall_violation_rate = bc_violations.any(dim=1).float().mean()
    clf_overall_violation_rate = clf_violations.any(dim=1).float().mean()
    
    # Compute max RoCoF values
    bc_max_rocof = torch.abs(rocof_bc).max(dim=0)[0]
    clf_max_rocof = torch.abs(rocof_clf).max(dim=0)[0]
    
    # Prepare results
    results = {
        "bc_node_violation_rates": bc_node_violation_rates.tolist(),
        "clf_node_violation_rates": clf_node_violation_rates.tolist(),
        "bc_overall_violation_rate": bc_overall_violation_rate.item(),
        "clf_overall_violation_rate": clf_overall_violation_rate.item(),
        "bc_max_rocof": bc_max_rocof.tolist(),
        "clf_max_rocof": clf_max_rocof.tolist(),
        "max_rocof_limit": system.max_rocof
    }
    
    # Plot RoCoF analysis
    plt.figure(figsize=(12, 10))
    
    # Plot violation rates by node
    plt.subplot(2, 2, 1)
    nodes = range(1, system.N_NODES + 1)
    plt.bar([n - 0.15 for n in nodes], bc_node_violation_rates.tolist(), width=0.3, color='blue', label='BC')
    plt.bar([n + 0.15 for n in nodes], clf_node_violation_rates.tolist(), width=0.3, color='red', label='CLF')
    plt.xlabel('Node')
    plt.ylabel('Violation Rate')
    plt.title('RoCoF Violation Rate by Node')
    plt.xticks(nodes)
    plt.ylim(0, 1)
    plt.legend()
    
    # Plot max RoCoF by node
    plt.subplot(2, 2, 2)
    plt.bar([n - 0.15 for n in nodes], bc_max_rocof.tolist(), width=0.3, color='blue', label='BC')
    plt.bar([n + 0.15 for n in nodes], clf_max_rocof.tolist(), width=0.3, color='red', label='CLF')
    plt.axhline(y=system.max_rocof, color='g', linestyle='--', label='Limit')
    plt.xlabel('Node')
    plt.ylabel('Max RoCoF (Hz/s)')
    plt.title('Maximum RoCoF by Node')
    plt.xticks(nodes)
    plt.legend()
    
    # Plot RoCoF distribution for BC controller
    plt.subplot(2, 2, 3)
    rocof_bc_flat = rocof_bc.flatten().numpy()
    plt.hist(rocof_bc_flat, bins=50, alpha=0.7, color='blue')
    plt.axvline(x=system.max_rocof, color='g', linestyle='--', label='Limit')
    plt.axvline(x=-system.max_rocof, color='g', linestyle='--')
    plt.xlabel('RoCoF (Hz/s)')
    plt.ylabel('Frequency')
    plt.title('BC Controller RoCoF Distribution')
    plt.legend()
    
    # Plot RoCoF distribution for CLF controller
    plt.subplot(2, 2, 4)
    rocof_clf_flat = rocof_clf.flatten().numpy()
    plt.hist(rocof_clf_flat, bins=50, alpha=0.7, color='red')
    plt.axvline(x=system.max_rocof, color='g', linestyle='--', label='Limit')
    plt.axvline(x=-system.max_rocof, color='g', linestyle='--')
    plt.xlabel('RoCoF (Hz/s)')
    plt.ylabel('Frequency')
    plt.title('CLF Controller RoCoF Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # Print summary
    print("\nRoCoF Analysis Results:")
    print(f"BC Controller overall violation rate: {bc_overall_violation_rate.item():.2%}")
    print(f"CLF Controller overall violation rate: {clf_overall_violation_rate.item():.2%}")
    print(f"Maximum BC Controller RoCoF: {torch.abs(rocof_bc).max().item():.4f} Hz/s")
    print(f"Maximum CLF Controller RoCoF: {torch.abs(rocof_clf).max().item():.4f} Hz/s")
    print(f"RoCoF safety limit: {system.max_rocof:.4f} Hz/s")
    
    return results

def main(max_rocof=1.0):
    # Create system and scenarios
    system, scenarios = create_swing_system(n_nodes=3, max_rocof=max_rocof)
    print(f"Created swing equation system with 3 nodes")
    print(f"RoCoF safety limit: {max_rocof} Hz/s")
    
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
    
    # Plot trajectories with RoCoF analysis
    plot_trajectories(system, bc_controller, clf_controller)
    print("Trajectories saved to trajectories.png")
    
    # Analyze RoCoF violations
    rocof_results = analyze_rocof(system, bc_controller, clf_controller)
    print("RoCoF analysis saved to rocof_analysis.png")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate BC controller for swing equation system")
    parser.add_argument("--max_rocof", type=float, default=1.0, help="Maximum allowed Rate of Change of Frequency (Hz/s)")
    
    args = parser.parse_args()
    main(max_rocof=args.max_rocof) 