import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Dict, List

def plot_trajectories(results: Dict[str, List[Dict]], save_dir: str, n_plots: int = 5):
    """Plot trajectories from comparison results.
    
    args:
        results: dictionary of trajectory results for each controller
        save_dir: directory to save plots
        n_plots: number of trajectories to plot
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get controller names
    controller_names = list(results.keys())
    
    # Number of trajectories to plot
    n_plots = min(n_plots, len(results[controller_names[0]]))
    
    # Plot state trajectories
    for i in range(n_plots):
        plt.figure(figsize=(15, 10))
        
        # Extract data for this trajectory
        trajectory_data = {}
        n_states = None
        for controller in controller_names:
            traj = results[controller][i]
            trajectory_data[controller] = {
                "x": traj["x"].detach().cpu().numpy(),
                "u": traj["u"].detach().cpu().numpy(),
                "t": traj["t"].detach().cpu().numpy(),
            }
            n_states = traj["x"].shape[2]
        
        # Create state plots
        for j in range(n_states):
            plt.subplot(n_states, 1, j+1)
            for controller in controller_names:
                data = trajectory_data[controller]
                plt.plot(data["t"], data["x"][:, 0, j], label=f"{controller}")
            plt.ylabel(f"State {j+1}")
            plt.grid(True)
            if j == 0:
                plt.title(f"Trajectory {i+1} - State Evolution")
            if j == n_states - 1:
                plt.xlabel("Time (s)")
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"trajectory_{i+1}_states.png"))
        plt.close()
        
        # Create control input plots
        n_controls = trajectory_data[controller_names[0]]["u"].shape[2]
        plt.figure(figsize=(15, 10))
        for j in range(n_controls):
            plt.subplot(n_controls, 1, j+1)
            for controller in controller_names:
                data = trajectory_data[controller]
                plt.plot(data["t"][:-1], data["u"][:, 0, j], label=f"{controller}")
            plt.ylabel(f"Control {j+1}")
            plt.grid(True)
            if j == 0:
                plt.title(f"Trajectory {i+1} - Control Inputs")
            if j == n_controls - 1:
                plt.xlabel("Time (s)")
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"trajectory_{i+1}_controls.png"))
        plt.close()

def plot_statistics(results: Dict[str, List[Dict]], save_dir: str):
    """Plot statistics from comparison results.
    
    args:
        results: dictionary of trajectory results for each controller
        save_dir: directory to save plots
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get controller names
    controller_names = list(results.keys())
    
    # Compute statistics
    stats = {}
    for controller in controller_names:
        trajectories = results[controller]
        n_trajectories = len(trajectories)
        
        # Extract data
        final_errors = []
        control_efforts = []
        for traj in trajectories:
            # Final state error (distance from goal)
            final_state = traj["x"][-1]
            goal_state = torch.zeros_like(final_state)
            final_error = torch.norm(final_state - goal_state).item()
            final_errors.append(final_error)
            
            # Control effort (sum of control signal magnitudes)
            control_signals = traj["u"]
            control_effort = torch.sum(torch.norm(control_signals, dim=2)).item()
            control_efforts.append(control_effort)
        
        stats[controller] = {
            "final_errors": final_errors,
            "control_efforts": control_efforts,
        }
    
    # Plot final errors
    plt.figure(figsize=(10, 6))
    for i, controller in enumerate(controller_names):
        errors = stats[controller]["final_errors"]
        plt.boxplot(errors, positions=[i+1], labels=[controller.upper()])
    plt.ylabel("Final Error")
    plt.title("Comparison of Final Errors")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "final_errors.png"))
    plt.close()
    
    # Plot control efforts
    plt.figure(figsize=(10, 6))
    for i, controller in enumerate(controller_names):
        efforts = stats[controller]["control_efforts"]
        plt.boxplot(efforts, positions=[i+1], labels=[controller.upper()])
    plt.ylabel("Control Effort")
    plt.title("Comparison of Control Efforts")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "control_efforts.png"))
    plt.close()

def plot_phase_portraits(results: Dict[str, List[Dict]], save_dir: str, n_plots: int = 5):
    """Plot phase portraits from comparison results.
    
    args:
        results: dictionary of trajectory results for each controller
        save_dir: directory to save plots
        n_plots: number of trajectories to plot
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get controller names
    controller_names = list(results.keys())
    
    # Number of trajectories to plot
    n_plots = min(n_plots, len(results[controller_names[0]]))
    
    # For swing equation, we're primarily interested in:
    # - Phase portrait of omega variables
    # - Phase portrait of theta variables
    for i in range(n_plots):
        plt.figure(figsize=(15, 7))
        
        # Extract data for this trajectory
        trajectory_data = {}
        n_states = None
        for controller in controller_names:
            traj = results[controller][i]
            trajectory_data[controller] = {
                "x": traj["x"].detach().cpu().numpy(),
            }
            n_states = traj["x"].shape[2]
        
        # Get omega and theta indices
        # Assuming state is [theta_12, ..., theta_1n, omega_1, ..., omega_n]
        n_nodes = (n_states + 1) // 2
        theta_indices = list(range(n_nodes - 1))
        omega_indices = list(range(n_nodes - 1, n_states))
        
        # Plot omega phase portrait (first two omegas)
        if len(omega_indices) >= 2:
            plt.subplot(1, 2, 1)
            for controller in controller_names:
                data = trajectory_data[controller]
                plt.plot(
                    data["x"][:, 0, omega_indices[0]], 
                    data["x"][:, 0, omega_indices[1]], 
                    label=f"{controller}",
                    marker='.'
                )
                # Mark start and end points
                plt.plot(
                    data["x"][0, 0, omega_indices[0]], 
                    data["x"][0, 0, omega_indices[1]], 
                    'go', 
                    markersize=10, 
                    label=f"{controller} start" if controller == controller_names[0] else ""
                )
                plt.plot(
                    data["x"][-1, 0, omega_indices[0]], 
                    data["x"][-1, 0, omega_indices[1]], 
                    'ro', 
                    markersize=10, 
                    label=f"{controller} end" if controller == controller_names[0] else ""
                )
            plt.xlabel(f"Omega 1")
            plt.ylabel(f"Omega 2")
            plt.title("Phase Portrait - Angular Frequencies")
            plt.grid(True)
            plt.legend()
        
        # Plot theta phase portrait (first two thetas)
        if len(theta_indices) >= 2:
            plt.subplot(1, 2, 2)
            for controller in controller_names:
                data = trajectory_data[controller]
                plt.plot(
                    data["x"][:, 0, theta_indices[0]], 
                    data["x"][:, 0, theta_indices[1]], 
                    label=f"{controller}",
                    marker='.'
                )
                # Mark start and end points
                plt.plot(
                    data["x"][0, 0, theta_indices[0]], 
                    data["x"][0, 0, theta_indices[1]], 
                    'go', 
                    markersize=10, 
                    label=f"{controller} start" if controller == controller_names[0] else ""
                )
                plt.plot(
                    data["x"][-1, 0, theta_indices[0]], 
                    data["x"][-1, 0, theta_indices[1]], 
                    'ro', 
                    markersize=10, 
                    label=f"{controller} end" if controller == controller_names[0] else ""
                )
            plt.xlabel(f"Theta 12")
            plt.ylabel(f"Theta 13")
            plt.title("Phase Portrait - Angles")
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"phase_portrait_{i+1}.png"))
        plt.close()

def plot_lyapunov_function(results: Dict[str, List[Dict]], V_fn, save_dir: str, n_plots: int = 5):
    """Plot Lyapunov function values along trajectories.
    
    args:
        results: dictionary of trajectory results for each controller
        V_fn: Lyapunov function
        save_dir: directory to save plots
        n_plots: number of trajectories to plot
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get controller names
    controller_names = list(results.keys())
    
    # Number of trajectories to plot
    n_plots = min(n_plots, len(results[controller_names[0]]))
    
    # Plot Lyapunov function values
    for i in range(n_plots):
        plt.figure(figsize=(10, 6))
        
        # Extract data for this trajectory
        for controller in controller_names:
            traj = results[controller][i]
            states = traj["x"]
            times = traj["t"]
            
            # Compute Lyapunov function values
            V_values = []
            for state in states:
                V = V_fn(state).detach().cpu().numpy()
                V_values.append(V)
            
            # Plot
            plt.plot(times.detach().cpu().numpy(), V_values, label=f"{controller}")
        
        plt.xlabel("Time (s)")
        plt.ylabel("Lyapunov Function Value")
        plt.title(f"Trajectory {i+1} - Lyapunov Function Evolution")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"lyapunov_{i+1}.png"))
        plt.close()

def main(args):
    # Load comparison results
    results_path = os.path.join(args.bc_logdir, "comparison_results.pt")
    if not os.path.exists(results_path):
        print(f"Error: Comparison results not found at {results_path}")
        print("Please run train_swing_clbf_bc.py with --evaluate flag first")
        return
    
    results = torch.load(results_path)
    print(f"Loaded comparison results from {results_path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot trajectories
    plot_trajectories(results, os.path.join(args.output_dir, "trajectories"), args.n_plots)
    print("Plotted trajectories")
    
    # Plot statistics
    plot_statistics(results, args.output_dir)
    print("Plotted statistics")
    
    # Plot phase portraits
    plot_phase_portraits(results, os.path.join(args.output_dir, "phase_portraits"), args.n_plots)
    print("Plotted phase portraits")
    
    # Plot Lyapunov function (if available)
    if args.plot_lyapunov:
        # Load CLBF controller to get Lyapunov function
        from neural_clbf.controllers import NeuralCLBFController
        from neural_clbf.systems import SwingEquationSystem
        
        # Find the best checkpoint
        checkpoints_dir = os.path.join(args.clbf_logdir, "checkpoints")
        checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith(".ckpt")]
        if checkpoints:
            best_checkpoint = sorted(checkpoints)[-1]  # Just take the latest for now
            checkpoint_path = os.path.join(checkpoints_dir, best_checkpoint)
            print(f"Loading CLBF controller from {checkpoint_path} for Lyapunov function visualization")
            clbf_controller = NeuralCLBFController.load_from_checkpoint(checkpoint_path)
            
            # Create Lyapunov function
            def V_fn(x):
                with torch.no_grad():
                    return clbf_controller.V(x)
            
            # Plot Lyapunov function
            plot_lyapunov_function(results, V_fn, os.path.join(args.output_dir, "lyapunov"), args.n_plots)
            print("Plotted Lyapunov function values")
        else:
            print("No CLBF checkpoints found. Skipping Lyapunov function visualization.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and compare CLBF and BC controllers for Swing Equation System")
    
    # Input parameters
    parser.add_argument("--clbf_logdir", type=str, default="logs/swing_equation_clbf", help="Directory with CLBF logs")
    parser.add_argument("--bc_logdir", type=str, default="logs/swing_equation_bc", help="Directory with BC logs")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Directory to save visualizations")
    parser.add_argument("--n_plots", type=int, default=5, help="Number of trajectories to plot")
    parser.add_argument("--plot_lyapunov", action="store_true", help="Plot Lyapunov function values")
    
    args = parser.parse_args()
    main(args)