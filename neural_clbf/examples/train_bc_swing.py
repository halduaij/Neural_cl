import torch
import numpy as np
from typing import List, Tuple
import torch.nn as nn

from neural_clbf.systems import SwingEquationSystem
from neural_clbf.systems.utils import Scenario, ScenarioList
from neural_clbf.controllers import CLFController, BCController
from neural_clbf.experiments import ExperimentSuite

def create_swing_system(n_nodes: int = 3) -> Tuple[SwingEquationSystem, ScenarioList]:
    """Create a swing equation system with n nodes.

    args:
        n_nodes: number of nodes in the system
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
    
    # Create system
    system = SwingEquationSystem(
        nominal_params=nominal_params,
        dt=0.01,
        controller_dt=0.01,
        scenarios=scenarios,
    )
    
    return system, scenarios

def create_experiment_suite() -> ExperimentSuite:
    """Create an experiment suite for evaluating the controller.

    returns:
        experiment_suite: the experiment suite
    """
    # Create experiment suite
    experiment_suite = ExperimentSuite([])
    
    return experiment_suite

def train_bc_controller(
    system: SwingEquationSystem,
    scenarios: ScenarioList,
    experiment_suite: ExperimentSuite,
    n_samples: int = 20000,  # Increased samples
    n_epochs: int = 200,     # Increased epochs
    hidden_layers: List[int] = [128, 128, 64],  # Deeper network
    learning_rate: float = 5e-4,  # Adjusted learning rate
    batch_size: int = 64,    # Increased batch size
    patience: int = 20,      # Early stopping patience
    lyapunov_weight: float = 0.1,  # Lyapunov regularization weight
) -> BCController:
    """Train a BC controller to mimic a CLF controller.

    args:
        system: the swing equation system
        scenarios: list of scenarios
        experiment_suite: the experiment suite
        n_samples: number of samples to collect
        n_epochs: number of training epochs
        hidden_layers: list of hidden layer sizes
        learning_rate: learning rate for the policy network
        batch_size: batch size for training
        patience: number of epochs to wait for improvement before early stopping
        lyapunov_weight: weight for the Lyapunov regularization term
    returns:
        bc_controller: the trained BC controller
    """
    # Create expert controller (CLF)
    expert_controller = CLFController(
        dynamics_model=system,
        scenarios=scenarios,
        experiment_suite=experiment_suite,
        clf_lambda=1.0,
        clf_relaxation_penalty=50.0,
        controller_period=0.01,
    )
    
    # Create BC controller
    bc_controller = BCController(
        dynamics_model=system,
        scenarios=scenarios,
        experiment_suite=experiment_suite,
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
        batch_size=batch_size,
        controller_period=0.01,
    )
    
    # Split data into train and validation
    n_train = int(0.8 * n_samples)
    n_val = n_samples - n_train
    
    # Train BC controller
    best_val_loss = float('inf')
    best_epoch = 0
    no_improve_count = 0
    
    for epoch in range(n_epochs):
        # Training
        train_losses = bc_controller.train(
            expert_controller=expert_controller,
            n_samples=n_train,
            n_epochs=1,  # Train for one epoch
            clf_aware=True,
            lyapunov_weight=lyapunov_weight,
        )
        
        # Validation
        val_states, val_actions = bc_controller.collect_expert_data(
            expert_controller=expert_controller,
            n_samples=n_val,
            clf_aware=True,
        )
        
        with torch.no_grad():
            val_pred = bc_controller.policy_network(val_states)
            val_loss = nn.MSELoss()(val_pred, val_actions)
            
            # Add Lyapunov validation if available
            if hasattr(expert_controller, 'V'):
                V = expert_controller.V(val_states)
                Lf_V, Lg_V = expert_controller.V_lie_derivatives(val_states)
                Vdot = Lf_V[:, 0, :] + torch.bmm(
                    Lg_V[:, 0, :].unsqueeze(1),
                    val_pred.reshape(-1, system.n_controls, 1),
                ).squeeze()
                lyapunov_loss = torch.relu(Vdot + expert_controller.clf_lambda * V).mean()
                val_loss = val_loss + lyapunov_weight * lyapunov_loss
        
        print(f"Epoch {epoch}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_count = 0
            # Save best model
            torch.save(bc_controller.state_dict(), "best_bc_controller.pt")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    bc_controller.load_state_dict(torch.load("best_bc_controller.pt"))
    print(f"Best model from epoch {best_epoch} with validation loss {best_val_loss:.6f}")
    
    return bc_controller

def main():
    # Create system and scenarios
    system, scenarios = create_swing_system(n_nodes=3)
    
    # Create experiment suite
    experiment_suite = create_experiment_suite()
    
    # Train BC controller
    bc_controller = train_bc_controller(
        system=system,
        scenarios=scenarios,
        experiment_suite=experiment_suite,
        n_samples=20000,
        n_epochs=200,
        hidden_layers=[128, 128, 64],
        learning_rate=5e-4,
        batch_size=64,
        patience=20,
        lyapunov_weight=0.1,
    )
    
    # Save the trained controller
    torch.save(bc_controller.state_dict(), "bc_controller.pt")
    print("Controller saved to bc_controller.pt")

if __name__ == "__main__":
    main() 