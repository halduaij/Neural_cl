import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, List, Dict
import numpy as np

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList
from neural_clbf.controllers.controller import Controller
from neural_clbf.experiments import ExperimentSuite

class BCController(Controller):
    """
    A behavior cloning controller that learns to mimic the actions of a CLF-based controller.
    """

    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        scenarios: ScenarioList,
        experiment_suite: ExperimentSuite,
        hidden_layers: List[int] = [64, 64],
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        controller_period: float = 0.01,
    ):
        """Initialize the BC controller.

        args:
            dynamics_model: the control-affine dynamics of the underlying system
            scenarios: a list of parameter scenarios to train on
            experiment_suite: defines the experiments to run during training
            hidden_layers: list of hidden layer sizes for the policy network
            learning_rate: learning rate for the policy network
            batch_size: batch size for training
            controller_period: the timestep to use in simulating forward
        """
        super(BCController, self).__init__(
            dynamics_model=dynamics_model,
            experiment_suite=experiment_suite,
            controller_period=controller_period,
        )

        # Save parameters
        self.scenarios = scenarios
        self.n_scenarios = len(scenarios)
        self.experiment_suite = experiment_suite
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Create the policy network
        self.policy_network = self._create_policy_network(hidden_layers)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def _create_policy_network(self, hidden_layers: List[int]) -> nn.Module:
        """Create the policy network.

        args:
            hidden_layers: list of hidden layer sizes
        returns:
            policy_network: the policy network
        """
        layers = []
        input_size = self.dynamics_model.n_dims
        
        # Add hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        # Add output layer
        layers.append(nn.Linear(input_size, self.dynamics_model.n_controls))
        
        return nn.Sequential(*layers)

    def collect_expert_data(
        self,
        expert_controller: Controller,
        n_samples: int = 10000,
        state_limits: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        clf_aware: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collect expert demonstrations from the expert controller.

        args:
            expert_controller: the expert controller to collect demonstrations from
            n_samples: number of samples to collect
            state_limits: optional tuple of (lower, upper) state limits for sampling
            clf_aware: if True, sample more points near the boundary of the CLF level sets
        returns:
            states: tensor of states
            actions: tensor of expert actions
        """
        if state_limits is None:
            state_limits = self.dynamics_model.state_limits
        
        lower_lim, upper_lim = state_limits
        
        if clf_aware and hasattr(expert_controller, 'V'):
            # Sample more points near CLF level sets
            states = []
            actions = []
            
            # Regular random sampling (70% of samples)
            n_regular = int(0.7 * n_samples)
            states_regular = torch.rand(n_regular, self.dynamics_model.n_dims) * (upper_lim - lower_lim) + lower_lim
            with torch.no_grad():
                actions_regular = expert_controller.u(states_regular)
            states.append(states_regular)
            actions.append(actions_regular)
            
            # CLF-aware sampling (30% of samples)
            n_clf = n_samples - n_regular
            states_clf = torch.rand(n_clf, self.dynamics_model.n_dims) * (upper_lim - lower_lim) + lower_lim
            with torch.no_grad():
                V = expert_controller.V(states_clf)
                # Perturb states to be near level sets
                grad_V = torch.autograd.grad(V.sum(), states_clf)[0]
                states_clf = states_clf + 0.1 * torch.randn_like(states_clf) * grad_V
                # Clip to state limits
                states_clf = torch.clamp(states_clf, lower_lim, upper_lim)
                actions_clf = expert_controller.u(states_clf)
            states.append(states_clf)
            actions.append(actions_clf)
            
            states = torch.cat(states, dim=0)
            actions = torch.cat(actions, dim=0)
        else:
            # Regular random sampling
            states = torch.rand(n_samples, self.dynamics_model.n_dims) * (upper_lim - lower_lim) + lower_lim
            with torch.no_grad():
                actions = expert_controller.u(states)
        
        return states, actions

    def train(
        self,
        expert_controller: Controller,
        n_samples: int = 10000,
        n_epochs: int = 100,
        state_limits: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        clf_aware: bool = True,
        lyapunov_weight: float = 0.1,
    ) -> List[float]:
        """Train the BC policy to mimic the expert controller.

        args:
            expert_controller: the expert controller to mimic
            n_samples: number of samples to collect
            n_epochs: number of training epochs
            state_limits: optional tuple of (lower, upper) state limits for sampling
            clf_aware: if True, use CLF-aware sampling
            lyapunov_weight: weight for the Lyapunov regularization term
        returns:
            losses: list of training losses
        """
        # Collect expert data
        states, expert_actions = self.collect_expert_data(
            expert_controller, n_samples, state_limits, clf_aware
        )
        
        # Training loop
        losses = []
        for epoch in range(n_epochs):
            # Shuffle data
            indices = torch.randperm(n_samples)
            states_shuffled = states[indices]
            expert_actions_shuffled = expert_actions[indices]
            
            # Train in batches
            epoch_loss = 0
            for i in range(0, n_samples, self.batch_size):
                batch_states = states_shuffled[i:i + self.batch_size]
                batch_expert_actions = expert_actions_shuffled[i:i + self.batch_size]
                
                # Forward pass
                predicted_actions = self.policy_network(batch_states)
                
                # Compute MSE loss
                mse_loss = nn.MSELoss()(predicted_actions, batch_expert_actions)
                
                # Add Lyapunov regularization if available
                if clf_aware and hasattr(expert_controller, 'V'):
                    V = expert_controller.V(batch_states)
                    Lf_V, Lg_V = expert_controller.V_lie_derivatives(batch_states)
                    
                    # Compute Vdot for predicted actions
                    Vdot = Lf_V[:, 0, :] + torch.bmm(
                        Lg_V[:, 0, :].unsqueeze(1),
                        predicted_actions.reshape(-1, self.dynamics_model.n_controls, 1),
                    ).squeeze()
                    
                    # Lyapunov condition: Vdot + lambda*V <= 0
                    lyapunov_loss = torch.relu(Vdot + expert_controller.clf_lambda * V).mean()
                    
                    # Total loss
                    loss = mse_loss + lyapunov_weight * lyapunov_loss
                else:
                    loss = mse_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Average loss for the epoch
            epoch_loss /= (n_samples // self.batch_size)
            losses.append(epoch_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.6f}")
        
        return losses

    def u(self, x: torch.Tensor) -> torch.Tensor:
        """Get the control input for a given state.

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
        """
        with torch.no_grad():
            return self.policy_network(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the policy network.

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
        """
        return self.u(x) 