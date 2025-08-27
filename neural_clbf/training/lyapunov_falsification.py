import torch
import numpy as np
import time
from typing import Tuple, Dict, List, Optional, Callable
import matplotlib.pyplot as plt

try:
    from neural_clbf.systems import ControlAffineSystem
except Exception:
    from control_affine_system import ControlAffineSystem
try:
    from neural_clbf.controllers import NeuralCLBFController, BCController
except Exception:
    NeuralCLBFController = object
    BCController = object


class LyapunovFalsifier:
    """
    Implements a falsification-based approach to improve Lyapunov function learning.
    
    This class attempts to find counterexamples where the Lyapunov decrease condition
    is violated, and uses these counterexamples to improve the training process.
    """
    
    def __init__(
        self,
        system: ControlAffineSystem,
        lyapunov_fn: Callable,
        controller: NeuralCLBFController,
        falsification_methods: List[str] = ["gradient", "random", "adaptive"],
        n_samples: int = 1000,
        max_iterations: int = 10,
        learning_rate: float = 0.1,
        verbose: bool = True,
    ):
        """
        Initialize the Lyapunov falsifier.
        
        Args:
            system: Control-affine system
            lyapunov_fn: Callable function V(x) that returns Lyapunov values
            controller: Controller that uses the Lyapunov function
            falsification_methods: List of methods to use for falsification
                - "gradient": Gradient-based optimization to maximize violation
                - "random": Random sampling in state space
                - "adaptive": Adaptive sampling focusing on boundary regions
            n_samples: Number of samples to use for each falsification attempt
            max_iterations: Maximum number of iterations for gradient-based methods
            learning_rate: Learning rate for gradient-based optimization
            verbose: Whether to print progress information
        """
        self.system = system
        self.lyapunov_fn = lyapunov_fn
        self.controller = controller
        self.falsification_methods = falsification_methods
        self.n_samples = n_samples
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.verbose = verbose
        
        # Get system limits
        self.upper_limit, self.lower_limit = self.system.state_limits
        
        # Save clf_lambda for Lyapunov decrease condition
        if hasattr(controller, 'clf_lambda'):
            self.clf_lambda = controller.clf_lambda
        else:
            self.clf_lambda = 1.0  # Default value if not specified
    
    def compute_violation(
        self, 
        states: torch.Tensor, 
        controller: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the Lyapunov decrease condition violation for given states.
        
        Args:
            states: Batch of states [batch_size, n_dims]
            controller: Optional controller to use (defaults to self.controller)
            
        Returns:
            violations: Tensor of violation values for each state
            vdot_values: Tensor of Vdot values for each state
            v_values: Tensor of V values for each state
        """
        if controller is None:
            controller = self.controller
        
        # Compute V(x)
        states.requires_grad_(True)
        v_values = self.lyapunov_fn(states)
        
        # Get actions from controller
        with torch.no_grad():
            actions = controller.u(states)
        
        # Compute Lie derivatives
        Lf_V = torch.autograd.grad(v_values.sum(), states, create_graph=True)[0]
        
        # Compute system dynamics for each state
        f_x = torch.zeros((states.shape[0], states.shape[1], 1), device=states.device)
        g_x = torch.zeros((states.shape[0], states.shape[1], self.system.n_controls), device=states.device)
        
        for i in range(states.shape[0]):
            f, g = self.system.control_affine_dynamics(states[i:i+1])
            f_x[i] = f
            g_x[i] = g
        
        # Compute Vdot = ∇V·f + ∇V·g·u
        vdot_f = torch.bmm(Lf_V.unsqueeze(1), f_x).squeeze()
        vdot_g = torch.bmm(Lf_V.unsqueeze(1), 
                          torch.bmm(g_x, actions.unsqueeze(2))).squeeze()
        vdot_values = vdot_f + vdot_g
        
        # Lyapunov decrease condition: Vdot + λV ≤ 0
        # Violation is positive when condition is not satisfied
        violations = torch.relu(vdot_values + self.clf_lambda * v_values)
        
        return violations, vdot_values, v_values
    
    def falsify_gradient(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use gradient-based optimization to find states that violate the Lyapunov condition.
        
        Returns:
            counter_examples: States that violate the Lyapunov condition
            violations: Corresponding violation values
        """
        # Initialize states randomly
        states = torch.rand(self.n_samples, self.system.n_dims) * (self.upper_limit - self.lower_limit) + self.lower_limit
        states.requires_grad_(True)
        
        # Create optimizer for states
        optimizer = torch.optim.Adam([states], lr=self.learning_rate)
        
        # Gradient ascent to maximize violation
        best_violations = torch.zeros(self.n_samples)
        best_states = states.clone().detach()
        
        for i in range(self.max_iterations):
            # Compute violations
            violations, _, _ = self.compute_violation(states)
            mean_violation = violations.mean()
            
            # Update best states
            improved_mask = violations > best_violations
            best_violations[improved_mask] = violations[improved_mask].detach()
            best_states[improved_mask] = states[improved_mask].detach()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            (-mean_violation).backward()  # Maximize violation
            optimizer.step()
            
            # Project back to state limits
            with torch.no_grad():
                states.data = torch.clamp(states.data, self.lower_limit, self.upper_limit)
            
            if self.verbose and (i == 0 or (i+1) % (self.max_iterations // 5) == 0):
                print(f"Iteration {i+1}/{self.max_iterations}, Mean Violation: {mean_violation.item():.6f}")
        
        # Get top k violations
        k = min(100, self.n_samples)
        _, indices = torch.topk(best_violations, k)
        counter_examples = best_states[indices].detach()
        violations = best_violations[indices].detach()
        
        return counter_examples, violations
    
    def falsify_random(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use random sampling to find states that violate the Lyapunov condition.
        
        Returns:
            counter_examples: States that violate the Lyapunov condition
            violations: Corresponding violation values
        """
        # Generate random states
        states = torch.rand(self.n_samples, self.system.n_dims) * (self.upper_limit - self.lower_limit) + self.lower_limit
        
        # Compute violations
        with torch.no_grad():
            violations, _, _ = self.compute_violation(states)
        
        # Get states with non-zero violations
        violation_mask = violations > 0
        if violation_mask.sum() > 0:
            counter_examples = states[violation_mask]
            violation_values = violations[violation_mask]
        else:
            # If no violations found, return empty tensors
            counter_examples = torch.tensor([])
            violation_values = torch.tensor([])
            
        return counter_examples, violation_values
    
    def falsify_adaptive(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use adaptive sampling to find states that violate the Lyapunov condition.
        
        This method focuses on states near the boundary of the Lyapunov level sets.
        
        Returns:
            counter_examples: States that violate the Lyapunov condition
            violations: Corresponding violation values
        """
        # First, get a sense of the Lyapunov function's values across the state space
        with torch.no_grad():
            # Sample random states
            random_states = torch.rand(1000, self.system.n_dims) * (self.upper_limit - self.lower_limit) + self.lower_limit
            v_values = self.lyapunov_fn(random_states)
            
            # Compute percentiles to find boundaries
            v_median = torch.median(v_values)
            v_p75 = torch.quantile(v_values, 0.75)
            
            # Generate focused samples around level sets
            focused_states = []
            
            # Sample around the median level set
            base_states = random_states[torch.abs(v_values - v_median) < v_median * 0.2]
            if len(base_states) > 0:
                noise = torch.randn(min(500, len(base_states)), self.system.n_dims) * 0.1 * (self.upper_limit - self.lower_limit)
                focused_states.append(torch.clamp(base_states[:noise.shape[0]] + noise, self.lower_limit, self.upper_limit))
            
            # Sample around the 75th percentile level set
            base_states = random_states[torch.abs(v_values - v_p75) < v_p75 * 0.2]
            if len(base_states) > 0:
                noise = torch.randn(min(500, len(base_states)), self.system.n_dims) * 0.1 * (self.upper_limit - self.lower_limit)
                focused_states.append(torch.clamp(base_states[:noise.shape[0]] + noise, self.lower_limit, self.upper_limit))
            
            # Add pure random samples
            random_samples = torch.rand(self.n_samples - sum(len(s) for s in focused_states), self.system.n_dims) * (self.upper_limit - self.lower_limit) + self.lower_limit
            focused_states.append(random_samples)
            
            # Combine all samples
            states = torch.cat(focused_states)
            
            # Compute violations
            violations, _, _ = self.compute_violation(states)
        
        # Get states with non-zero violations
        violation_mask = violations > 0
        if violation_mask.sum() > 0:
            counter_examples = states[violation_mask]
            violation_values = violations[violation_mask]
        else:
            # If no violations found, return empty tensors
            counter_examples = torch.tensor([])
            violation_values = torch.tensor([])
            
        return counter_examples, violation_values
    
    def falsify(self) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Run falsification using all specified methods.
        
        Returns:
            counter_examples: Combined states that violate the Lyapunov condition
            violations: Corresponding violation values
            stats: Statistics about the falsification process
        """
        start_time = time.time()
        all_counter_examples = []
        all_violations = []
        stats = {method: {"n_examples": 0, "max_violation": 0.0} for method in self.falsification_methods}
        
        for method in self.falsification_methods:
            if self.verbose:
                print(f"\nRunning falsification method: {method}")
            
            if method == "gradient":
                examples, violations = self.falsify_gradient()
            elif method == "random":
                examples, violations = self.falsify_random()
            elif method == "adaptive":
                examples, violations = self.falsify_adaptive()
            else:
                raise ValueError(f"Unknown falsification method: {method}")
            
            if len(violations) > 0:
                all_counter_examples.append(examples)
                all_violations.append(violations)
                stats[method]["n_examples"] = len(violations)
                stats[method]["max_violation"] = float(violations.max().item())
                
                if self.verbose:
                    print(f"  Found {len(violations)} counterexamples")
                    print(f"  Max violation: {violations.max().item():.6f}")
            else:
                if self.verbose:
                    print(f"  No counterexamples found")
        
        # Combine all counterexamples
        if all_counter_examples:
            counter_examples = torch.cat(all_counter_examples)
            violations = torch.cat(all_violations)
        else:
            counter_examples = torch.tensor([])
            violations = torch.tensor([])
        
        # Add overall statistics
        stats["total"] = {
            "n_examples": len(counter_examples),
            "max_violation": float(violations.max().item()) if len(violations) > 0 else 0.0,
            "runtime": time.time() - start_time
        }
        
        if self.verbose:
            print(f"\nFalsification complete in {stats['total']['runtime']:.2f} seconds")
            print(f"Total counterexamples found: {stats['total']['n_examples']}")
        
        return counter_examples, violations, stats
    
    def visualization_2d(
        self, 
        counter_examples: torch.Tensor,
        violations: torch.Tensor,
        dims: Tuple[int, int] = (0, 1),
        title: str = "Lyapunov Violations",
        save_path: Optional[str] = None
    ):
        """
        Visualize Lyapunov violations in a 2D plot.
        
        Args:
            counter_examples: States that violate the Lyapunov condition
            violations: Corresponding violation values
            dims: Tuple of dimensions to plot
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        if counter_examples.shape[1] < 2:
            print("Cannot create 2D visualization for 1D system")
            return
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a meshgrid for the background
        x_dim, y_dim = dims
        x = np.linspace(self.lower_limit[x_dim], self.upper_limit[x_dim], 100)
        y = np.linspace(self.lower_limit[y_dim], self.upper_limit[y_dim], 100)
        X, Y = np.meshgrid(x, y)
        
        # Create states for the grid
        grid_shape = X.shape
        states = torch.zeros((grid_shape[0] * grid_shape[1], self.system.n_dims))
        
        # Use mean values for dimensions not being plotted
        mean_state = (self.upper_limit + self.lower_limit) / 2
        for i in range(self.system.n_dims):
            if i == x_dim:
                states[:, i] = torch.tensor(X.flatten())
            elif i == y_dim:
                states[:, i] = torch.tensor(Y.flatten())
            else:
                states[:, i] = mean_state[i]
        
        # Compute Lyapunov values for the grid
        with torch.no_grad():
            v_values = self.lyapunov_fn(states).reshape(grid_shape)
        
        # Plot Lyapunov level sets
        contour = ax.contour(X, Y, v_values.numpy(), levels=10, colors='k', alpha=0.3)
        plt.clabel(contour, inline=True, fontsize=8)
        
        # Add colorbar for the contour plot
        plt.colorbar(contour, ax=ax, label='Lyapunov Value V(x)')
        
        # Plot violations as a scatter plot
        if len(counter_examples) > 0:
            scatter = ax.scatter(
                counter_examples[:, x_dim].numpy(),
                counter_examples[:, y_dim].numpy(),
                c=violations.numpy(),
                cmap='plasma',
                s=50,
                alpha=0.7
            )
            plt.colorbar(scatter, ax=ax, label='Violation Value')
        
        # Set labels and title
        ax.set_xlabel(f'State Dimension {x_dim}')
        ax.set_ylabel(f'State Dimension {y_dim}')
        ax.set_title(title)
        
        # Save or show the plot
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    def visualization_3d(
        self, 
        counter_examples: torch.Tensor,
        violations: torch.Tensor,
        dims: Tuple[int, int, int] = (0, 1, 2),
        title: str = "Lyapunov Violations in 3D",
        save_path: Optional[str] = None
    ):
        """
        Visualize Lyapunov violations in a 3D plot.
        
        Args:
            counter_examples: States that violate the Lyapunov condition
            violations: Corresponding violation values
            dims: Tuple of dimensions to plot (x, y, z)
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        if counter_examples.shape[1] < 3:
            print("Cannot create 3D visualization for system with fewer than 3 dimensions")
            return
            
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot violations as a 3D scatter plot
        if len(counter_examples) > 0:
            x_dim, y_dim, z_dim = dims
            scatter = ax.scatter(
                counter_examples[:, x_dim].numpy(),
                counter_examples[:, y_dim].numpy(),
                counter_examples[:, z_dim].numpy(),
                c=violations.numpy(),
                cmap='plasma',
                s=50,
                alpha=0.7
            )
            fig.colorbar(scatter, ax=ax, label='Violation Value')
        
        # Set labels and title
        ax.set_xlabel(f'State Dimension {dims[0]}')
        ax.set_ylabel(f'State Dimension {dims[1]}')
        ax.set_zlabel(f'State Dimension {dims[2]}')
        ax.set_title(title)
        
        # Save or show the plot
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def generate_counterexample_dataset(
        self, 
        counter_examples: torch.Tensor,
        violations: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a dataset of counterexamples for retraining.
        
        Args:
            counter_examples: States that violate the Lyapunov condition
            violations: Corresponding violation values
            weights: Optional weights for each counterexample
            
        Returns:
            states: Tensor of states
            actions: Tensor of actions
            sample_weights: Tensor of sample weights for training
        """
        if len(counter_examples) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        # Get actions from the controller
        with torch.no_grad():
            actions = self.controller.u(counter_examples)
        
        # Generate sample weights based on violation values
        if weights is None:
            # Normalize violations to [0, 1] range
            if violations.max() > violations.min():
                normalized_violations = (violations - violations.min()) / (violations.max() - violations.min())
            else:
                normalized_violations = torch.ones_like(violations)
            
            # Use sqrt of normalized violations as weights (emphasizes worse violations)
            sample_weights = torch.sqrt(normalized_violations)
        else:
            sample_weights = weights
        
        return counter_examples, actions, sample_weights