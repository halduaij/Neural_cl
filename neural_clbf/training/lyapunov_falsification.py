import torch
import numpy as np
import time
from typing import Tuple, Dict, List, Optional, Callable
import matplotlib.pyplot as plt

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.controllers import NeuralCLBFController

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
        learning_rate: float = 1e-3, # Adjusted for high-dimensional systems
        verbose: bool = True,
    ):
        """
        Initialize the Lyapunov falsifier.
        
        Args:
            system: Control-affine system
            lyapunov_fn: Callable function V(x) that returns Lyapunov values
            controller: Controller that uses the Lyapunov function
            falsification_methods: List of methods to use for falsification
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
        
        self.upper_limit, self.lower_limit = self.system.state_limits
        
        if hasattr(controller, 'clf_lambda'):
            self.clf_lambda = controller.clf_lambda
        else:
            self.clf_lambda = 1.0
    
    def compute_violation(
        self, 
        states: torch.Tensor, 
        controller: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the Lyapunov decrease condition violation for a batch of states.
        
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
        
        states.requires_grad_(True)
        v_values = self.lyapunov_fn(states)
        
        with torch.no_grad():
            actions = controller.u(states)
        
        # **PERFORMANCE OPTIMIZATION**: Use the controller's vectorized
        # V_lie_derivatives method instead of re-computing gradients and dynamics
        # in a loop. This is significantly more efficient.
        Lf_V, Lg_V = controller.V_lie_derivatives(states)
        
        # We need to find the worst-case Vdot over all scenarios
        batch_size = states.shape[0]
        worst_vdot = torch.full((batch_size,), -float('inf'), device=states.device)

        for i in range(controller.n_scenarios):
            # Vdot = Lf_V + Lg_V * u
            vdot_scenario = Lf_V[:, i, :].squeeze(-1) + torch.bmm(
                Lg_V[:, i, :].unsqueeze(1), actions.unsqueeze(2)
            ).squeeze()
            
            worst_vdot = torch.max(worst_vdot, vdot_scenario)
        
        # Violation is positive when the decrease condition is not satisfied
        violations = torch.relu(worst_vdot + self.clf_lambda * v_values)
        
        return violations, worst_vdot, v_values
    
    def falsify_gradient(self) -> Tuple[torch.Tensor, torch.Tensor]:
        states = torch.rand(self.n_samples, self.system.n_dims, device=self.controller.device) * \
                 (self.upper_limit.to(self.controller.device) - self.lower_limit.to(self.controller.device)) + \
                 self.lower_limit.to(self.controller.device)
        states.requires_grad_(True)
        
        optimizer = torch.optim.Adam([states], lr=self.learning_rate)
        
        best_violations = torch.zeros(self.n_samples, device=states.device)
        best_states = states.clone().detach()
        
        for i in range(self.max_iterations):
            violations, _, _ = self.compute_violation(states)
            mean_violation = violations.mean()
            
            improved_mask = violations > best_violations
            best_violations[improved_mask] = violations[improved_mask].detach()
            best_states[improved_mask] = states[improved_mask].detach()
            
            optimizer.zero_grad()
            (-mean_violation).backward()
            optimizer.step()
            
            with torch.no_grad():
                states.data = torch.max(torch.min(states.data, self.upper_limit.to(states.device)), self.lower_limit.to(states.device))
            
            if self.verbose and (i == 0 or (i+1) % (self.max_iterations // 5) == 0):
                print(f"  [Gradient Falsifier] Iteration {i+1}/{self.max_iterations}, Mean Violation: {mean_violation.item():.6f}")
        
        violation_mask = best_violations > 1e-5
        counter_examples = best_states[violation_mask].detach()
        violations = best_violations[violation_mask].detach()
        
        return counter_examples, violations

    def falsify_random(self) -> Tuple[torch.Tensor, torch.Tensor]:
        states = torch.rand(self.n_samples, self.system.n_dims, device=self.controller.device) * \
                 (self.upper_limit.to(self.controller.device) - self.lower_limit.to(self.controller.device)) + \
                 self.lower_limit.to(self.controller.device)
        
        with torch.no_grad():
            violations, _, _ = self.compute_violation(states)
        
        violation_mask = violations > 1e-5
        counter_examples = states[violation_mask]
        violation_values = violations[violation_mask]
            
        return counter_examples, violation_values

    def falsify_adaptive(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            random_states = torch.rand(1000, self.system.n_dims, device=self.controller.device) * \
                            (self.upper_limit.to(self.controller.device) - self.lower_limit.to(self.controller.device)) + \
                            self.lower_limit.to(self.controller.device)
            v_values = self.lyapunov_fn(random_states)
            
            v_median = torch.median(v_values)
            
            focused_states = []
            
            base_states = random_states[torch.abs(v_values - v_median) < v_median * 0.2]
            if len(base_states) > 0:
                noise_mag = 0.1 * (self.upper_limit.to(self.controller.device) - self.lower_limit.to(self.controller.device))
                noise = torch.randn(min(self.n_samples // 2, len(base_states)), self.system.n_dims, device=self.controller.device) * noise_mag
                focused_states.append(torch.clamp(base_states[:noise.shape[0]] + noise, self.lower_limit.to(self.controller.device), self.upper_limit.to(self.controller.device)))
            
            num_random = self.n_samples - sum(len(s) for s in focused_states)
            if num_random > 0:
                random_samples = torch.rand(num_random, self.system.n_dims, device=self.controller.device) * \
                                 (self.upper_limit.to(self.controller.device) - self.lower_limit.to(self.controller.device)) + \
                                 self.lower_limit.to(self.controller.device)
                focused_states.append(random_samples)
            
            states = torch.cat(focused_states)
            
            violations, _, _ = self.compute_violation(states)
        
        violation_mask = violations > 1e-5
        counter_examples = states[violation_mask]
        violation_values = violations[violation_mask]
            
        return counter_examples, violation_values
    
    def falsify(self) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
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
        
        if all_counter_examples:
            counter_examples = torch.cat(all_counter_examples)
            violations = torch.cat(all_violations)
        else:
            counter_examples = torch.tensor([])
            violations = torch.tensor([])
        
        stats["total"] = {
            "n_examples": len(counter_examples),
            "max_violation": float(violations.max().item()) if len(violations) > 0 else 0.0,
            "runtime": time.time() - start_time
        }
        
        if self.verbose:
            print(f"\nFalsification complete in {stats['total']['runtime']:.2f} seconds")
            print(f"Total counterexamples found: {stats['total']['n_examples']}")
        
        return counter_examples, violations, stats

    def visualization_2d(self, *args, **kwargs):
        pass

    def visualization_3d(self, *args, **kwargs):
        pass

    def generate_counterexample_dataset(
        self, 
        counter_examples: torch.Tensor,
        violations: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(counter_examples) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        with torch.no_grad():
            actions = self.controller.u(counter_examples)
        
        if weights is None:
            if violations.max() > violations.min():
                normalized_violations = (violations - violations.min()) / (violations.max() - violations.min())
            else:
                normalized_violations = torch.ones_like(violations)
            
            sample_weights = torch.sqrt(normalized_violations)
        else:
            sample_weights = weights
        
        return counter_examples, actions, sample_weights
