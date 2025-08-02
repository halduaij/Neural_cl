import torch
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import os
import time
import argparse
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from neural_clbf.systems import SwingEquationSystem
from neural_clbf.systems.utils import Scenario, ScenarioList
from neural_clbf.controllers import NeuralCLBFController, BCController
from neural_clbf.experiments import ExperimentSuite
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_clbf.controllers.clf_controller import CLFController
from neural_clbf.training.lyapunov_falsification import LyapunovFalsifier

def create_swing_system(n_nodes: int = 3, dt: float = 0.01, max_rocof: float = 1.0) -> Tuple[SwingEquationSystem, ScenarioList]:
    """Create a swing equation system with n nodes.

    args:
        n_nodes: number of nodes in the system
        dt: timestep for simulation and control
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
        dt=dt,
        controller_dt=dt,
        scenarios=scenarios,
        max_rocof=max_rocof
    )
    
    # Create more realistic scenarios with parameter variations
    additional_scenarios = system.create_realistic_scenarios(n_scenarios=5, variation=0.2)
    scenarios.extend(additional_scenarios)
    
    return system, scenarios

def create_datamodule(
    system: SwingEquationSystem,
    scenarios: ScenarioList,
    batch_size: int = 128,
    initial_state_generator_fn: Optional[callable] = None,
) -> EpisodicDataModule:
    """Create a datamodule for training the neural CLBF controller.

    args:
        system: the swing equation system
        scenarios: list of scenarios
        batch_size: batch size for training
        initial_state_generator_fn: optional function to generate initial states
    returns:
        datamodule: the episodic datamodule
    """
    if initial_state_generator_fn is None:
        # Default initial state generator samples uniformly from state limits
        def initial_state_generator_fn(n_samples: int) -> torch.Tensor:
            upper_limit, lower_limit = system.state_limits
            return torch.rand(n_samples, system.n_dims) * (upper_limit - lower_limit) + lower_limit
    
    datamodule = EpisodicDataModule(
        dynamics_model=system,
        scenarios=scenarios,
        initial_points_fn=initial_state_generator_fn,
        batch_size=batch_size,
        num_samples=5000,  # Initial number of samples
        val_split=0.2,
    )
    
    return datamodule

def create_experiment_suite() -> ExperimentSuite:
    """Create an experiment suite for evaluating the controller.

    returns:
        experiment_suite: the experiment suite
    """
    # Create experiment suite (empty for now, can be extended)
    experiment_suite = ExperimentSuite([])
    
    return experiment_suite

def train_neural_clbf(
    system: SwingEquationSystem,
    scenarios: ScenarioList,
    datamodule: EpisodicDataModule,
    experiment_suite: ExperimentSuite,
    max_epochs: int = 100,
    hidden_layers: int = 3,
    hidden_size: int = 128,
    learning_rate: float = 1e-3,
    logdir: str = "logs/swing_equation_clbf",
    barrier_weight: float = 0.5,
) -> NeuralCLBFController:
    """Train a neural CLBF controller for the swing equation system.

    args:
        system: the swing equation system
        scenarios: list of scenarios
        datamodule: datamodule for training
        experiment_suite: experiment suite for evaluation
        max_epochs: maximum number of epochs for training
        hidden_layers: number of hidden layers in the CLBF network
        hidden_size: number of neurons per hidden layer
        learning_rate: learning rate for training
        logdir: directory for logging
        barrier_weight: weight for RoCoF barrier loss
    returns:
        controller: the trained neural CLBF controller
    """
    print("Training Neural CLBF Controller with RoCoF Safety...")
    
    # Create the neural CLBF controller
    controller = NeuralCLBFController(
        dynamics_model=system,
        scenarios=scenarios,
        datamodule=datamodule,
        experiment_suite=experiment_suite,
        clbf_hidden_layers=hidden_layers,
        clbf_hidden_size=hidden_size,
        clf_lambda=1.0,
        safe_level=1.0,
        clf_relaxation_penalty=50.0,
        controller_period=system.dt,
        primal_learning_rate=learning_rate,
        epochs_per_episode=5,
        num_init_epochs=10,
        barrier=True,
        add_nominal=True,
        normalize_V_nominal=True,
        use_batch_norm=True,
        dropout_rate=0.1,
        barrier_weight=barrier_weight,  # Add barrier weight to controller config
    )
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(logdir, "checkpoints"),
        filename="epoch={epoch}-step={step}",
        save_top_k=3,
        mode="min",
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
    )
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=logdir,
        name="",
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=10,
        deterministic=False,
        accelerator="auto",
    )
    
    # Train the controller
    trainer.fit(controller, datamodule=datamodule)
    
    # Load the best checkpoint
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Loading best model from {best_model_path}")
        controller = NeuralCLBFController.load_from_checkpoint(best_model_path)
    
    return controller

def extract_trained_V(controller: NeuralCLBFController) -> callable:
    """Extract the trained Lyapunov function from the CLBF controller.
    
    args:
        controller: the trained CLBF controller
    returns:
        V_fn: callable function that computes V(x)
    """
    # Create a function that computes V(x)
    def V_fn(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return controller.V(x)
    
    return V_fn

def collect_clbf_trajectories(
    controller: NeuralCLBFController,
    system: SwingEquationSystem,
    scenarios: ScenarioList,
    n_trajectories: int = 100,
    n_steps: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect trajectories using the CLBF controller.
    
    args:
        controller: the trained CLBF controller
        system: the swing equation system
        scenarios: list of scenarios
        n_trajectories: number of trajectories to collect
        n_steps: number of steps per trajectory
    returns:
        states: tensor of states [n_trajectories * n_steps, n_dims]
        actions: tensor of actions [n_trajectories * n_steps, n_controls]
    """
    print("Collecting CLBF trajectories...")
    
    # Initialize lists to store states and actions
    all_states = []
    all_actions = []
    
    # Sample initial states from state limits
    upper_limit, lower_limit = system.state_limits
    x_inits = torch.rand(n_trajectories, system.n_dims) * (upper_limit - lower_limit) + lower_limit
    
    # Simulate trajectories
    for i in range(n_trajectories):
        if i % 10 == 0:
            print(f"Simulating trajectory {i}/{n_trajectories}")
        
        # Choose a random scenario for this trajectory
        scenario_idx = np.random.randint(len(scenarios))
        scenario = scenarios[scenario_idx]
        
        # Simulate the trajectory
        trajectory = system.simulate(
            x_inits[i].unsqueeze(0),
            n_steps,
            controller.u,
            controller_period=controller.controller_period,
            params=scenario,
        )
        
        # Extract states and actions
        states = trajectory["x"]
        actions = trajectory["u"]
        
        # Add to lists
        all_states.append(states)
        all_actions.append(actions)
    
    # Concatenate all trajectories
    states = torch.cat(all_states, dim=0)
    actions = torch.cat(all_actions, dim=0)
    
    return states, actions

def compute_lyapunov_loss(
    system: SwingEquationSystem,
    expert_controller: NeuralCLBFController,
    V_fn: callable,
    states: torch.Tensor,
    actions: torch.Tensor,
    clf_lambda: float = 1.0,
    epsilon: float = 1e-5,
    barrier_weight: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the Lyapunov-based loss for behavior cloning.
    
    args:
        system: The swing equation system
        expert_controller: The expert CLBF controller
        V_fn: The Lyapunov function
        states: Batch of states [batch_size, n_dims]
        actions: Batch of actions [batch_size, n_controls]
        clf_lambda: Convergence rate for CLF
        epsilon: Small constant for numerical stability
        barrier_weight: Weight for the barrier term
    
    returns:
        lyapunov_loss: The combined Lyapunov and barrier loss
        violation_rate: Rate of Lyapunov condition violations
        worst_vdot: Worst-case V-dot values
    """
    # Compute V(x)
    V = V_fn(states)
    
    # Compute Lie derivatives for all scenarios (robust formulation)
    Lf_V, Lg_V = expert_controller.V_lie_derivatives(states)
    
    # Initialize variables to track worst-case violation
    batch_size = states.shape[0]
    worst_vdot = torch.zeros(batch_size, 1, device=states.device)
    
    # Compute Vdot across all scenarios and find worst case
    for i in range(len(expert_controller.scenarios)):
        # Compute Vdot = Lf_V + Lg_V * u for this scenario
        Vdot_scenario = Lf_V[:, i, :] + torch.bmm(
            Lg_V[:, i, :].unsqueeze(1),
            actions.reshape(-1, system.n_controls, 1),
        ).squeeze()
        
        # Update worst-case Vdot (most positive/least negative)
        if i == 0:
            worst_vdot = Vdot_scenario
        else:
            worst_vdot = torch.maximum(worst_vdot, Vdot_scenario)
    
    # Lyapunov condition: Vdot + lambda*V <= 0
    # Add epsilon for numerical stability
    violation = F.relu(worst_vdot + clf_lambda * V + epsilon)
    
    # Compute violation rate (for monitoring)
    violation_rate = (violation > epsilon).float().mean()
    
    # Compute CLF loss (mean squared violation)
    clf_loss = (violation ** 2).mean()
    
    # Compute RoCoF barrier loss
    barrier_values = system.get_rocof_barrier_value(states, actions)
    barrier_violation = F.relu(-barrier_values + epsilon)
    barrier_loss = (barrier_violation ** 2).mean()
    
    # Combine losses
    lyapunov_loss = clf_loss + barrier_weight * barrier_loss
    
    return lyapunov_loss, violation_rate, worst_vdot

def verify_safety_constraints(
    system: SwingEquationSystem,
    controller: BCController,
    expert_controller: NeuralCLBFController,
    V_fn: callable,
    n_samples: int = 1000,
    clf_lambda: float = 1.0,
    epsilon: float = 1e-5
) -> Dict[str, float]:
    """Verify the safety constraints of the BC controller.
    
    args:
        system: The swing equation system
        controller: The BC controller to verify
        expert_controller: The expert CLBF controller
        V_fn: The Lyapunov function
        n_samples: Number of samples for verification
        clf_lambda: Convergence rate for CLF
        epsilon: Small constant for numerical stability
    
    returns:
        stats: Dictionary of safety statistics
    """
    print("\nVerifying safety constraints...")
    
    # Generate random states within the state limits
    upper_limit, lower_limit = system.state_limits
    states = torch.rand(n_samples, system.n_dims) * (upper_limit - lower_limit) + lower_limit
    
    # Get actions from both controllers
    with torch.no_grad():
        bc_actions = controller.u(states)
        expert_actions = expert_controller.u(states)
    
    # Compute V and V_dot for both controllers
    with torch.no_grad():
        V = V_fn(states)
        Lf_V, Lg_V = expert_controller.V_lie_derivatives(states)
        
        # For BC controller
        bc_vdot = torch.zeros_like(V)
        for i in range(len(expert_controller.scenarios)):
            bc_vdot_i = Lf_V[:, i, :] + torch.bmm(
                Lg_V[:, i, :].unsqueeze(1), 
                bc_actions.reshape(-1, system.n_controls, 1)
            ).squeeze()
            bc_vdot = torch.maximum(bc_vdot, bc_vdot_i)
        
        # For expert controller
        expert_vdot = torch.zeros_like(V)
        for i in range(len(expert_controller.scenarios)):
            expert_vdot_i = Lf_V[:, i, :] + torch.bmm(
                Lg_V[:, i, :].unsqueeze(1), 
                expert_actions.reshape(-1, system.n_controls, 1)
            ).squeeze()
            expert_vdot = torch.maximum(expert_vdot, expert_vdot_i)
    
    # Calculate CLF constraint violations: V_dot + lambda*V > 0
    bc_violation = (bc_vdot + clf_lambda * V > epsilon).float().mean().item()
    expert_violation = (expert_vdot + clf_lambda * V > epsilon).float().mean().item()
    
    # Calculate RoCoF barrier violations
    with torch.no_grad():
        bc_rocof = system.compute_rocof(states, bc_actions)
        expert_rocof = system.compute_rocof(states, expert_actions)
        
        bc_rocof_violation = (torch.abs(bc_rocof) > system.max_rocof).any(dim=1).float().mean().item()
        expert_rocof_violation = (torch.abs(expert_rocof) > system.max_rocof).any(dim=1).float().mean().item()
        
        max_bc_rocof = torch.abs(bc_rocof).max().item()
        max_expert_rocof = torch.abs(expert_rocof).max().item()
    
    # Calculate statistics
    stats = {
        "bc_clf_violation_rate": bc_violation,
        "expert_clf_violation_rate": expert_violation,
        "bc_rocof_violation_rate": bc_rocof_violation,
        "expert_rocof_violation_rate": expert_rocof_violation,
        "max_bc_rocof": max_bc_rocof,
        "max_expert_rocof": max_expert_rocof,
        "imitation_error": F.mse_loss(bc_actions, expert_actions).item(),
    }
    
    # Print results
    print(f"BC controller CLF violation rate: {bc_violation:.4f}")
    print(f"Expert controller CLF violation rate: {expert_violation:.4f}")
    print(f"BC controller RoCoF violation rate: {bc_rocof_violation:.4f}")
    print(f"Expert controller RoCoF violation rate: {expert_rocof_violation:.4f}")
    print(f"Maximum BC controller RoCoF: {max_bc_rocof:.4f} Hz/s")
    print(f"Maximum Expert controller RoCoF: {max_expert_rocof:.4f} Hz/s")
    print(f"Imitation error (MSE): {stats['imitation_error']:.6f}")
    
    return stats

def run_lyapunov_falsification(
    system: SwingEquationSystem,
    controller: NeuralCLBFController,
    V_fn: callable,
    n_samples: int = 2000,
    max_iterations: int = 20,
    use_methods: List[str] = ["gradient", "random", "adaptive"],
    logdir: str = "logs/swing_equation_clbf/falsification",
    visualize: bool = True,
    verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Run Lyapunov falsification to find counterexamples.
    
    Args:
        system: The swing equation system
        controller: The trained CLBF controller
        V_fn: The Lyapunov function
        n_samples: Number of samples for falsification
        max_iterations: Maximum iterations for gradient-based falsification
        use_methods: Falsification methods to use
        logdir: Directory to save falsification results
        visualize: Whether to create visualizations
        verbose: Whether to print progress
    
    Returns:
        counter_examples: States that violate the Lyapunov condition
        violations: Corresponding violation values
        stats: Statistics about the falsification process
    """
    # Create falsifier
    falsifier = LyapunovFalsifier(
        system=system,
        lyapunov_fn=V_fn,
        controller=controller,
        falsification_methods=use_methods,
        n_samples=n_samples,
        max_iterations=max_iterations,
        verbose=verbose
    )
    
    # Run falsification
    if verbose:
        print("\nRunning Lyapunov falsification...")
    
    counter_examples, violations, stats = falsifier.falsify()
    
    # Save results
    os.makedirs(logdir, exist_ok=True)
    if len(counter_examples) > 0:
        torch.save(counter_examples, os.path.join(logdir, "counter_examples.pt"))
        torch.save(violations, os.path.join(logdir, "violations.pt"))
    
    # Save statistics
    with open(os.path.join(logdir, "falsification_stats.txt"), 'w') as f:
        f.write("Lyapunov Falsification Statistics:\n")
        f.write(f"Total runtime: {stats['total']['runtime']:.2f} seconds\n")
        f.write(f"Total counterexamples found: {stats['total']['n_examples']}\n")
        f.write(f"Maximum violation: {stats['total']['max_violation']:.6f}\n\n")
        
        f.write("Method-specific statistics:\n")
        for method, method_stats in stats.items():
            if method != "total":
                f.write(f"- {method}: {method_stats['n_examples']} examples, "
                        f"max violation {method_stats['max_violation']:.6f}\n")
    
    # Create visualizations
    if visualize and len(counter_examples) > 0:
        vis_dir = os.path.join(logdir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 2D visualizations for different dimension pairs
        n_dims = system.n_dims
        
        # Create visualization for first two dimensions
        falsifier.visualization_2d(
            counter_examples=counter_examples,
            violations=violations,
            dims=(0, 1),
            title="Lyapunov Violations (Dims 0, 1)",
            save_path=os.path.join(vis_dir, "violations_2d_dims_0_1.png")
        )
        
        # If there are more dimensions, create additional visualizations
        if n_dims > 2:
            falsifier.visualization_2d(
                counter_examples=counter_examples,
                violations=violations,
                dims=(1, 2),
                title="Lyapunov Violations (Dims 1, 2)",
                save_path=os.path.join(vis_dir, "violations_2d_dims_1_2.png")
            )
            
            # If we have angular dimensions and frequencies, visualize them separately
            if hasattr(system, 'angle_dims') and len(system.angle_dims) > 0:
                # Find the first non-angle dimension
                non_angle_dims = [i for i in range(n_dims) if i not in system.angle_dims]
                if len(non_angle_dims) > 0 and len(system.angle_dims) > 0:
                    falsifier.visualization_2d(
                        counter_examples=counter_examples,
                        violations=violations,
                        dims=(system.angle_dims[0], non_angle_dims[0]),
                        title="Lyapunov Violations (Angle vs. Frequency)",
                        save_path=os.path.join(vis_dir, "violations_2d_angle_freq.png")
                    )
        
        # Create 3D visualization if we have at least 3 dimensions
        if n_dims >= 3:
            falsifier.visualization_3d(
                counter_examples=counter_examples,
                violations=violations,
                dims=(0, 1, 2),
                title="Lyapunov Violations in 3D",
                save_path=os.path.join(vis_dir, "violations_3d.png")
            )
    
    return counter_examples, violations, stats

def train_bc_controller_with_falsification(
    system: SwingEquationSystem,
    scenarios: ScenarioList,
    experiment_suite: ExperimentSuite,
    expert_controller: NeuralCLBFController,
    V_fn: callable,
    states: torch.Tensor,
    actions: torch.Tensor,
    counter_examples: torch.Tensor = None,
    violations: torch.Tensor = None,
    hidden_layers: List[int] = [128, 128, 64],
    learning_rate: float = 5e-4,
    batch_size: int = 64,
    epochs: int = 200,
    lyapunov_weight: float = 0.1,
    falsification_weight: float = 2.0,
    logdir: str = "logs/swing_equation_bc",
    weight_decay: float = 1e-4,
    grad_clip_value: float = 1.0,
    validation_frequency: int = 5
) -> BCController:
    """Train a BC controller with falsification counterexamples.
    
    This training method includes counterexamples from Lyapunov falsification
    to improve the safety guarantees of the BC controller.
    
    Args:
        system: The swing equation system
        scenarios: List of parameter scenarios
        experiment_suite: Suite of experiments for evaluation
        expert_controller: The trained CLBF controller
        V_fn: The Lyapunov function
        states: Regular training states
        actions: Regular training actions
        counter_examples: States that violate the Lyapunov condition
        violations: Corresponding violation values
        hidden_layers: List of hidden layer sizes
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        epochs: Number of training epochs
        lyapunov_weight: Weight for Lyapunov regularization
        falsification_weight: Weight for falsification examples in training
        logdir: Directory for logging
        weight_decay: L2 regularization parameter
        grad_clip_value: Maximum norm of gradients
        validation_frequency: How often to run validation
        
    Returns:
        bc_controller: The trained BC controller
    """
    print("Training BC Controller with Falsification...")
    
    # Create the falsifier object to help process counterexamples
    falsifier = LyapunovFalsifier(
        system=system,
        lyapunov_fn=V_fn,
        controller=expert_controller,
        n_samples=1  # Dummy value, we're not using falsification directly
    )
    
    # Process counterexamples if provided
    counterexample_dataset = None
    if counter_examples is not None and len(counter_examples) > 0:
        # Generate actions and weights for counterexamples
        counter_states, counter_actions, sample_weights = falsifier.generate_counterexample_dataset(
            counter_examples, violations
        )
        
        # Create weighted dataset for counterexamples
        counter_weights = sample_weights * falsification_weight
        counterexample_dataset = TensorDataset(counter_states, counter_actions, counter_weights)
        print(f"Added {len(counter_states)} counterexamples to training")
    
    # Create the BC controller
    bc_controller = BCController(
        dynamics_model=system,
        scenarios=scenarios,
        experiment_suite=experiment_suite,
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
        batch_size=batch_size,
        controller_period=system.dt,
    )
    
    # Use Adam optimizer with weight decay for better regularization
    bc_controller.optimizer = torch.optim.AdamW(
        bc_controller.policy_network.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        bc_controller.optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True,
        min_lr=1e-6,
    )
    
    # Create regular dataset
    regular_dataset = TensorDataset(states, actions, torch.ones(len(states)))
    
    # Combine datasets if we have counterexamples
    if counterexample_dataset is not None:
        combined_dataset = ConcatDataset([regular_dataset, counterexample_dataset])
    else:
        combined_dataset = regular_dataset
    
    # Split into train and validation
    dataset_size = len(combined_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    # Initialize lists to track metrics
    train_losses = []
    imitation_losses = []
    lyapunov_losses = []
    val_losses = []
    val_violation_rates = []
    lr_history = []
    
    # Save the initial model
    torch.save(bc_controller.state_dict(), os.path.join(logdir, "initial_model.pt"))
    
    for epoch in range(epochs):
        # Training
        bc_controller.train()
        train_loss = 0.0
        train_imitation_loss = 0.0
        train_lyapunov_loss = 0.0
        n_batches = 0
        
        for batch_states, batch_actions, batch_weights in train_loader:
            # Forward pass
            predicted_actions = bc_controller.policy_network(batch_states)
            
            # Compute weighted imitation (MSE) loss
            batch_weights = batch_weights.view(-1, 1)  # Reshape for broadcasting
            imitation_loss = torch.mean(batch_weights * F.mse_loss(predicted_actions, batch_actions, reduction='none'))
            
            # Compute Lyapunov loss using our helper function
            lyapunov_loss, violation_rate, _ = compute_lyapunov_loss(
                system, expert_controller, V_fn, batch_states, predicted_actions,
                expert_controller.clf_lambda
            )
            
            # Adaptive weighting: increase weight if violations are high
            adaptive_weight = lyapunov_weight * (1.0 + 10.0 * violation_rate.item())
            
            # Total loss
            loss = imitation_loss + adaptive_weight * lyapunov_loss
            
            # Backward pass with gradient clipping for stability
            bc_controller.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bc_controller.policy_network.parameters(), grad_clip_value)
            bc_controller.optimizer.step()
            
            # Accumulate losses
            train_loss += loss.item()
            train_imitation_loss += imitation_loss.item()
            train_lyapunov_loss += lyapunov_loss.item()
            n_batches += 1
        
        # Compute average training losses
        train_loss /= n_batches
        train_imitation_loss /= n_batches
        train_lyapunov_loss /= n_batches
        
        # Save losses for tracking
        train_losses.append(train_loss)
        imitation_losses.append(train_imitation_loss)
        lyapunov_losses.append(train_lyapunov_loss)
        lr_history.append(bc_controller.optimizer.param_groups[0]['lr'])
        
        # Validation (run less frequently to speed up training)
        if epoch % validation_frequency == 0:
            bc_controller.eval()
            val_loss = 0.0
            val_violation_sum = 0.0
            n_batches = 0
            
            with torch.no_grad():
                for batch_states, batch_actions, batch_weights in val_loader:
                    # Forward pass
                    predicted_actions = bc_controller.policy_network(batch_states)
                    
                    # Compute weighted imitation (MSE) loss
                    batch_weights = batch_weights.view(-1, 1)  # Reshape for broadcasting
                    imitation_loss = torch.mean(batch_weights * F.mse_loss(predicted_actions, batch_actions, reduction='none'))
                    
                    # Compute Lyapunov loss
                    lyapunov_loss, violation_rate, _ = compute_lyapunov_loss(
                        system, expert_controller, V_fn, batch_states, predicted_actions,
                        expert_controller.clf_lambda
                    )
                    
                    # Total loss (using same weight as in training)
                    loss = imitation_loss + lyapunov_weight * lyapunov_loss
                    
                    val_loss += loss.item()
                    val_violation_sum += violation_rate.item()
                    n_batches += 1
            
            # Compute average validation loss and violation rate
            val_loss /= n_batches
            val_violation_rate = val_violation_sum / n_batches
            
            # Save for tracking
            val_losses.append(val_loss)
            val_violation_rates.append(val_violation_rate)
            
            # Update learning rate scheduler
            scheduler.step(val_loss)
            
            # Print progress with more detailed information
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"Imitation Loss: {train_imitation_loss:.6f} | Lyapunov Loss: {train_lyapunov_loss:.6f} | "
                  f"Violation Rate: {val_violation_rate:.4f} | LR: {lr_history[-1]:.6f}")
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(bc_controller.state_dict(), os.path.join(logdir, "best_model.pt"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        else:
            # Print less detailed progress info when not running validation
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f}")
    
    # Load best model
    bc_controller.load_state_dict(torch.load(os.path.join(logdir, "best_model.pt")))
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Save training metrics
    metrics = {
        'train_losses': train_losses,
        'imitation_losses': imitation_losses,
        'lyapunov_losses': lyapunov_losses,
        'val_losses': val_losses,
        'val_violation_rates': val_violation_rates,
        'lr_history': lr_history
    }
    torch.save(metrics, os.path.join(logdir, "training_metrics.pt"))
    
    # Plot training curves
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        val_epochs = list(range(0, epochs, validation_frequency))[:len(val_losses)]
        plt.plot(val_epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(imitation_losses, label='Imitation Loss')
    plt.plot(lyapunov_losses, label='Lyapunov Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Component Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    if val_violation_rates:
        plt.plot(val_epochs, val_violation_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Violation Rate')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(lr_history)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, "training_curves.png"))
    plt.close()
    
    # Verify safety constraints of the trained controller
    safety_stats = verify_safety_constraints(
        system, bc_controller, expert_controller, V_fn,
        n_samples=5000, clf_lambda=expert_controller.clf_lambda
    )
    
    # Save and print safety verification results
    with open(os.path.join(logdir, "safety_verification.txt"), 'w') as f:
        f.write("Safety Verification Results:\n")
        for key, value in safety_stats.items():
            f.write(f"{key}: {value:.6f}\n")
    
    print("\nSafety Verification Results:")
    for key, value in safety_stats.items():
        print(f"  {key}: {value:.6f}")
    
    return bc_controller

def train_bc_controller(
    system: SwingEquationSystem,
    scenarios: ScenarioList,
    experiment_suite: ExperimentSuite,
    expert_controller: NeuralCLBFController,
    V_fn: callable,
    states: torch.Tensor,
    actions: torch.Tensor,
    hidden_layers: List[int] = [128, 128, 64],
    learning_rate: float = 5e-4,
    batch_size: int = 64,
    epochs: int = 200,
    lyapunov_weight: float = 0.1,
    logdir: str = "logs/swing_equation_bc",
    weight_decay: float = 1e-4,
    grad_clip_value: float = 1.0,
    validation_frequency: int = 5
) -> BCController:
    """Train a BC controller to mimic the CLBF controller.
    
    args:
        system: the swing equation system
        scenarios: list of scenarios
        experiment_suite: experiment suite for evaluation
        expert_controller: the trained CLBF controller
        V_fn: the trained Lyapunov function
        states: tensor of states [n_samples, n_dims]
        actions: tensor of actions [n_samples, n_controls]
        hidden_layers: list of hidden layer sizes
        learning_rate: learning rate for training
        batch_size: batch size for training
        epochs: number of epochs for training
        lyapunov_weight: weight for Lyapunov regularization
        logdir: directory for logging
        weight_decay: L2 regularization parameter
        grad_clip_value: maximum norm of gradients
        validation_frequency: how often to run validation
    returns:
        bc_controller: the trained BC controller
    """
    print("Training BC Controller...")
    
    # Create the BC controller with improved architecture
    bc_controller = BCController(
        dynamics_model=system,
        scenarios=scenarios,
        experiment_suite=experiment_suite,
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
        batch_size=batch_size,
        controller_period=system.dt,
    )
    
    # Use Adam optimizer with weight decay for better regularization
    bc_controller.optimizer = torch.optim.AdamW(
        bc_controller.policy_network.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        bc_controller.optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True,
        min_lr=1e-6,
    )
    
    # Create dataset
    dataset = TensorDataset(states, actions)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    # Initialize lists to track metrics
    train_losses = []
    imitation_losses = []
    lyapunov_losses = []
    val_losses = []
    val_violation_rates = []
    lr_history = []
    
    # Save the initial model
    torch.save(bc_controller.state_dict(), os.path.join(logdir, "initial_model.pt"))
    
    for epoch in range(epochs):
        # Training
        bc_controller.train()
        train_loss = 0.0
        train_imitation_loss = 0.0
        train_lyapunov_loss = 0.0
        n_batches = 0
        
        for batch_states, batch_actions in train_loader:
            # Forward pass
            predicted_actions = bc_controller.policy_network(batch_states)
            
            # Compute imitation (MSE) loss
            imitation_loss = F.mse_loss(predicted_actions, batch_actions)
            
            # Compute Lyapunov loss using our helper function
            lyapunov_loss, violation_rate, _ = compute_lyapunov_loss(
                system, expert_controller, V_fn, batch_states, predicted_actions,
                expert_controller.clf_lambda
            )
            
            # Adaptive weighting: increase weight if violations are high
            adaptive_weight = lyapunov_weight * (1.0 + 10.0 * violation_rate.item())
            
            # Total loss
            loss = imitation_loss + adaptive_weight * lyapunov_loss
            
            # Backward pass with gradient clipping for stability
            bc_controller.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bc_controller.policy_network.parameters(), grad_clip_value)
            bc_controller.optimizer.step()
            
            # Accumulate losses
            train_loss += loss.item()
            train_imitation_loss += imitation_loss.item()
            train_lyapunov_loss += lyapunov_loss.item()
            n_batches += 1
        
        # Compute average training losses
        train_loss /= n_batches
        train_imitation_loss /= n_batches
        train_lyapunov_loss /= n_batches
        
        # Save losses for tracking
        train_losses.append(train_loss)
        imitation_losses.append(train_imitation_loss)
        lyapunov_losses.append(train_lyapunov_loss)
        lr_history.append(bc_controller.optimizer.param_groups[0]['lr'])
        
        # Validation (run less frequently to speed up training)
        if epoch % validation_frequency == 0:
            bc_controller.eval()
            val_loss = 0.0
            val_violation_sum = 0.0
            n_batches = 0
            
            with torch.no_grad():
                for batch_states, batch_actions in val_loader:
                    # Forward pass
                    predicted_actions = bc_controller.policy_network(batch_states)
                    
                    # Compute imitation (MSE) loss
                    imitation_loss = F.mse_loss(predicted_actions, batch_actions)
                    
                    # Compute Lyapunov loss
                    lyapunov_loss, violation_rate, _ = compute_lyapunov_loss(
                        system, expert_controller, V_fn, batch_states, predicted_actions,
                        expert_controller.clf_lambda
                    )
                    
                    # Total loss (using same weight as in training)
                    loss = imitation_loss + lyapunov_weight * lyapunov_loss
                    
                    val_loss += loss.item()
                    val_violation_sum += violation_rate.item()
                    n_batches += 1
            
            # Compute average validation loss and violation rate
            val_loss /= n_batches
            val_violation_rate = val_violation_sum / n_batches
            
            # Save for tracking
            val_losses.append(val_loss)
            val_violation_rates.append(val_violation_rate)
            
            # Update learning rate scheduler
            scheduler.step(val_loss)
            
            # Print progress with more detailed information
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"Imitation Loss: {train_imitation_loss:.6f} | Lyapunov Loss: {train_lyapunov_loss:.6f} | "
                  f"Violation Rate: {val_violation_rate:.4f} | LR: {lr_history[-1]:.6f}")
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(bc_controller.state_dict(), os.path.join(logdir, "best_model.pt"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        else:
            # Print less detailed progress info when not running validation
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f}")
    
    # Load best model
    bc_controller.load_state_dict(torch.load(os.path.join(logdir, "best_model.pt")))
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Save training metrics
    metrics = {
        'train_losses': train_losses,
        'imitation_losses': imitation_losses,
        'lyapunov_losses': lyapunov_losses,
        'val_losses': val_losses,
        'val_violation_rates': val_violation_rates,
        'lr_history': lr_history
    }
    torch.save(metrics, os.path.join(logdir, "training_metrics.pt"))
    
    # Plot training curves
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        val_epochs = list(range(0, epochs, validation_frequency))[:len(val_losses)]
        plt.plot(val_epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(imitation_losses, label='Imitation Loss')
    plt.plot(lyapunov_losses, label='Lyapunov Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Component Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    if val_violation_rates:
        plt.plot(val_epochs, val_violation_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Violation Rate')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(lr_history)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, "training_curves.png"))
    plt.close()
    
    # Verify safety constraints of the trained controller
    safety_stats = verify_safety_constraints(
        system, bc_controller, expert_controller, V_fn,
        n_samples=5000, clf_lambda=expert_controller.clf_lambda
    )
    
    # Save and print safety verification results
    with open(os.path.join(logdir, "safety_verification.txt"), 'w') as f:
        f.write("Safety Verification Results:\n")
        for key, value in safety_stats.items():
            f.write(f"{key}: {value:.6f}\n")
    
    print("\nSafety Verification Results:")
    for key, value in safety_stats.items():
        print(f"{key}: {value:.6f}")
    
    return bc_controller

def compare_controllers(
    system: SwingEquationSystem,
    scenarios: ScenarioList,
    clbf_controller: NeuralCLBFController,
    bc_controller: BCController,
    V_fn: callable,
    n_trajectories: int = 20,
    n_steps: int = 100,
    perturbation_test: bool = True,
    disturbance_test: bool = True,
) -> Dict[str, List[Dict]]:
    """Compare the CLBF and BC controllers on random trajectories.
    
    args:
        system: the swing equation system
        scenarios: list of scenarios
        clbf_controller: the trained CLBF controller
        bc_controller: the trained BC controller
        V_fn: the trained Lyapunov function
        n_trajectories: number of trajectories to simulate
        n_steps: number of steps per trajectory
        perturbation_test: whether to test with initial state perturbations
        disturbance_test: whether to test with external disturbances
    returns:
        results: dictionary of trajectory results for each controller
    """
    print("Comparing controllers...")
    
    # Initialize dictionary to store results
    results = {
        "clbf": {
            "nominal": [],
            "perturbed": [] if perturbation_test else None,
            "disturbed": [] if disturbance_test else None,
        },
        "bc": {
            "nominal": [],
            "perturbed": [] if perturbation_test else None,
            "disturbed": [] if disturbance_test else None,
        },
    }
    
    # Sample initial states from state limits
    upper_limit, lower_limit = system.state_limits
    x_inits = torch.rand(n_trajectories, system.n_dims) * (upper_limit - lower_limit) + lower_limit
    
    # Simulate nominal trajectories
    for i in range(n_trajectories):
        print(f"Simulating nominal trajectory {i+1}/{n_trajectories}")
        
        # Choose a random scenario for this trajectory
        scenario_idx = np.random.randint(len(scenarios))
        scenario = scenarios[scenario_idx]
        
        # Simulate CLBF trajectory
        clbf_trajectory = system.simulate(
            x_inits[i].unsqueeze(0),
            n_steps,
            clbf_controller.u,
            controller_period=clbf_controller.controller_period,
            params=scenario,
        )
        
        # Simulate BC trajectory
        bc_trajectory = system.simulate(
            x_inits[i].unsqueeze(0),
            n_steps,
            bc_controller.u,
            controller_period=bc_controller.controller_period,
            params=scenario,
        )
        
        # Add Lyapunov function values to trajectories
        with torch.no_grad():
            clbf_trajectory["V"] = torch.tensor([V_fn(state).item() for state in clbf_trajectory["x"]])
            bc_trajectory["V"] = torch.tensor([V_fn(state).item() for state in bc_trajectory["x"]])
        
        # Store results
        results["clbf"]["nominal"].append(clbf_trajectory)
        results["bc"]["nominal"].append(bc_trajectory)
    
    # Simulations with perturbed initial conditions
    if perturbation_test:
        print("\nTesting with perturbed initial conditions...")
        perturbation_magnitude = 0.2  # 20% of state range
        
        for i in range(n_trajectories):
            print(f"Simulating perturbed trajectory {i+1}/{n_trajectories}")
            
            # Choose a random scenario
            scenario_idx = np.random.randint(len(scenarios))
            scenario = scenarios[scenario_idx]
            
            # Apply perturbation to initial state
            perturbation = torch.randn_like(x_inits[i]) * perturbation_magnitude * (upper_limit - lower_limit)
            perturbed_init = torch.clamp(x_inits[i] + perturbation, lower_limit, upper_limit)
            
            # Simulate CLBF trajectory
            clbf_trajectory = system.simulate(
                perturbed_init.unsqueeze(0),
                n_steps,
                clbf_controller.u,
                controller_period=clbf_controller.controller_period,
                params=scenario,
            )
            
            # Simulate BC trajectory
            bc_trajectory = system.simulate(
                perturbed_init.unsqueeze(0),
                n_steps,
                bc_controller.u,
                controller_period=bc_controller.controller_period,
                params=scenario,
            )
            
            # Add Lyapunov function values
            with torch.no_grad():
                clbf_trajectory["V"] = torch.tensor([V_fn(state).item() for state in clbf_trajectory["x"]])
                bc_trajectory["V"] = torch.tensor([V_fn(state).item() for state in bc_trajectory["x"]])
            
            # Store results
            results["clbf"]["perturbed"].append(clbf_trajectory)
            results["bc"]["perturbed"].append(bc_trajectory)
    
    # Simulations with disturbances
    if disturbance_test:
        print("\nTesting with external disturbances...")
        disturbance_magnitude = 0.1  # 10% of control range
        
        # Define a disturbance function
        def add_disturbance(u, t):
            # Time-varying sinusoidal disturbance
            freq = 0.1  # Hz
            phase = torch.sin(2 * np.pi * freq * t)
            disturbance = disturbance_magnitude * phase * torch.randn_like(u)
            return u + disturbance
        
        for i in range(n_trajectories):
            print(f"Simulating disturbed trajectory {i+1}/{n_trajectories}")
            
            # Choose a random scenario
            scenario_idx = np.random.randint(len(scenarios))
            scenario = scenarios[scenario_idx]
            
            # Create controller wrappers that add disturbances
            def clbf_disturbed(x):
                t = torch.norm(x - x_inits[i]).item()  # Use distance as proxy for time
                return add_disturbance(clbf_controller.u(x), t)
            
            def bc_disturbed(x):
                t = torch.norm(x - x_inits[i]).item()  # Use distance as proxy for time
                return add_disturbance(bc_controller.u(x), t)
            
            # Simulate trajectories with disturbances
            clbf_trajectory = system.simulate(
                x_inits[i].unsqueeze(0),
                n_steps,
                clbf_disturbed,
                controller_period=clbf_controller.controller_period,
                params=scenario,
            )
            
            bc_trajectory = system.simulate(
                x_inits[i].unsqueeze(0),
                n_steps,
                bc_disturbed,
                controller_period=bc_controller.controller_period,
                params=scenario,
            )
            
            # Add Lyapunov function values
            with torch.no_grad():
                clbf_trajectory["V"] = torch.tensor([V_fn(state).item() for state in clbf_trajectory["x"]])
                bc_trajectory["V"] = torch.tensor([V_fn(state).item() for state in bc_trajectory["x"]])
            
            # Store results
            results["clbf"]["disturbed"].append(clbf_trajectory)
            results["bc"]["disturbed"].append(bc_trajectory)
    
    return results

def compute_statistics(results: Dict[str, Dict[str, List[Dict]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute comprehensive statistics from the comparison results.
    
    args:
        results: dictionary of trajectory results for each controller and test type
    returns:
        stats: dictionary of statistics for each controller and test type
    """
    stats = {}
    
    # Process each controller
    for controller_name, test_results in results.items():
        stats[controller_name] = {}
        
        # Process each test type
        for test_type, trajectories in test_results.items():
            if trajectories is None:
                continue
                
            # Initialize statistics
            test_stats = {
                # Convergence metrics
                "mean_final_error": 0.0,
                "std_final_error": 0.0,
                "convergence_rate": 0.0,
                "settling_time": 0.0,
                
                # Control effort metrics
                "mean_control_effort": 0.0,
                "std_control_effort": 0.0,
                "peak_control": 0.0,
                
                # Lyapunov metrics
                "mean_lyapunov_value": 0.0,
                "final_lyapunov_value": 0.0,
                "lyapunov_decrease_rate": 0.0,
                "lyapunov_violations": 0.0,
                
                # Simulation metrics
                "success_rate": 0.0,
            }
            
            # Collect metrics for each trajectory
            final_errors = []
            control_efforts = []
            peak_controls = []
            settling_times = []
            convergence_rates = []
            mean_lyapunov_values = []
            final_lyapunov_values = []
            lyapunov_decrease_rates = []
            lyapunov_violations = []
            success_count = 0
            
            # Process each trajectory
            for traj in trajectories:
                # Extract data
                states = traj["x"]
                controls = traj["u"]
                times = traj["t"]
                lyapunov_values = traj["V"]
                
                # Skip if trajectory is empty
                if len(states) == 0:
                    continue
                
                # Final state error (Euclidean distance from goal)
                final_state = states[-1]
                goal_state = torch.zeros_like(final_state)
                final_error = torch.norm(final_state - goal_state).item()
                final_errors.append(final_error)
                
                # Control effort (sum of control signal magnitudes)
                control_effort = torch.sum(torch.norm(controls, dim=2)).item()
                control_efforts.append(control_effort)
                
                # Peak control
                peak_control = torch.max(torch.norm(controls, dim=2)).item()
                peak_controls.append(peak_control)
                
                # Settling time (time to reach and stay within 5% of goal)
                tolerance = 0.05 * torch.norm(states[0] - goal_state).item()
                settled = False
                settling_time = times[-1].item()  # Default to final time
                
                for j in range(len(states)):
                    error = torch.norm(states[j] - goal_state).item()
                    if error <= tolerance:
                        # Check if it stays within tolerance
                        all_settled = True
                        for k in range(j, len(states)):
                            if torch.norm(states[k] - goal_state).item() > tolerance:
                                all_settled = False
                                break
                        
                        if all_settled:
                            settled = True
                            settling_time = times[j].item()
                            break
                
                settling_times.append(settling_time)
                
                # Convergence rate (exponential decay rate of error)
                # Fit e^(-alpha*t) to error curve
                errors = [torch.norm(state - goal_state).item() for state in states]
                if len(errors) > 1 and errors[0] > 0:
                    try:
                        # Use linear regression on log(error) vs. time
                        log_errors = np.log(np.array(errors) + 1e-10)  # Add small constant for numerical stability
                        times_np = times.detach().cpu().numpy()
                        
                        # Simple linear regression
                        n = len(times_np)
                        alpha = (n * np.sum(times_np * log_errors) - np.sum(times_np) * np.sum(log_errors)) / \
                                (n * np.sum(times_np**2) - np.sum(times_np)**2)
                        
                        # Negate because we expect a negative slope for decay
                        convergence_rate = -alpha
                        convergence_rates.append(convergence_rate)
                    except:
                        # Fallback if regression fails
                        convergence_rates.append(0.0)
                
                # Lyapunov metrics
                mean_lyapunov = torch.mean(lyapunov_values).item()
                mean_lyapunov_values.append(mean_lyapunov)
                
                final_lyapunov = lyapunov_values[-1].item()
                final_lyapunov_values.append(final_lyapunov)
                
                # Lyapunov decrease rate
                lyapunov_decreases = []
                for j in range(1, len(lyapunov_values)):
                    decrease = (lyapunov_values[j-1] - lyapunov_values[j]) / (times[j] - times[j-1])
                    lyapunov_decreases.append(decrease.item())
                
                if lyapunov_decreases:
                    lyapunov_decrease_rate = np.mean(lyapunov_decreases)
                    lyapunov_decrease_rates.append(lyapunov_decrease_rate)
                
                # Count Lyapunov violations (increases in V)
                violations = sum(1 for j in range(1, len(lyapunov_values)) if lyapunov_values[j] > lyapunov_values[j-1])
                violation_rate = violations / (len(lyapunov_values) - 1) if len(lyapunov_values) > 1 else 0
                lyapunov_violations.append(violation_rate)
                
                # Success (reached within tolerance of goal)
                if final_error <= tolerance:
                    success_count += 1
            
            # Skip this test type if no valid trajectories
            if not final_errors:
                continue
            
            # Compute final statistics
            test_stats["mean_final_error"] = np.mean(final_errors)
            test_stats["std_final_error"] = np.std(final_errors)
            test_stats["mean_control_effort"] = np.mean(control_efforts)
            test_stats["std_control_effort"] = np.std(control_efforts)
            test_stats["peak_control"] = np.mean(peak_controls)
            test_stats["settling_time"] = np.mean(settling_times)
            
            if convergence_rates:
                test_stats["convergence_rate"] = np.mean(convergence_rates)
            
            test_stats["mean_lyapunov_value"] = np.mean(mean_lyapunov_values)
            test_stats["final_lyapunov_value"] = np.mean(final_lyapunov_values)
            
            if lyapunov_decrease_rates:
                test_stats["lyapunov_decrease_rate"] = np.mean(lyapunov_decrease_rates)
            
            test_stats["lyapunov_violations"] = np.mean(lyapunov_violations)
            test_stats["success_rate"] = success_count / len(trajectories)
            
            # Add to results
            stats[controller_name][test_type] = test_stats
    
    return stats

def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create system and scenarios
    system, scenarios = create_swing_system(
        n_nodes=args.n_nodes, 
        dt=args.dt,
        max_rocof=args.max_rocof
    )
    print(f"Created swing equation system with {args.n_nodes} nodes")
    print(f"RoCoF safety limit: {args.max_rocof} Hz/s")
    
    # Create datamodule
    datamodule = create_datamodule(
        system=system,
        scenarios=scenarios,
        batch_size=args.batch_size,
    )
    print("Created datamodule")
    
    # Create experiment suite
    experiment_suite = create_experiment_suite()
    print("Created experiment suite")
    
    # Create log directories
    os.makedirs(args.clbf_logdir, exist_ok=True)
    os.makedirs(args.bc_logdir, exist_ok=True)
    falsification_logdir = os.path.join(args.clbf_logdir, "falsification")
    
    # Training steps
    if args.train_clbf or not os.path.exists(os.path.join(args.clbf_logdir, "checkpoints")):
        # Train neural CLBF controller
        clbf_controller = train_neural_clbf(
            system=system,
            scenarios=scenarios,
            datamodule=datamodule,
            experiment_suite=experiment_suite,
            max_epochs=args.clbf_epochs,
            hidden_layers=args.clbf_hidden_layers,
            hidden_size=args.clbf_hidden_size,
            learning_rate=args.clbf_lr,
            logdir=args.clbf_logdir,
            barrier_weight=args.barrier_weight,
        )
        
        # Save the trained controller
        torch.save(clbf_controller.state_dict(), os.path.join(args.clbf_logdir, "clbf_controller.pt"))
        print(f"Saved CLBF controller to {os.path.join(args.clbf_logdir, 'clbf_controller.pt')}")
    else:
        # Load the trained CLBF controller
        print("Loading pre-trained CLBF controller...")
        # Find the best checkpoint
        checkpoints_dir = os.path.join(args.clbf_logdir, "checkpoints")
        checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith(".ckpt")]
        if checkpoints:
            best_checkpoint = sorted(checkpoints)[-1]  # Just take the latest for now
            checkpoint_path = os.path.join(checkpoints_dir, best_checkpoint)
            clbf_controller = NeuralCLBFController.load_from_checkpoint(checkpoint_path)
            print(f"Loaded CLBF controller from {checkpoint_path}")
        else:
            print("No checkpoints found. Training a new CLBF controller...")
            clbf_controller = train_neural_clbf(
                system=system,
                scenarios=scenarios,
                datamodule=datamodule,
                experiment_suite=experiment_suite,
                max_epochs=args.clbf_epochs,
                hidden_layers=args.clbf_hidden_layers,
                hidden_size=args.clbf_hidden_size,
                learning_rate=args.clbf_lr,
                logdir=args.clbf_logdir,
                barrier_weight=args.barrier_weight,
            )
    
    # Extract trained Lyapunov function
    V_fn = extract_trained_V(clbf_controller)
    print("Extracted trained Lyapunov function")
    
    # Collect trajectories using the CLBF controller
    states, actions = collect_clbf_trajectories(
        controller=clbf_controller,
        system=system,
        scenarios=scenarios,
        n_trajectories=args.n_trajectories,
        n_steps=args.n_steps_per_trajectory,
    )
    print(f"Collected {len(states)} state-action pairs")
    
    # Run Lyapunov falsification if requested
    counter_examples = None
    violations = None
    if args.run_falsification:
        if os.path.exists(os.path.join(falsification_logdir, "counter_examples.pt")) and not args.force_falsification:
            # Load existing falsification results
            print("Loading existing falsification results...")
            counter_examples = torch.load(os.path.join(falsification_logdir, "counter_examples.pt"))
            violations = torch.load(os.path.join(falsification_logdir, "violations.pt"))
            print(f"Loaded {len(counter_examples)} existing counterexamples")
        else:
            # Run new falsification
            counter_examples, violations, falsification_stats = run_lyapunov_falsification(
                system=system,
                controller=clbf_controller,
                V_fn=V_fn,
                n_samples=args.falsification_samples,
                max_iterations=args.falsification_iterations,
                use_methods=args.falsification_methods.split(','),
                logdir=falsification_logdir,
                visualize=True,
                verbose=True,
            )
            
            if len(counter_examples) > 0:
                print(f"Found {len(counter_examples)} Lyapunov violation counterexamples")
                print(f"Maximum violation: {violations.max().item():.6f}")
                print(f"Results saved to {falsification_logdir}")
            else:
                print("No Lyapunov violations found. This is good!")
    
    if args.train_bc or not os.path.exists(os.path.join(args.bc_logdir, "best_model.pt")):
        # Train BC controller with or without falsification
        if args.run_falsification and counter_examples is not None and len(counter_examples) > 0:
            # Train with falsification counterexamples
            bc_controller = train_bc_controller_with_falsification(
                system=system,
                scenarios=scenarios,
                experiment_suite=experiment_suite,
                expert_controller=clbf_controller,
                V_fn=V_fn,
                states=states,
                actions=actions,
                counter_examples=counter_examples,
                violations=violations,
                hidden_layers=args.bc_hidden_layers,
                learning_rate=args.bc_lr,
                batch_size=args.batch_size,
                epochs=args.bc_epochs,
                lyapunov_weight=args.lyapunov_weight,
                falsification_weight=args.falsification_weight,
                logdir=args.bc_logdir,
                weight_decay=args.weight_decay,
                grad_clip_value=args.grad_clip_value,
                validation_frequency=args.validation_frequency,
            )
        else:
            # Train without falsification
            bc_controller = train_bc_controller(
                system=system,
                scenarios=scenarios,
                experiment_suite=experiment_suite,
                expert_controller=clbf_controller,
                V_fn=V_fn,
                states=states,
                actions=actions,
                hidden_layers=args.bc_hidden_layers,
                learning_rate=args.bc_lr,
                batch_size=args.batch_size,
                epochs=args.bc_epochs,
                lyapunov_weight=args.lyapunov_weight,
                logdir=args.bc_logdir,
                weight_decay=args.weight_decay,
                grad_clip_value=args.grad_clip_value,
                validation_frequency=args.validation_frequency,
            )
        
        # Save the trained controller
        torch.save(bc_controller.state_dict(), os.path.join(args.bc_logdir, "bc_controller.pt"))
        print(f"Saved BC controller to {os.path.join(args.bc_logdir, 'bc_controller.pt')}")
    else:
        # Load the trained BC controller
        print("Loading pre-trained BC controller...")
        bc_controller = BCController(
            dynamics_model=system,
            scenarios=scenarios,
            experiment_suite=experiment_suite,
            hidden_layers=args.bc_hidden_layers,
            learning_rate=args.bc_lr,
            batch_size=args.batch_size,
            controller_period=system.dt,
        )
        bc_controller.load_state_dict(torch.load(os.path.join(args.bc_logdir, "best_model.pt")))
        print(f"Loaded BC controller from {os.path.join(args.bc_logdir, 'best_model.pt')}")
    
    # Compare controllers
    if args.evaluate:
        # Run comprehensive evaluation
        results = compare_controllers(
            system=system,
            scenarios=scenarios,
            clbf_controller=clbf_controller,
            bc_controller=bc_controller,
            V_fn=V_fn,
            n_trajectories=args.eval_n_trajectories,
            n_steps=args.eval_n_steps,
            perturbation_test=args.eval_perturbations,
            disturbance_test=args.eval_disturbances,
        )
        
        # Compute comprehensive statistics
        stats = compute_statistics(results)
        
        # Save detailed results
        torch.save(results, os.path.join(args.bc_logdir, "comparison_results.pt"))
        torch.save(stats, os.path.join(args.bc_logdir, "comparison_stats.pt"))
        print(f"Saved comparison results to {os.path.join(args.bc_logdir, 'comparison_results.pt')}")
        
        # Print summary of comparison results
        print("\nComparison Results Summary:")
        for controller, test_results in stats.items():
            print(f"\n{controller.upper()} Controller:")
            for test_type, test_stats in test_results.items():
                print(f"  {test_type.upper()}:")
                print(f"    Success rate: {test_stats['success_rate']:.2f}")
                print(f"    Final error: {test_stats['mean_final_error']:.6f} ± {test_stats['std_final_error']:.6f}")
                print(f"    Control effort: {test_stats['mean_control_effort']:.6f}")
                print(f"    Lyapunov violations: {test_stats['lyapunov_violations']:.4f}")
        
        # Generate a summary table for publication
        with open(os.path.join(args.bc_logdir, "comparison_table.txt"), 'w') as f:
            # Header
            f.write("| Metric | CLBF (Nominal) | BC (Nominal) | CLBF (Perturbed) | BC (Perturbed) | CLBF (Disturbed) | BC (Disturbed) |\n")
            f.write("|--------|---------------|-------------|-----------------|---------------|-----------------|---------------|\n")
            
            # Key metrics
            metrics = [
                ("Success Rate", "success_rate", "{:.2f}"),
                ("Final Error", "mean_final_error", "{:.4f}"),
                ("Control Effort", "mean_control_effort", "{:.4f}"),
                ("Lyapunov Violations", "lyapunov_violations", "{:.4f}"),
                ("Convergence Rate", "convergence_rate", "{:.4f}"),
                ("Settling Time", "settling_time", "{:.4f}"),
            ]
            
            for metric_name, metric_key, format_str in metrics:
                row = f"| {metric_name} | "
                
                # Add CLBF and BC values for all test types
                for controller in ["clbf", "bc"]:
                    for test_type in ["nominal", "perturbed", "disturbed"]:
                        if test_type in stats[controller] and metric_key in stats[controller][test_type]:
                            value = stats[controller][test_type][metric_key]
                            row += format_str.format(value) + " | "
                        else:
                            row += "N/A | "
                
                f.write(row + "\n")
            
            f.write("\nTable generated on " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            
            # Add information about falsification if used
            if args.run_falsification and counter_examples is not None:
                f.write(f"\nUsed {len(counter_examples)} counterexamples from Lyapunov falsification.\n")
                f.write(f"Falsification methods: {args.falsification_methods}\n")
        
        print(f"\nDetailed comparison table saved to {os.path.join(args.bc_logdir, 'comparison_table.txt')}")
        
        # Verify safety constraints explicitly
        safety_stats = verify_safety_constraints(
            system, bc_controller, clbf_controller, V_fn,
            n_samples=10000, clf_lambda=clbf_controller.clf_lambda
        )
        
        print("\nSafety Verification Results:")
        for key, value in safety_stats.items():
            print(f"  {key}: {value:.6f}")
        
        # Save verification results
        with open(os.path.join(args.bc_logdir, "safety_verification.txt"), 'w') as f:
            f.write("Safety Verification Results:\n")
            for key, value in safety_stats.items():
                f.write(f"{key}: {value:.6f}\n")
                
        # Run another falsification on the BC controller to find any remaining issues
        if args.falsify_bc:
            bc_falsification_logdir = os.path.join(args.bc_logdir, "falsification")
            print("\nRunning falsification on the trained BC controller...")
            
            bc_counter_examples, bc_violations, bc_falsification_stats = run_lyapunov_falsification(
                system=system,
                controller=bc_controller,  # Use BC controller here
                V_fn=V_fn,  # Same Lyapunov function
                n_samples=args.falsification_samples,
                max_iterations=args.falsification_iterations,
                use_methods=args.falsification_methods.split(','),
                logdir=bc_falsification_logdir,
                visualize=True,
                verbose=True,
            )
            
            if len(bc_counter_examples) > 0:
                print(f"Found {len(bc_counter_examples)} Lyapunov violations in BC controller")
                print(f"Maximum violation: {bc_violations.max().item():.6f}")
                print(f"This suggests the BC controller might need further refinement")
            else:
                print("No Lyapunov violations found in BC controller. Great success!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Neural CLBF controller followed by a BC controller for Swing Equation System")
    
    # System parameters
    parser.add_argument("--n_nodes", type=int, default=3, help="Number of nodes in the swing equation system")
    parser.add_argument("--dt", type=float, default=0.01, help="Timestep for simulation and control")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max_rocof", type=float, default=1.0, help="Maximum allowed Rate of Change of Frequency (Hz/s)")
    parser.add_argument("--barrier_weight", type=float, default=0.5, help="Weight for RoCoF barrier loss in CLBF training")
    
    # CLBF parameters
    parser.add_argument("--train_clbf", action="store_true", help="Train CLBF controller even if a checkpoint exists")
    parser.add_argument("--clbf_epochs", type=int, default=100, help="Maximum number of epochs for CLBF training")
    parser.add_argument("--clbf_hidden_layers", type=int, default=3, help="Number of hidden layers in CLBF network")
    parser.add_argument("--clbf_hidden_size", type=int, default=128, help="Number of neurons per hidden layer in CLBF network")
    parser.add_argument("--clbf_lr", type=float, default=1e-3, help="Learning rate for CLBF training")
    parser.add_argument("--clbf_logdir", type=str, default="logs/swing_equation_clbf", help="Directory for CLBF logs")
    
    # Falsification parameters
    parser.add_argument("--run_falsification", action="store_true", help="Run Lyapunov falsification to find counterexamples")
    parser.add_argument("--force_falsification", action="store_true", help="Force re-running falsification even if results exist")
    parser.add_argument("--falsification_samples", type=int, default=2000, help="Number of samples for falsification")
    parser.add_argument("--falsification_iterations", type=int, default=20, help="Maximum iterations for gradient-based falsification")
    parser.add_argument("--falsification_methods", type=str, default="gradient,random,adaptive", 
                        help="Falsification methods to use (comma-separated)")
    parser.add_argument("--falsification_weight", type=float, default=2.0, 
                        help="Weight for falsification examples in BC training")
    parser.add_argument("--falsify_bc", action="store_true", 
                        help="Run falsification on the trained BC controller after training")
    
    # BC parameters
    parser.add_argument("--train_bc", action="store_true", help="Train BC controller even if a checkpoint exists")
    parser.add_argument("--bc_epochs", type=int, default=200, help="Number of epochs for BC training")
    parser.add_argument("--bc_hidden_layers", type=str, default="128,128,64", help="Hidden layers for BC network (comma-separated)")
    parser.add_argument("--bc_lr", type=float, default=5e-4, help="Learning rate for BC training")
    parser.add_argument("--lyapunov_weight", type=float, default=0.1, help="Weight for Lyapunov regularization")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization parameter")
    parser.add_argument("--grad_clip_value", type=float, default=1.0, help="Maximum norm for gradient clipping")
    parser.add_argument("--validation_frequency", type=int, default=5, help="Validate every N epochs during training")
    parser.add_argument("--bc_logdir", type=str, default="logs/swing_equation_bc", help="Directory for BC logs")
    
    # Data collection parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--n_trajectories", type=int, default=100, help="Number of trajectories to collect for BC training")
    parser.add_argument("--n_steps_per_trajectory", type=int, default=50, help="Number of steps per trajectory for BC training")
    
    # Evaluation parameters
    parser.add_argument("--evaluate", action="store_true", help="Evaluate and compare controllers")
    parser.add_argument("--eval_n_trajectories", type=int, default=20, help="Number of trajectories for evaluation")
    parser.add_argument("--eval_n_steps", type=int, default=100, help="Number of steps per trajectory for evaluation")
    parser.add_argument("--eval_perturbations", action="store_true", help="Test with perturbed initial conditions")
    parser.add_argument("--eval_disturbances", action="store_true", help="Test with external disturbances")
    
    args = parser.parse_args()
    
    # Convert BC hidden layers string to list
    args.bc_hidden_layers = [int(x) for x in args.bc_hidden_layers.split(",")]
    
    main(args)