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

# Import the new IEEE-39 system
from ieee39_control_affine import IEEE39HybridSystem

from neural_clbf.systems.utils import Scenario, ScenarioList
from neural_clbf.controllers import NeuralCLBFController, BCController
from neural_clbf.experiments import ExperimentSuite
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_clbf.training.lyapunov_falsification import LyapunovFalsifier

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_ieee39_system(
    pv_penetration: float = 0.2,
    dt: float = 0.01
) -> Tuple[IEEE39HybridSystem, ScenarioList]:
    """Create the IEEE-39 hybrid system.

    Args:
        pv_penetration (float): The fraction of generation from PV at each bus.
        dt (float): Timestep for simulation and control.

    Returns:
        Tuple[IEEE39HybridSystem, ScenarioList]: The system instance and a list of scenarios.
    """
    # Define nominal parameters for the system
    nominal_params: Scenario = {
        "pv_ratio": torch.full((10,), pv_penetration),
        "T_pv": torch.tensor([0.05, 0.05]),      # P, Q inverter time constants (s)
        "tau_network": torch.tensor([0.01, 0.01]), # V, theta network dynamics (s)
    }

    # Create a list of scenarios for robust training
    # Here, we vary the PV penetration level
    scenarios = []
    for pv_level in np.linspace(0.1, 0.5, 5):
        scenarios.append({
            "pv_ratio": torch.full((10,), pv_level),
            "T_pv": torch.tensor([0.05, 0.05]),
            "tau_network": torch.tensor([0.01, 0.01]),
        })

    # Instantiate the system
    system = IEEE39HybridSystem(
        nominal_params=nominal_params,
        dt=dt,
        controller_dt=dt,
        scenarios=scenarios
    )

    return system, scenarios


def collect_clbf_trajectories_with_events(
    controller: NeuralCLBFController,
    system: IEEE39HybridSystem,
    scenarios: ScenarioList,
    n_trajectories: int = 100,
    n_steps: int = 250,
    pre_fault_steps: int = 50,
    load_change_range: Tuple[float, float] = (1.10, 1.30)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect trajectories using the CLBF controller by simulating a sudden
    load change event. This is the physically realistic way to generate
    disturbance data for power systems.

    Args:
        controller: The trained CLBF controller.
        system: The IEEE-39 system.
        scenarios: List of scenarios for simulation.
        n_trajectories: Number of trajectories to collect.
        n_steps: Total number of steps per trajectory.
        pre_fault_steps: The number of steps to simulate before the event.
        load_change_range: The range [min, max] for the load multiplier.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of state and action tensors.
    """
    logger.info("Collecting trajectories from physically realistic load change events...")

    all_states = []
    all_actions = []

    # All trajectories start from the exact same pre-fault equilibrium point.
    x_init = system.goal_point.repeat(n_trajectories, 1)

    for i in range(n_trajectories):
        if i % 10 == 0:
            logger.info(f"Simulating trajectory {i+1}/{n_trajectories}")

        # Choose a random base scenario for this trajectory's parameters
        scenario_idx = np.random.randint(len(scenarios))
        base_scenario = scenarios[scenario_idx]

        # 1. Simulate the stable pre-fault period.
        traj1 = system.simulate(
            x_init[i].unsqueeze(0),
            pre_fault_steps,
            controller.u,
            controller_period=controller.controller_period,
            params=base_scenario,
        )
        
        # 2. Define the post-fault scenario by applying a sudden load change.
        load_multiplier = np.random.uniform(load_change_range[0], load_change_range[1])
        bus_to_change = np.random.randint(0, system.n_gen)
        
        event_scenario = base_scenario.copy()
        new_PL_base = system.PL_base.clone()
        new_QL_base = system.QL_base.clone()
        new_PL_base[bus_to_change] *= load_multiplier
        new_QL_base[bus_to_change] *= load_multiplier
        
        post_fault_system = IEEE39HybridSystem(nominal_params=event_scenario)
        post_fault_system.PL_base = new_PL_base
        post_fault_system.QL_base = new_QL_base
        
        # 3. Simulate the post-fault dynamics.
        x_post_fault_init = traj1[0, -1, :].unsqueeze(0)
        post_fault_steps = n_steps - pre_fault_steps
        if post_fault_steps > 0:
            traj2 = post_fault_system.simulate(
                x_post_fault_init,
                post_fault_steps,
                controller.u,
                controller_period=controller.controller_period,
                params=event_scenario,
            )
            # 4. Stitch trajectories.
            full_traj_states = torch.cat((traj1[0, :, :], traj2[0, 1:, :]), dim=0)
        else:
            full_traj_states = traj1[0, :, :]

        # Get the corresponding actions for the full state trajectory
        with torch.no_grad():
            actions = controller.u(full_traj_states)

        all_states.append(full_traj_states)
        all_actions.append(actions)

    # Concatenate all trajectories into a single dataset
    states_tensor = torch.cat(all_states, dim=0)
    actions_tensor = torch.cat(all_actions, dim=0)

    return states_tensor, actions_tensor


def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create system and scenarios
    system, scenarios = create_ieee39_system(
        pv_penetration=args.pv_penetration,
        dt=args.dt
    )
    logger.info(f"Created IEEE-39 Hybrid System with {args.pv_penetration*100:.0f}% PV penetration.")

    # Create datamodule (used by the CLBF controller for its internal workings)
    datamodule = EpisodicDataModule(
        dynamics_model=system,
        scenarios=scenarios,
        initial_points_fn=lambda n: system.goal_point.repeat(n, 1),
        batch_size=args.batch_size,
    )

    # Create experiment suite (can be used for plotting/evaluation)
    experiment_suite = ExperimentSuite([])

    # Create log directories
    os.makedirs(args.logdir, exist_ok=True)
    
    # Train CLBF controller
    clbf_controller = NeuralCLBFController(
        dynamics_model=system,
        scenarios=scenarios,
        datamodule=datamodule,
        experiment_suite=experiment_suite,
        clbf_hidden_layers=args.clbf_hidden_layers,
        clbf_hidden_size=args.clbf_hidden_size,
        clf_lambda=1.0,
        safe_level=1.0,
        clf_relaxation_penalty=1e2,
        controller_period=system.dt,
        primal_learning_rate=args.clbf_lr,
        epochs_per_episode=5,
        num_init_epochs=10,
        barrier=False,
        use_batch_norm=True,
    )

    # Set up logger and callbacks
    tb_logger = TensorBoardLogger(save_dir=args.logdir, name="clbf")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(args.logdir, "checkpoints"),
        filename="best_clbf",
        save_top_k=1,
        mode="min",
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=args.clbf_epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator="auto",
        log_every_n_steps=10,
    )

    # Train the CLBF controller
    trainer.fit(clbf_controller, datamodule=datamodule)
    logger.info("CLBF Controller training complete.")
    
    # Load the best model
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"Loading best CLBF model from {best_model_path}")
        # **CRITICAL FIX**: Pass the dynamics model when loading from checkpoint
        clbf_controller = NeuralCLBFController.load_from_checkpoint(best_model_path, dynamics_model=system)
    
    # Collect trajectories using the expert CLBF controller
    states, actions = collect_clbf_trajectories_with_events(
        controller=clbf_controller,
        system=system,
        scenarios=scenarios,
        n_trajectories=args.n_trajectories,
        n_steps=args.n_steps_per_trajectory,
    )
    logger.info(f"Collected {states.shape[0]} state-action pairs for BC training.")
    
    logger.info("Script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Neural CLBF controller for the IEEE-39 Hybrid System")

    # System parameters
    parser.add_argument("--pv_penetration", type=float, default=0.2, help="Fraction of PV generation at each bus (0.0 to 1.0)")
    parser.add_argument("--dt", type=float, default=0.01, help="Timestep for simulation and control")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # CLBF parameters
    parser.add_argument("--clbf_epochs", type=int, default=50, help="Maximum number of epochs for CLBF training")
    parser.add_argument("--clbf_hidden_layers", type=int, default=3, help="Number of hidden layers in CLBF network")
    parser.add_argument("--clbf_hidden_size", type=int, default=256, help="Size of hidden layers in CLBF network")
    parser.add_argument("--clbf_lr", type=float, default=1e-4, help="Learning rate for CLBF training")
    parser.add_argument("--logdir", type=str, default="logs/ieee39_clbf", help="Directory for logs and checkpoints")

    # Data collection parameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--n_trajectories", type=int, default=200, help="Number of trajectories to collect for BC training")
    parser.add_argument("--n_steps_per_trajectory", type=int, default=250, help="Number of steps per trajectory")

    args = parser.parse_args()
    main(args)
