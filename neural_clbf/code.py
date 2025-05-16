import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_clbf.experiments import ExperimentSuite, CLFContourExperiment, RolloutStateSpaceExperiment

from neural_clbf.systems import SwingEquationSystem  # Your implemented class
def main():
    # Define the scenarios with updated parameters
    nominal_params = {
        "M_1": 1.5,  # Increased inertia for better stability
        "M_2": 1.7,
        "D_1": 1.2,  # Adjusted damping
        "D_2": 1.5,
        "P_1": 1,  # Lowered power input for stability
        "P_2": -1,
        "K": 1,    # Coupling strength remains low
    }

    scenarios = [nominal_params]  # Add more scenarios if needed

    # Define the goal point for the system
    goal_point = torch.tensor([[torch.pi/4, -torch.pi/4, 0.0, 0.0]])  # [theta_1, theta_2, omega_1, omega_2]

    # Define the dynamics model
    dynamics_model = SwingEquationSystem(
        nominal_params,
        dt=0.01,
        controller_dt=0.05,
        scenarios=scenarios,
        goal_point=goal_point  # Pass the goal point here
    )

    # Initialize the DataModule
    initial_conditions = [
        (-torch.pi/4, torch.pi/4),  # theta ranges
        (-1.0, 1.0),  # omega ranges
    ] * 2  # For 2 nodes
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=0,
        trajectory_length=1,
        fixed_samples=10000,
        max_points=100000,
        val_split=0.1,
        batch_size=64,
    )

    # Define the experiment suite
    V_contour_experiment = CLFContourExperiment(
        "V_Contour",
        domain=[(-2.0, 2.0), (-2.0, 2.0)],
        n_grid=30,
        x_axis_index=dynamics_model.OMEGA_START,
        y_axis_index=dynamics_model.OMEGA_START + 1,
        x_axis_label="$\\omega_1$",
        y_axis_label="$\\omega_2$",
    )
    rollout_experiment = RolloutStateSpaceExperiment(
        "Rollout",
        start_x=torch.randn(4, dynamics_model.N_DIMS) * 0.1,  # Random initial conditions
        plot_x_index=dynamics_model.OMEGA_START,
        plot_x_label="$\\omega_1$",
        plot_y_index=dynamics_model.OMEGA_START + 1,
        plot_y_label="$\\omega_2$",
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=5.0,
    )
    experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment])

    # Initialize the controller
    clbf_controller = NeuralCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite=experiment_suite,
        clbf_hidden_layers=2,
        clbf_hidden_size=64,
        clf_lambda=1.0,
        controller_period=0.05,
        clf_relaxation_penalty=1e2,
        num_init_epochs=5,
        epochs_per_episode=100,
        barrier=False,  # Set to True if you want to use barrier functions
    )

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger("logs/swing_equation")
    trainer = pl.Trainer(
        logger=tb_logger,
        max_epochs=51,
        check_val_every_n_epoch=5,
    )

    # Train
    trainer.fit(clbf_controller)

if __name__ == "__main__":
    main()
