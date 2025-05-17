from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

from neural_clbf.controllers.neural_clbf_controller import NeuralCLBFController
from neural_clbf.controllers.bc_policy import BehaviorCloningPolicy
from neural_clbf.datamodules.bc_data_module import BehaviorCloningDataModule


def main(args):
    # Load trained CLBF controller
    controller = NeuralCLBFController.load_from_checkpoint(args.controller_ckpt)
    system = controller.dynamics_model

    # Create data module
    dm = BehaviorCloningDataModule(
        system,
        controller,
        system.training_domain,
        trajectories_per_episode=args.trajectories,
        trajectory_length=args.traj_length,
        val_split=args.val_split,
        batch_size=args.batch_size,
    )
    dm.prepare_data()

    # Policy network
    policy = BehaviorCloningPolicy(
        system.n_dims,
        system.n_controls,
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        learning_rate=args.lr,
    )

    trainer = pl.Trainer(max_epochs=args.epochs)
    trainer.fit(policy, dm)

    if args.save_path:
        trainer.save_checkpoint(args.save_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--controller_ckpt", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="bc_policy.ckpt")
    parser.add_argument("--trajectories", type=int, default=100)
    parser.add_argument("--traj_length", type=int, default=500)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--hidden_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    main(args)
