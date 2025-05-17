from typing import List, Tuple

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.controllers import Controller


class BehaviorCloningDataModule(pl.LightningDataModule):
    """DataModule for collecting state-action pairs from a controller."""

    def __init__(
        self,
        model: ControlAffineSystem,
        controller: Controller,
        initial_domain: List[Tuple[float, float]],
        trajectories_per_episode: int = 100,
        trajectory_length: int = 500,
        val_split: float = 0.1,
        batch_size: int = 64,
    ) -> None:
        super().__init__()

        assert len(initial_domain) == model.n_dims
        self.model = model
        self.controller = controller
        self.initial_domain = initial_domain
        self.trajectories_per_episode = trajectories_per_episode
        self.trajectory_length = trajectory_length
        self.val_split = val_split
        self.batch_size = batch_size

    def _sample_initial_states(self) -> torch.Tensor:
        x_init = torch.rand(self.trajectories_per_episode, self.model.n_dims)
        for i, (low, high) in enumerate(self.initial_domain):
            x_init[:, i] = x_init[:, i] * (high - low) + low
        return x_init

    def sample_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x_init = self._sample_initial_states()
        with torch.no_grad():
            x_traj = self.model.simulate(
                x_init, self.trajectory_length, self.controller.u
            )
        x = x_traj.reshape(-1, self.model.n_dims)
        with torch.no_grad():
            u = self.controller.u(x)
        return x, u

    def prepare_data(self) -> None:
        x, u = self.sample_dataset()
        indices = torch.randperm(x.shape[0])
        val_pts = int(x.shape[0] * self.val_split)
        val_idx = indices[:val_pts]
        train_idx = indices[val_pts:]
        self.x_train = x[train_idx]
        self.u_train = u[train_idx]
        self.x_val = x[val_idx]
        self.u_val = u[val_idx]
        self.train_data = TensorDataset(self.x_train, self.u_train)
        self.val_data = TensorDataset(self.x_val, self.u_val)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, batch_size=self.batch_size)
