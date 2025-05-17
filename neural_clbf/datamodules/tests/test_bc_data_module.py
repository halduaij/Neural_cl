import random
import torch

from neural_clbf.datamodules.bc_data_module import BehaviorCloningDataModule
from neural_clbf.systems.tests.mock_system import MockSystem


class MockController:
    def __init__(self, model):
        self.dynamics_model = model

    def u(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], self.dynamics_model.n_controls)


def test_bc_data_module():
    random.seed(0)
    torch.manual_seed(0)
    model = MockSystem({})
    controller = MockController(model)
    initial_domain = [(-1.0, 1.0) for _ in range(model.n_dims)]
    dm = BehaviorCloningDataModule(
        model,
        controller,
        initial_domain,
        trajectories_per_episode=10,
        trajectory_length=5,
        val_split=0.2,
        batch_size=2,
    )
    dm.prepare_data()
    expected_pts = dm.trajectories_per_episode * dm.trajectory_length
    val_pts = int(expected_pts * dm.val_split)
    train_pts = expected_pts - val_pts
    assert dm.x_train.shape[0] == train_pts
    assert dm.u_train.shape[0] == train_pts
    assert dm.x_val.shape[0] == val_pts
    assert dm.u_val.shape[0] == val_pts
    assert len(dm.train_dataloader()) == train_pts // dm.batch_size
    assert len(dm.val_dataloader()) == val_pts // dm.batch_size
