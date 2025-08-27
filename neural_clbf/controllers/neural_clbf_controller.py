import itertools
from typing import Tuple, List, Optional
from collections import OrderedDict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.utils import ScenarioList
from neural_clbf.controllers.clf_controller import CLFController
from neural_clbf.controllers.controller_utils import normalize_with_angles
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_clbf.experiments import ExperimentSuite

def to_numpy(tensor):
    if tensor.device.type == 'cuda':
        return tensor.cpu().numpy()
    return tensor.numpy()

class NeuralCLBFController(pl.LightningModule, CLFController):
    """
    A neural rCLBF controller. Differs from the CLFController in that it uses a
    neural network to learn the CLF, and it turns it from a CLF to a CLBF by making sure
    that a level set of the CLF separates the safe and unsafe regions.
    """

    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        scenarios: ScenarioList,
        datamodule: EpisodicDataModule,
        experiment_suite: ExperimentSuite,
        clbf_hidden_layers: int = 3,
        clbf_hidden_size: int = 256,  # Increased size for complex system
        clf_lambda: float = 1.0,
        safe_level: float = 1.0,
        clf_relaxation_penalty: float = 1e2, # Increased penalty
        controller_period: float = 0.01,
        primal_learning_rate: float = 1e-4, # Reduced learning rate
        epochs_per_episode: int = 5,
        penalty_scheduling_rate: float = 0.0,
        num_init_epochs: int = 5,
        barrier: bool = True,
        add_nominal: bool = True,
        normalize_V_nominal: bool = True,
        disable_gurobi: bool = False,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1,
        reducer=None,
    ):
        """Initialize the controller."""
        super(NeuralCLBFController, self).__init__(
            dynamics_model=dynamics_model,
            scenarios=scenarios,
            experiment_suite=experiment_suite,
            clf_lambda=clf_lambda,
            clf_relaxation_penalty=clf_relaxation_penalty,
            controller_period=controller_period,
            disable_gurobi=disable_gurobi,
        )
        self.save_hyperparameters()

        self.scenarios = scenarios
        self.n_scenarios = len(scenarios)
        self.reducer = reducer 
        self.datamodule = datamodule
        self.experiment_suite = experiment_suite
        self.safe_level = safe_level
        self.unsafe_level = safe_level
        self.primal_learning_rate = primal_learning_rate
        self.epochs_per_episode = epochs_per_episode
        self.penalty_scheduling_rate = penalty_scheduling_rate
        self.num_init_epochs = num_init_epochs
        self.barrier = barrier
        self.add_nominal = add_nominal
        self.normalize_V_nominal = normalize_V_nominal
        self.V_nominal_mean = 1.0
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        x_max, x_min = dynamics_model.state_limits
        self.x_center = (x_max + x_min) / 2.0
        self.x_range = (x_max - x_min) / 2.0
        # **NUMERICAL STABILITY FIX**: Add a small epsilon to the range to avoid
        # division by zero if a state variable has a zero range.
        self.x_range[self.x_range == 0.0] += 1e-6

        self.k = 1.0
        self.x_range = self.x_range / self.k
        self.x_center[self.dynamics_model.angle_dims] = 0.0
        self.x_range[self.dynamics_model.angle_dims] = 1.0

        n_angles = len(self.dynamics_model.angle_dims)
        self.n_dims_extended = self.dynamics_model.n_dims + n_angles

        self.clbf_hidden_layers = clbf_hidden_layers
        self.clbf_hidden_size = clbf_hidden_size
        self.V_layers: OrderedDict[str, nn.Module] = OrderedDict()
        
        self.V_layers["input_linear"] = nn.Linear(
            self.n_dims_extended, self.clbf_hidden_size
        )
        if self.use_batch_norm:
            self.V_layers["input_bn"] = nn.BatchNorm1d(self.clbf_hidden_size)
        self.V_layers["input_activation"] = nn.ReLU()
        self.V_layers["input_dropout"] = nn.Dropout(self.dropout_rate)
        
        for i in range(self.clbf_hidden_layers):
            self.V_layers[f"layer_{i}_linear"] = nn.Linear(
                self.clbf_hidden_size, self.clbf_hidden_size
            )
            if self.use_batch_norm:
                self.V_layers[f"layer_{i}_bn"] = nn.BatchNorm1d(self.clbf_hidden_size)
            if i < self.clbf_hidden_layers - 1:
                self.V_layers[f"layer_{i}_activation"] = nn.ReLU()
                self.V_layers[f"layer_{i}_dropout"] = nn.Dropout(self.dropout_rate)
        
        self.V_layers["output_linear"] = nn.Linear(self.clbf_hidden_size, 1)
        
        self.V_nn = nn.Sequential(self.V_layers)

    def V_with_jacobian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the CLBF value and its Jacobian"""
        x_norm = normalize_with_angles(self.dynamics_model, x, self.x_center, self.x_range)

        bs = x_norm.shape[0]
        JV = torch.zeros(
            (bs, self.n_dims_extended, self.dynamics_model.n_dims)
        ).type_as(x)
        
        non_angle_dims = [
            i for i in range(self.dynamics_model.n_dims)
            if i not in self.dynamics_model.angle_dims
        ]
        
        for dim in non_angle_dims:
            JV[:, dim, dim] = 1.0 / self.x_range[dim].type_as(x)

        for offset, sin_idx in enumerate(self.dynamics_model.angle_dims):
            cos_idx = self.dynamics_model.n_dims + offset
            JV[:, sin_idx, sin_idx] = x_norm[:, cos_idx]
            JV[:, cos_idx, sin_idx] = -x_norm[:, sin_idx]

        V_in = x_norm
        for layer in self.V_nn:
            V_out = layer(V_in)
            
            if isinstance(layer, nn.Linear):
                JV = torch.matmul(layer.weight, JV)
            elif isinstance(layer, nn.ReLU):
                deriv = (V_in > 0).float()
                deriv_matrix = torch.diag_embed(deriv)
                JV = torch.bmm(deriv_matrix, JV)
            
            V_in = V_out

        V = 0.5 * (V_out * V_out).squeeze()
        JV = V_out.unsqueeze(2) * JV
        JV = JV.squeeze(1).unsqueeze(1)

        if self.add_nominal:
            P = self.dynamics_model.P.type_as(x)
            x0 = self.dynamics_model.goal_point.type_as(x)
            P_batched = P.unsqueeze(0).expand(x.shape[0], -1, -1)
            V_nominal = 0.5 * torch.bmm( (x-x0).unsqueeze(1), torch.bmm(P_batched, (x-x0).unsqueeze(2))).squeeze()
            JV_nominal = torch.bmm((x-x0).unsqueeze(1), P_batched)

            if self.normalize_V_nominal and self.V_nominal_mean > 0:
                V_nominal /= self.V_nominal_mean
                JV_nominal /= self.V_nominal_mean

            V = V + V_nominal
            JV = JV + JV_nominal

        return V, JV

    def prepare_data(self):
        return self.datamodule.prepare_data()

    def setup(self, stage: Optional[str] = None):
        return self.datamodule.setup(stage)

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()

    def forward(self, x):
        return self.u(x)

    def boundary_loss(
        self,
        x: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        accuracy: bool = False,
    ) -> List[Tuple[str, torch.Tensor]]:
        eps = 1e-2
        loss = []
        V = self.V(x)

        V_goal_pt = self.V(self.dynamics_model.goal_point.type_as(x))
        goal_term = 1e1 * V_goal_pt.mean()
        loss.append(("CLBF goal term", goal_term))

        if self.barrier:
            V_safe = V[safe_mask]
            safe_violation = F.relu(eps + V_safe - self.safe_level)
            safe_V_term = 1e2 * safe_violation.mean() if safe_violation.numel() > 0 else torch.tensor(0.0).type_as(x)
            loss.append(("CLBF safe region term", safe_V_term))
            if accuracy:
                safe_V_acc = (safe_violation <= eps).sum() / max(1, safe_violation.numel())
                loss.append(("CLBF safe region accuracy", safe_V_acc))

            V_unsafe = V[unsafe_mask]
            unsafe_violation = F.relu(eps + self.unsafe_level - V_unsafe)
            unsafe_V_term = 1e2 * unsafe_violation.mean() if unsafe_violation.numel() > 0 else torch.tensor(0.0).type_as(x)
            loss.append(("CLBF unsafe region term", unsafe_V_term))
            if accuracy:
                unsafe_V_acc = (
                    unsafe_violation <= eps
                ).sum() / max(1, unsafe_violation.numel())
                loss.append(("CLBF unsafe region accuracy", unsafe_V_acc))

        return loss

    def descent_loss(
        self,
        x: torch.Tensor,
        accuracy: bool = False,
        requires_grad: bool = False,
    ) -> List[Tuple[str, torch.Tensor]]:
        loss = []
        eps = 0.1
        V = self.V(x)
        if self.barrier:
            condition_active = torch.sigmoid(10 * (self.safe_level + eps - V))
        else:
            condition_active = torch.tensor(1.0).type_as(x)

        u_qp, qp_relaxation = self.solve_CLF_QP(x, requires_grad=requires_grad)
        qp_relaxation = torch.mean(qp_relaxation, dim=-1)

        qp_relaxation_loss = (qp_relaxation * condition_active).mean()
        loss.append(("QP relaxation", qp_relaxation_loss))

        eps = 1.0
        clbf_descent_term_lin = torch.tensor(0.0).type_as(x)
        clbf_descent_acc_lin = torch.tensor(0.0).type_as(x)
        Lf_V, Lg_V = self.V_lie_derivatives(x)
        for i in range(self.n_scenarios):
            Vdot = Lf_V[:, i, :].squeeze(-1) + torch.bmm(
                Lg_V[:, i, :].unsqueeze(1),
                u_qp.unsqueeze(2),
            ).squeeze()
            violation = F.relu(eps + Vdot + self.clf_lambda * V)
            violation = violation * condition_active
            clbf_descent_term_lin = clbf_descent_term_lin + violation.mean()
            clbf_descent_acc_lin = clbf_descent_acc_lin + (violation <= eps).sum() / (
                violation.nelement() * self.n_scenarios
            )

        loss.append(("CLBF descent term (linearized)", clbf_descent_term_lin))
        if accuracy:
            loss.append(("CLBF descent accuracy (linearized)", clbf_descent_acc_lin))

        return loss

    def initial_loss(self, x: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
        loss = []
        epoch_count = max(self.current_epoch - self.num_init_epochs, 0)
        decrease_factor = 0.8 ** epoch_count
        V = self.V(x)
        P = self.dynamics_model.P.type_as(x)
        x0 = self.dynamics_model.goal_point.type_as(x)
        P_batched = P.unsqueeze(0).expand(x.shape[0], -1, -1)
        V_nominal = 0.5 * torch.bmm( (x-x0).unsqueeze(1), torch.bmm(P_batched, (x-x0).unsqueeze(2))).squeeze()

        if self.normalize_V_nominal:
            if self.training:
                self.V_nominal_mean = V_nominal.mean()
            if self.V_nominal_mean > 0:
                V_nominal /= self.V_nominal_mean

        clbf_mse_loss = F.mse_loss(V, V_nominal)
        clbf_mse_loss = decrease_factor * clbf_mse_loss
        loss.append(("CLBF MSE", clbf_mse_loss))
        return loss

    def training_step(self, batch, batch_idx):
        x, goal_mask, safe_mask, unsafe_mask = batch
        losses = []
        if self.current_epoch < self.num_init_epochs:
            losses.extend(self.initial_loss(x))
        else:
            losses.extend(self.boundary_loss(x, goal_mask, safe_mask, unsafe_mask))
            losses.extend(self.descent_loss(x, requires_grad=True))
        
        total_loss = sum(loss[1] for loss in losses)
        
        log_dict = {f"train_{name}": val for name, val in losses}
        self.log_dict(log_dict, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, goal_mask, safe_mask, unsafe_mask = batch
        losses = []
        if self.current_epoch < self.num_init_epochs:
            losses.extend(self.initial_loss(x))
        else:
            losses.extend(self.boundary_loss(x, goal_mask, safe_mask, unsafe_mask, accuracy=True))
            losses.extend(self.descent_loss(x, accuracy=True))
        
        total_loss = sum(loss[1] for loss in losses if torch.is_tensor(loss[1]))
        
        log_dict = {f"val_{name}": val for name, val in losses}
        self.log_dict(log_dict)
        self.log("val_loss", total_loss)
        
        return total_loss

    def on_validation_epoch_end(self):
        if self.current_epoch % 5 == 0:
            self.experiment_suite.run_all_and_log_plots(
                self, self.logger, self.current_epoch
            )

        if self.current_epoch > self.num_init_epochs and self.current_epoch % self.epochs_per_episode == 0:
            self.datamodule.add_data(self.u)

    def configure_optimizers(self):
        clbf_params = list(self.V_nn.parameters())
        clbf_opt = torch.optim.AdamW(
            clbf_params,
            lr=self.primal_learning_rate,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            clbf_opt,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
        )
        return {
            "optimizer": clbf_opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
