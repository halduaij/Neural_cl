import torch
import torch.nn as nn
import pytorch_lightning as pl


class BehaviorCloningPolicy(pl.LightningModule):
    """Simple MLP policy trained with behavior cloning."""

    def __init__(self, n_dims: int, n_controls: int, hidden_size: int = 64, hidden_layers: int = 2, learning_rate: float = 1e-3) -> None:
        super().__init__()
        layers = []
        in_size = n_dims
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, n_controls))
        self.network = nn.Sequential(*layers)
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, u = batch
        u_hat = self(x)
        loss = nn.functional.mse_loss(u_hat, u)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, u = batch
        u_hat = self(x)
        loss = nn.functional.mse_loss(u_hat, u)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
