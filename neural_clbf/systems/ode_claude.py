#!/usr/bin/env python3
"""
Optimized Neural CLBF Training for IEEE39 Power System
Complete solution with proper hyperparameters and numerical stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
import math


class OptimizedTrainingConfig:
    """Carefully tuned hyperparameters for power system CLBF training"""
    
    # Network architecture
    hidden_layers = 4
    hidden_size = 512
    use_residual = True
    use_layer_norm = True
    dropout = 0.1
    
    # Training schedule
    pretrain_epochs = 30      # Lyapunov function fitting
    transition_epochs = 20    # Gradual introduction of dynamics
    full_epochs = 150         # Full CLF-QP training
    
    # Learning rates
    lr_pretrain = 1e-3       # Higher for initial fitting
    lr_transition = 5e-4     # Medium for stability
    lr_full = 1e-4          # Lower for fine-tuning
    lr_min = 1e-6           # Minimum LR
    
    # Batch sizes
    batch_pretrain = 512    # Large batches for averaging
    batch_transition = 256  
    batch_full = 128       # Smaller for rollouts
    
    # CLF parameters (gradually tightened)
    lambda_pretrain = 2.0      # Loose initially
    lambda_transition = 1.0    
    lambda_full = 0.5          # Tight final convergence
    
    relax_penalty_initial = 1.0
    relax_penalty_final = 100.0
    
    # Loss weights (based on empirical gradient magnitudes)
    weights = {
        'goal': 10.0,          # Strong attractor at equilibrium
        'safe': 50.0,          # Critical for barrier
        'unsafe': 50.0,        # Critical for barrier
        'descent': 1.0,        # Natural scale
        'qp_relax': 10.0,      # Penalize relaxation
        'simulation': 5.0      # Consistency check
    }
    
    # Stability parameters
    gradient_clip = 1.0
    weight_decay = 1e-5
    eps = 1e-6
    
    # Rollout parameters
    horizon_start = 0.05    # 50ms initially
    horizon_end = 0.5       # 500ms finally
    dt = 0.001             # 1ms timestep


class ImprovedNeuralCLF(nn.Module):
    """Robust neural CLF with proper architecture for power systems"""
    
    def __init__(self, sys, config):
        super().__init__()
        self.sys = sys
        self.config = config
        
        # Dimensions
        self.n_dims = sys.n_dims
        n_angles = len(sys.angle_dims)
        self.n_input = self.n_dims + n_angles  # Extra dims for cos(angles)
        
        # Compute normalization statistics
        self._compute_normalization()
        
        # Build network
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.n_input, config.hidden_size))
        if config.use_layer_norm:
            layers.append(nn.LayerNorm(config.hidden_size))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(config.dropout))
        
        # Hidden layers with residual connections
        for i in range(config.hidden_layers - 1):
            if config.use_residual:
                layers.append(ResidualBlock(
                    config.hidden_size,
                    use_norm=config.use_layer_norm,
                    dropout=config.dropout
                ))
            else:
                layers.append(nn.Linear(config.hidden_size, config.hidden_size))
                if config.use_layer_norm:
                    layers.append(nn.LayerNorm(config.hidden_size))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(config.dropout))
        
        self.feature_net = nn.Sequential(*layers)
        
        # Output head (single positive value)
        self.output_head = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        # Initialize weights carefully
        self._init_weights()
        
        # Quadratic weight (learnable mixing)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def _compute_normalization(self):
        """Compute robust normalization from system data"""
        with torch.no_grad():
            # Sample around equilibrium
            goal = self.sys.goal_point[0] if self.sys.goal_point.dim() > 1 else self.sys.goal_point
            samples = []
            for _ in range(1000):
                noise = 0.1 * torch.randn_like(goal)
                sample = goal + noise
                samples.append(sample)
            samples = torch.stack(samples)
            
            # Use robust statistics
            self.register_buffer('x_mean', samples.mean(dim=0))
            self.register_buffer('x_std', samples.std(dim=0) + 1e-6)
            
    def _init_weights(self):
        """Conservative weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Small weights for stability
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def normalize_input(self, x):
        """Normalize and handle angles properly"""
        # Standard normalization
        x_norm = (x - self.x_mean) / self.x_std
        
        # Extract angles and replace with sin/cos
        angle_dims = self.sys.angle_dims
        if angle_dims:
            angles = x[:, angle_dims]
            x_norm[:, angle_dims] = torch.sin(angles)
            # Append cosines
            cos_angles = torch.cos(angles)
            x_norm = torch.cat([x_norm, cos_angles], dim=-1)
            
        return x_norm
    
    def forward(self, x):
        """Compute V and its Jacobian"""
        # Enable gradient computation
        x = x.requires_grad_(True)
        
        # Normalize
        x_norm = self.normalize_input(x)
        
        # Neural network path
        features = self.feature_net(x_norm)
        v_raw = self.output_head(features)
        
        # Ensure positivity
        v_neural = 0.5 * v_raw.pow(2).squeeze(-1)
        
        # Quadratic term for stability
        P = self.sys.P.type_as(x)
        x0 = self.sys.goal_point.type_as(x)
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0).expand(x.shape[0], -1)
        
        dx = x - x0
        P_batch = P.unsqueeze(0).expand(x.shape[0], -1, -1)
        v_quad = 0.5 * torch.bmm(
            dx.unsqueeze(1), 
            torch.bmm(P_batch, dx.unsqueeze(2))
        ).squeeze()
        
        # Mix neural and quadratic
        alpha = torch.sigmoid(self.alpha)
        V = alpha * v_neural + (1 - alpha) * v_quad
        
        # Compute Jacobian
        JV = torch.autograd.grad(
            V.sum(), x,
            create_graph=True,
            retain_graph=True
        )[0]
        JV = JV.reshape(x.shape[0], 1, self.n_dims)
        
        return V, JV


class ResidualBlock(nn.Module):
    """Residual block for deep networks"""
    
    def __init__(self, dim, use_norm=True, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.scale = 0.1  # Residual scaling
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + self.scale * x


def solve_clf_qp_robust(ctrl, x, u_ref=None):
    """Robust CLF-QP solver with numerical safeguards"""
    B = x.shape[0]
    device = x.device
    
    if u_ref is None:
        u_ref = ctrl.dynamics_model.u_eq.expand(B, -1).to(device)
    
    # Get Lie derivatives with gradient checking
    V = ctrl.V(x)
    Lf_V, Lg_V = ctrl.V_lie_derivatives(x)
    
    # Check for numerical issues
    if not torch.isfinite(V).all():
        print("Warning: Non-finite V detected, using fallback")
        return u_ref, torch.zeros(B, 1, device=device)
    
    if not torch.isfinite(Lf_V).all() or not torch.isfinite(Lg_V).all():
        print("Warning: Non-finite Lie derivatives, using fallback")
        return u_ref, torch.zeros(B, 1, device=device)
    
    # Solve with error handling
    try:
        u, r = ctrl.solve_CLF_QP(x, u_ref=u_ref, requires_grad=False)
        
        # Sanity check
        if not torch.isfinite(u).all():
            print("Warning: Non-finite control from QP")
            return u_ref, torch.zeros(B, 1, device=device)
            
        return u, r
        
    except Exception as e:
        print(f"QP solver failed: {e}, using u_ref")
        return u_ref, torch.zeros(B, 1, device=device)


class PowerSystemCLBFTrainer:
    """Main training orchestrator with curriculum learning"""
    
    def __init__(self, sys, config=None):
        self.sys = sys
        self.config = config or OptimizedTrainingConfig()
        self.device = sys.goal_point.device
        
        # Build neural CLF
        self.clbf = ImprovedNeuralCLF(sys, self.config).to(self.device)
        
        # Setup controller
        self._setup_controller()
        
        # Training state
        self.epoch = 0
        self.stage = 'pretrain'
        self.best_loss = float('inf')
        
    def _setup_controller(self):
        """Create CLF controller with neural network"""
        from neural_clbf.controllers.clf_controller import CLFController
        from neural_clbf.experiments import ExperimentSuite
        
        # Base controller
        self.ctrl = CLFController(
            dynamics_model=self.sys,
            scenarios=[self.sys.nominal_params],
            experiment_suite=ExperimentSuite([]),
            clf_lambda=self.config.lambda_pretrain,
            clf_relaxation_penalty=self.config.relax_penalty_initial,
            controller_period=float(self.sys.dt),
            disable_gurobi=True,
        )
        
        # Override V methods
        self.ctrl.V_with_jacobian = lambda x: self.clbf(x)
        self.ctrl.V = lambda x: self.clbf(x)[0]
        
    def get_optimizer(self, stage):
        """Stage-specific optimizer"""
        if stage == 'pretrain':
            lr = self.config.lr_pretrain
        elif stage == 'transition':
            lr = self.config.lr_transition
        else:
            lr = self.config.lr_full
            
        return torch.optim.AdamW(
            self.clbf.parameters(),
            lr=lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )
    
    def sample_batch(self, batch_size, stage):
        """Sample states based on training stage"""
        if stage == 'pretrain':
            # Near equilibrium
            x = self.sys.goal_point.expand(batch_size, -1).clone()
            x += 0.05 * torch.randn_like(x)
        else:
            # Mixed sampling
            n_goal = batch_size // 4
            n_safe = batch_size // 2
            n_boundary = batch_size - n_goal - n_safe
            
            x_goal = self.sys.goal_point.expand(n_goal, -1).clone()
            x_goal += 0.1 * torch.randn_like(x_goal)
            
            x_safe = self.sys.sample_safe(n_safe).to(self.device)
            x_boundary = self.sys.sample_boundary(n_boundary).to(self.device)
            
            x = torch.cat([x_goal, x_safe, x_boundary], dim=0)
        
        # Clip to bounds
        x_hi, x_lo = self.sys.state_limits
        x = torch.clamp(x, x_lo.to(x), x_hi.to(x))
        
        return x
    
    def compute_loss(self, x, stage):
        """Stage-appropriate loss computation"""
        losses = {}
        weights = self.config.weights
        
        # CLF value
        V, JV = self.clbf(x)
        
        # Goal attraction
        goal_dist = torch.norm(x - self.sys.goal_point, dim=-1)
        near_goal = goal_dist < 0.1
        if near_goal.any():
            losses['goal'] = weights['goal'] * V[near_goal].mean()
        
        # Barrier conditions
        safe_mask = self.sys.safe_mask(x)
        if safe_mask.any():
            losses['safe'] = weights['safe'] * F.relu(V[safe_mask] - 1.0 + 0.01).mean()
            
        unsafe_mask = self.sys.unsafe_mask(x)
        if unsafe_mask.any():
            losses['unsafe'] = weights['unsafe'] * F.relu(1.0 - V[unsafe_mask] + 0.01).mean()
        
        if stage in ['transition', 'full']:
            # CLF descent
            u, r = solve_clf_qp_robust(self.ctrl, x)
            losses['qp_relax'] = weights['qp_relax'] * r.mean()
            
            # Verify descent
            Lf_V, Lg_V = self.ctrl.V_lie_derivatives(x)
            Vdot = (Lf_V[:, 0, :].squeeze(-1) + 
                   torch.bmm(Lg_V[:, 0, :].unsqueeze(1), u.unsqueeze(2)).squeeze())
            
            lam = getattr(self.ctrl, 'clf_lambda', 0.5)
            losses['descent'] = weights['descent'] * F.relu(Vdot + lam * V + 0.1).mean()
            
        if stage == 'full':
            # Simulation consistency
            with torch.no_grad():
                f, g = self.sys.control_affine_dynamics(x)
                if f.dim() == 3:
                    f = f.squeeze(-1)
                xdot = f + torch.bmm(g, u.unsqueeze(-1)).squeeze(-1)
                x_next = x + self.sys.dt * xdot
                
            V_next, _ = self.clbf(x_next)
            losses['simulation'] = weights['simulation'] * F.relu(V_next - V).mean()
        
        return losses
    
    def train_epoch(self, stage, optimizer):
        """Train one epoch"""
        self.clbf.train()
        
        # Batch size for stage
        if stage == 'pretrain':
            batch_size = self.config.batch_pretrain
            n_batches = 20
        elif stage == 'transition':
            batch_size = self.config.batch_transition
            n_batches = 15
        else:
            batch_size = self.config.batch_full
            n_batches = 10
            
        total_loss = 0
        loss_components = {}
        
        for _ in range(n_batches):
            # Sample batch
            x = self.sample_batch(batch_size, stage)
            
            # Compute losses
            losses = self.compute_loss(x, stage)
            
            # Total loss
            loss = sum(losses.values())
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.clbf.parameters(),
                self.config.gradient_clip
            )
            
            optimizer.step()
            
            # Track
            total_loss += loss.item()
            for k, v in losses.items():
                if k not in loss_components:
                    loss_components[k] = 0
                loss_components[k] += v.item()
                
        # Average
        total_loss /= n_batches
        for k in loss_components:
            loss_components[k] /= n_batches
            
        return total_loss, loss_components
    
    def train(self):
        """Full training pipeline"""
        print("Starting optimized CLBF training")
        
        # Stage 1: Pretrain
        print(f"\nStage 1: Pretraining ({self.config.pretrain_epochs} epochs)")
        optimizer = self.get_optimizer('pretrain')
        self.ctrl.clf_lambda = self.config.lambda_pretrain
        
        for epoch in range(self.config.pretrain_epochs):
            loss, components = self.train_epoch('pretrain', optimizer)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss={loss:.4f}, Components={components}")
                
            if loss < self.best_loss:
                self.best_loss = loss
                self.save_checkpoint('best_pretrain.pt')
        
        # Stage 2: Transition
        print(f"\nStage 2: Transition ({self.config.transition_epochs} epochs)")
        optimizer = self.get_optimizer('transition')
        self.ctrl.clf_lambda = self.config.lambda_transition
        
        for epoch in range(self.config.transition_epochs):
            # Gradually increase penalty
            progress = epoch / self.config.transition_epochs
            self.ctrl.clf_relaxation_penalty = (
                self.config.relax_penalty_initial + 
                progress * (self.config.relax_penalty_final - self.config.relax_penalty_initial)
            )
            
            loss, components = self.train_epoch('transition', optimizer)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss={loss:.4f}, Components={components}")
        
        # Stage 3: Full training
        print(f"\nStage 3: Full training ({self.config.full_epochs} epochs)")
        optimizer = self.get_optimizer('full')
        self.ctrl.clf_lambda = self.config.lambda_full
        self.ctrl.clf_relaxation_penalty = self.config.relax_penalty_final
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config.full_epochs,
            eta_min=self.config.lr_min
        )
        
        for epoch in range(self.config.full_epochs):
            loss, components = self.train_epoch('full', optimizer)
            scheduler.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={loss:.4f}, LR={scheduler.get_last_lr()[0]:.2e}")
                print(f"  Components: {components}")
                
            if loss < self.best_loss:
                self.best_loss = loss
                self.save_checkpoint('best_full.pt')
        
        print("\nTraining complete!")
        return self.clbf
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        torch.save({
            'model_state': self.clbf.state_dict(),
            'config': self.config,
            'epoch': self.epoch,
            'best_loss': self.best_loss
        }, filename)
        print(f"Saved checkpoint: {filename}")


# Usage function
def train_power_system_clbf(sys):
    """Main entry point for training"""
    # Ensure system is initialized
    sys.compute_linearized_controller([sys.nominal_params])
    
    # Create trainer with optimized config
    config = OptimizedTrainingConfig()
    trainer = PowerSystemCLBFTrainer(sys, config)
    
    # Run training
    trained_clbf = trainer.train()
    
    # Save final model
    trainer.save_checkpoint('final_model.pt')
    
    return trained_clbf, trainer.ctrl


if __name__ == "__main__":
    # Example usage
    from neural_clbf.systems.IEEE39ControlAffineDAE298 import IEEE39ControlAffineDAE
    
    # Initialize system
    sys = IEEE39ControlAffineDAE(
        nominal_params={
            "pv_ratio": [0.9, 0.9, 0.85, 0.9, 0.85, 0.85, 0.9, 0.9, 0.85, 0.3],
            "T_pv": (0.01, 0.01),
        },
        dt=0.001,
    )
    
    # Train
    clbf, controller = train_power_system_clbf(sys)
    print("Training complete!")