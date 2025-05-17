# Swing Equation Control with Neural CLBFs and Behavior Cloning

This project extends the Neural CLBF framework to control power systems modeled by the swing equation. It implements a three-stage approach:

1. First, train a Neural Control Lyapunov-Barrier Function (CLBF) controller for the swing equation system
2. Run a falsification module to identify potential issues in the Lyapunov function
3. Then, use behavior cloning to learn a policy that directly produces the same actions as the QP-based CLBF controller, with special attention to the counterexamples found through falsification

## Approach

- The Neural CLBF controller provides safety guarantees by ensuring that a level set of the learned Lyapunov function separates safe and unsafe regions
- The falsification module uses gradient-based optimization and adaptive sampling to find states where the Lyapunov decrease condition is violated
- The behavior cloning (BC) controller offers computational efficiency by avoiding online QP solving during deployment
- We maintain safety by regularizing the BC controller with the learned Lyapunov function and by emphasizing the counterexamples found during falsification

## Mathematical Formulation

### Swing Equation System

The power system with n nodes is modeled by the swing equation:

```
θ̇ᵢⱼ = ωᵢ - ωⱼ                                       (angle dynamics)
Mᵢω̇ᵢ = Pᵢ - Dᵢωᵢ - Σⱼ Kᵢⱼsin(θᵢⱼ) + uᵢ            (frequency dynamics)
```

where:
- θᵢⱼ is the angle difference between nodes i and j
- ωᵢ is the angular frequency of node i
- Mᵢ is the inertia constant
- Dᵢ is the damping coefficient
- Pᵢ is the mechanical power input
- Kᵢⱼ is the coupling strength between nodes i and j
- uᵢ is the control input (governor action)

### Neural CLBF Controller

The CLBF controller solves the following QP at each time step:

```
min_u,δ  ||u||² + ρδ²
s.t.     L_f V(x) + L_g V(x)u + λV(x) ≤ δ
         u_min ≤ u ≤ u_max
```

where:
- V(x) is the learned Lyapunov function
- L_f V and L_g V are the Lie derivatives of V along f and g
- λ is the convergence rate parameter
- ρ is the relaxation penalty
- δ is the relaxation variable

### Lyapunov Falsification

The falsification module attempts to find counterexamples where the Lyapunov decrease condition is violated:

```
Find x such that: L_f V(x) + L_g V(x)π(x) + λV(x) > 0
```

The module uses multiple approaches:
- Gradient-based optimization to maximize the violation
- Random sampling in the state space
- Adaptive sampling focusing on level set boundaries

### Behavior Cloning

The BC controller is trained with special emphasis on counterexamples from falsification:

```
L(θ) = E_x[ w(x)·||π_θ(x) - π_CLBF(x)||² + α·max(0, L_f V(x) + L_g V(x)π_θ(x) + λV(x))² ]
```

where:
- π_θ is the BC policy with parameters θ
- π_CLBF is the CLBF controller
- w(x) is a weighting function that emphasizes counterexamples
- α is the Lyapunov regularization weight
- The second term penalizes violations of the Lyapunov decrease condition

## Usage

### Training

To train the controllers with falsification, run:

```bash
python neural_clbf/training/train_swing_clbf_bc.py --train_clbf --run_falsification --train_bc --evaluate --eval_perturbations --eval_disturbances --falsify_bc
```

This will:
1. Train a Neural CLBF controller
2. Extract the learned Lyapunov function
3. Run falsification to find states where the Lyapunov decrease condition is violated
4. Train a BC controller with emphasis on the identified counterexamples
5. Evaluate and compare both controllers on nominal, perturbed, and disturbed trajectories
6. Run falsification on the trained BC controller to verify safety
7. Generate a comprehensive comparison table suitable for publication

For a sequential approach (train CLBF first, then run falsification, then train BC):

```bash
# Train CLBF
python neural_clbf/training/train_swing_clbf_bc.py --train_clbf

# Run falsification
python neural_clbf/training/train_swing_clbf_bc.py --run_falsification --force_falsification

# Train BC with falsification results
python neural_clbf/training/train_swing_clbf_bc.py --train_bc

# Evaluate
python neural_clbf/training/train_swing_clbf_bc.py --evaluate --eval_perturbations --eval_disturbances --falsify_bc
```

### Key Arguments

#### System Configuration:
- `--n_nodes`: Number of nodes in the swing equation system (default: 3)
- `--dt`: Timestep for simulation and control (default: 0.01)
- `--seed`: Random seed for reproducibility (default: 42)

#### CLBF Training:
- `--train_clbf`: Train CLBF controller even if a checkpoint exists
- `--clbf_epochs`: Maximum number of epochs for CLBF training (default: 100)
- `--clbf_hidden_layers`: Number of hidden layers in CLBF network (default: 3)
- `--clbf_hidden_size`: Neurons per hidden layer in CLBF network (default: 128)
- `--clbf_lr`: Learning rate for CLBF training (default: 1e-3)

#### Falsification:
- `--run_falsification`: Run Lyapunov falsification to find counterexamples
- `--force_falsification`: Force re-running falsification even if results exist
- `--falsification_samples`: Number of samples for falsification (default: 2000)
- `--falsification_iterations`: Maximum iterations for gradient falsification (default: 20)
- `--falsification_methods`: Falsification methods to use (default: "gradient,random,adaptive")
- `--falsification_weight`: Weight for falsification examples in BC training (default: 2.0)
- `--falsify_bc`: Run falsification on the BC controller after training

#### BC Training:
- `--train_bc`: Train BC controller even if a checkpoint exists
- `--bc_epochs`: Number of epochs for BC training (default: 200)
- `--bc_hidden_layers`: Hidden layers for BC network (default: "128,128,64")
- `--bc_lr`: Learning rate for BC training (default: 5e-4)
- `--lyapunov_weight`: Weight for Lyapunov regularization (default: 0.1)
- `--weight_decay`: L2 regularization parameter (default: 1e-4)
- `--grad_clip_value`: Maximum norm for gradient clipping (default: 1.0)
- `--validation_frequency`: Validate every N epochs (default: 5)

#### Evaluation:
- `--evaluate`: Run comprehensive evaluation after training
- `--eval_n_trajectories`: Number of trajectories for evaluation (default: 20)
- `--eval_perturbations`: Test with perturbed initial conditions
- `--eval_disturbances`: Test with external disturbances

### Visualization

To visualize the results after training and evaluation, run:

```bash
python neural_clbf/evaluation/visualize_swing_clbf_bc.py --plot_lyapunov --clbf_logdir=logs/swing_equation_clbf --bc_logdir=logs/swing_equation_bc
```

This generates:
- State trajectories and control inputs
- Phase portraits showing controller behavior
- Lyapunov function values and violation rates
- Statistical comparisons including convergence rates and control efforts
- Comparison visualizations across nominal, perturbed, and disturbed conditions

## Implementation Details

### SwingEquationSystem Class

The `SwingEquationSystem` class in `neural_clbf/systems/SwingEquationSystems.py` implements:
- Control-affine dynamics based on the swing equation
- State and control limits
- Scenario generation with parameter variations for robust training

### Neural CLBF Controller

Key features of the CLBF controller implementation:
- Neural network architecture with batch normalization and regularization
- Robust methods for computing Lie derivatives and solving QPs
- Boundary and descent losses for proper Lyapunov function training
- Adaptive penalty scheduling for improved convergence

### Lyapunov Falsification Module

The falsification module in `neural_clbf/training/lyapunov_falsification.py` implements:
- Multiple falsification strategies (gradient-based, random, adaptive)
- Visualization of discovered violations in state space
- Integration with the training pipeline to improve safety guarantees
- Counterexample generation for focused training of the BC controller
- Post-training verification to ensure safety guarantees

### Behavior Cloning Controller

The BC controller implements:
- Neural network with configurable architecture and regularization
- Lyapunov-based regularization for maintaining safety guarantees
- Weighted training that emphasizes counterexamples from falsification
- Adaptive weighting that increases regularization when violations occur
- Robust numerical techniques including gradient clipping and LR scheduling
- Comprehensive safety verification and validation

### Evaluation Framework

The evaluation framework provides:
- Comparison across nominal, perturbed, and disturbed conditions
- Computation of key metrics including success rate, control effort, and convergence
- Lyapunov-specific metrics to measure safety constraint satisfaction
- Statistical analysis suitable for academic publication

## Results and Analysis

After training and evaluation, the following artifacts are produced:

### Logs and Checkpoints
- CLBF controller checkpoints in `logs/swing_equation_clbf/checkpoints/`
- Falsification results in `logs/swing_equation_clbf/falsification/`
- BC controller model in `logs/swing_equation_bc/best_model.pt`
- Training metrics and curves in `logs/swing_equation_bc/training_metrics.pt`

### Falsification Results
- Counterexample states in `logs/swing_equation_clbf/falsification/counter_examples.pt`
- Violation values in `logs/swing_equation_clbf/falsification/violations.pt`
- Falsification statistics in `logs/swing_equation_clbf/falsification/falsification_stats.txt`
- Visualizations of violations in `logs/swing_equation_clbf/falsification/visualizations/`
- BC controller falsification results in `logs/swing_equation_bc/falsification/`

### Evaluation Results
- Trajectory data in `logs/swing_equation_bc/comparison_results.pt`
- Statistical analysis in `logs/swing_equation_bc/comparison_stats.pt`
- Publication-ready comparison table in `logs/swing_equation_bc/comparison_table.txt`
- Safety verification results in `logs/swing_equation_bc/safety_verification.txt`

### Visualizations
- Trajectory plots in `visualizations/trajectories/`
- Phase portraits in `visualizations/phase_portraits/`
- Statistical visualizations directly in `visualizations/`
- Lyapunov violation visualizations showing problematic regions in state space

## References

This implementation builds upon the following research:
1. "Neural Lyapunov Control" by Yin et al.
2. "Neural Control Barrier Functions for Safety-Critical Control" by Dawson et al.
3. "CLBF: A Control Lyapunov-Barrier Function Approach to Safe Reinforcement Learning" by Zhao et al.
4. "Falsification-Based Robust Adversarial Reinforcement Learning" by Uesato et al.
5. "Finding and Understanding Bugs in Neural Control Policies" by Tuncali et al.
6. "Falsification of Cyber-Physical Systems with Reinforcement Learning" by Akazaki et al.
7. "Imitating Latent Policies from Observation" by Torabi, Warnell, and Stone
8. "Safe Imitation Learning via Fast Bayesian Reward Inference from Preferences" by Brown et al.