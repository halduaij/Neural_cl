"""
Comprehensive Transient Stability Validation Framework
=====================================================

Validates dimension reduction methods for power system transient stability analysis
with industry-standard metrics and acceptance criteria.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Callable
import time
from dataclasses import dataclass


@dataclass
class TransientStabilityMetrics:
    """Metrics for transient stability analysis validation."""
    # CCT metrics
    cct_accuracy: float  # Relative error in CCT estimation
    cct_values: Dict[str, float]  # CCT for different fault locations
    
    # Stability classification
    stability_accuracy: float  # Percentage of correct stability predictions
    false_stable: int  # Incorrectly classified as stable
    false_unstable: int  # Incorrectly classified as unstable
    
    # State tracking
    angle_rmse: float  # RMS error in rotor angles (degrees)
    angle_max_error: float  # Maximum angle error (degrees)
    frequency_rmse: float  # RMS error in frequency (Hz)
    frequency_max_error: float  # Maximum frequency error (Hz)
    
    # Physical quantities
    energy_error: float  # Relative error in energy conservation
    rocof_error: float  # Error in rate of change of frequency
    
    # Perturbation sensitivity
    perturbation_preservation: float  # Ratio of perturbation preserved
    
    # Computational performance
    computation_time: float  # Average time per simulation (seconds)
    speedup_factor: float  # Speedup vs full model
    
    # Overall assessment
    passed: bool  # Whether all criteria are met
    score: float  # Combined score (lower is better)


class TransientStabilityValidator:
    """
    Comprehensive validation framework for transient stability analysis.
    
    Implements industry-standard tests and metrics for validating
    dimension reduction methods in power system applications.
    """
    
    def __init__(
        self,
        system,
        acceptable_thresholds: Optional[Dict] = None
    ):
        """
        Initialize validator.
        
        Args:
            system: Power system object
            acceptable_thresholds: Dict of acceptable error thresholds
        """
        self.sys = system
        
        # Default thresholds based on industry standards
        self.thresholds = {
            'cct_error': 0.05,  # 5% CCT error
            'stability_accuracy': 0.98,  # 98% correct classification
            'angle_rmse': 1.0,  # 1 degree RMSE
            'frequency_rmse': 0.1,  # 0.1 Hz RMSE
            'energy_error': 0.05,  # 5% energy error
            'min_perturbation': 0.5,  # 50% perturbation preservation
            'max_computation_time': 0.1,  # 100ms
        }
        
        if acceptable_thresholds:
            self.thresholds.update(acceptable_thresholds)
    
    def validate_reducer(
        self,
        reducer,
        test_scenarios: Optional[List[Dict]] = None,
        verbose: bool = True
    ) -> TransientStabilityMetrics:
        """
        Perform comprehensive validation of a dimension reducer.
        
        Args:
            reducer: Dimension reduction object
            test_scenarios: List of test scenarios (generated if None)
            verbose: Print progress information
            
        Returns:
            TransientStabilityMetrics object with all results
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Validating {type(reducer).__name__} (d={reducer.latent_dim})")
            print(f"{'='*60}")
        
        # Generate test scenarios if not provided
        if test_scenarios is None:
            test_scenarios = self._generate_test_scenarios()
        
        # Initialize result storage
        results = {
            'cct_errors': [],
            'cct_values': {},
            'stability_predictions': [],
            'angle_errors': [],
            'frequency_errors': [],
            'energy_errors': [],
            'rocof_errors': [],
            'perturbation_ratios': [],
            'computation_times': [],
            'speedup_factors': []
        }
        
        # Run all tests
        for i, scenario in enumerate(test_scenarios):
            if verbose and i % 5 == 0:
                print(f"  Testing scenario {i+1}/{len(test_scenarios)}...")
            
            # 1. CCT Assessment
            self._test_cct(reducer, scenario, results)
            
            # 2. Stability Classification
            self._test_stability_classification(reducer, scenario, results)
            
            # 3. Trajectory Tracking
            self._test_trajectory_tracking(reducer, scenario, results)
            
            # 4. Energy Conservation
            self._test_energy_conservation(reducer, scenario, results)
            
            # 5. Perturbation Sensitivity
            self._test_perturbation_sensitivity(reducer, scenario, results)
            
            # 6. Computational Performance
            self._test_computational_performance(reducer, scenario, results)
        
        # Compute aggregate metrics
        metrics = self._compute_aggregate_metrics(results, verbose)
        
        return metrics
    
    def _generate_test_scenarios(self) -> List[Dict]:
        """Generate comprehensive test scenarios."""
        scenarios = []
        
        # Base operating conditions
        base_conditions = [
            {'load_level': 0.5, 'name': 'Light load'},
            {'load_level': 0.8, 'name': 'Normal load'},
            {'load_level': 1.0, 'name': 'Heavy load'},
        ]
        
        # Fault locations (different electrical distances)
        if hasattr(self.sys, 'N_NODES'):
            n_buses = self.sys.N_NODES
            fault_buses = [0, n_buses // 2, n_buses - 1]  # Near, middle, far
        else:
            fault_buses = [0]
        
        # Fault types and durations
        fault_durations = [0.05, 0.1, 0.15, 0.2, 0.25]  # 50ms to 250ms
        
        # Generate scenarios
        for condition in base_conditions:
            for fault_bus in fault_buses:
                for duration in fault_durations:
                    scenario = {
                        'name': f"{condition['name']}_Bus{fault_bus}_t{duration:.0f}ms",
                        'load_level': condition['load_level'],
                        'fault_bus': fault_bus,
                        'fault_duration': duration,
                        'fault_type': '3-phase',
                        'initial_state': self._get_initial_state(condition['load_level'])
                    }
                    scenarios.append(scenario)
        
        return scenarios
    
    def _get_initial_state(self, load_level: float):
        """Get initial state for given load level."""
        # Start from equilibrium
        x0 = self.sys.goal_point.clone()
        
        # Add small perturbation based on load level
        upper, lower = self.sys.state_limits
        perturbation = 0.02 * load_level * torch.randn_like(x0) * (upper - lower)
        
        return x0 + perturbation
    
    def _test_cct(self, reducer, scenario, results):
        """Test Critical Clearing Time estimation."""
        # Compute CCT with full model
        cct_full = self._compute_cct_full(scenario)
        
        # Compute CCT with reduced model
        cct_reduced = self._compute_cct_reduced(reducer, scenario)
        
        # Store results
        if cct_full > 0:
            error = abs(cct_reduced - cct_full) / cct_full
            results['cct_errors'].append(error)
            results['cct_values'][scenario['name']] = {
                'full': cct_full,
                'reduced': cct_reduced,
                'error': error
            }
    
    def _compute_cct_full(self, scenario, dt=0.001, max_time=0.5):
        """Compute CCT using full model (binary search)."""
        fault_bus = scenario['fault_bus']
        x0 = scenario['initial_state']
        
        # Binary search for CCT
        t_min, t_max = 0.0, max_time
        tolerance = 0.005  # 5ms precision
        
        while t_max - t_min > tolerance:
            t_clear = (t_min + t_max) / 2
            
            # Simulate with this clearing time
            stable = self._simulate_fault_full(x0, fault_bus, t_clear, dt)
            
            if stable:
                t_min = t_clear
            else:
                t_max = t_clear
        
        return (t_min + t_max) / 2
    
    def _compute_cct_reduced(self, reducer, scenario, dt=0.001, max_time=0.5):
        """Compute CCT using reduced model."""
        fault_bus = scenario['fault_bus']
        x0 = scenario['initial_state']
        
        # Binary search for CCT
        t_min, t_max = 0.0, max_time
        tolerance = 0.005
        
        while t_max - t_min > tolerance:
            t_clear = (t_min + t_max) / 2
            
            # Simulate with reduced model
            stable = self._simulate_fault_reduced(reducer, x0, fault_bus, t_clear, dt)
            
            if stable:
                t_min = t_clear
            else:
                t_max = t_clear
        
        return (t_min + t_max) / 2
    
    def _simulate_fault_full(self, x0, fault_bus, t_clear, dt, t_sim=2.0):
        """Simulate fault with full model and check stability."""
        x = x0.clone().unsqueeze(0) if x0.dim() == 1 else x0
        
        # Pre-fault
        for _ in range(int(0.1 / dt)):  # 100ms pre-fault
            f = self.sys._f(x, self.sys.nominal_params)
            x = x + dt * f.squeeze(-1)
        
        # During fault (simplified - zero injection at fault bus)
        for _ in range(int(t_clear / dt)):
            f = self.sys._f(x, self.sys.nominal_params)
            # Modify dynamics for fault
            if hasattr(self.sys, 'N_NODES') and fault_bus < self.sys.N_NODES:
                f[:, self.sys.N_NODES - 1 + fault_bus, 0] *= 0.1  # Severe fault
            x = x + dt * f.squeeze(-1)
        
        # Post-fault
        for _ in range(int((t_sim - t_clear) / dt)):
            f = self.sys._f(x, self.sys.nominal_params)
            x = x + dt * f.squeeze(-1)
            
            # Check stability (angle deviation)
            if hasattr(self.sys, 'N_NODES'):
                angles = x[:, :self.sys.N_NODES-1]
                if angles.abs().max() > np.pi:
                    return False
        
        return True
    
    def _simulate_fault_reduced(self, reducer, x0, fault_bus, t_clear, dt, t_sim=2.0):
        """Simulate fault with reduced model."""
        x = x0.clone().unsqueeze(0) if x0.dim() == 1 else x0
        z = reducer.forward(x)
        
        # Pre-fault
        for _ in range(int(0.1 / dt)):
            x_full = reducer.inverse(z)
            f_full = self.sys._f(x_full, self.sys.nominal_params)
            J = reducer.jacobian(x_full)
            f_reduced = torch.bmm(J, f_full).squeeze(-1)
            z = z + dt * f_reduced
        
        # During fault
        for _ in range(int(t_clear / dt)):
            x_full = reducer.inverse(z)
            f_full = self.sys._f(x_full, self.sys.nominal_params)
            
            # Modify for fault
            if hasattr(self.sys, 'N_NODES') and fault_bus < self.sys.N_NODES:
                f_full[:, self.sys.N_NODES - 1 + fault_bus, 0] *= 0.1
            
            J = reducer.jacobian(x_full)
            f_reduced = torch.bmm(J, f_full).squeeze(-1)
            z = z + dt * f_reduced
        
        # Post-fault
        for _ in range(int((t_sim - t_clear) / dt)):
            x_full = reducer.inverse(z)
            f_full = self.sys._f(x_full, self.sys.nominal_params)
            J = reducer.jacobian(x_full)
            f_reduced = torch.bmm(J, f_full).squeeze(-1)
            z = z + dt * f_reduced
            
            # Check stability
            if hasattr(self.sys, 'N_NODES'):
                angles = x_full[:, :self.sys.N_NODES-1]
                if angles.abs().max() > np.pi:
                    return False
        
        return True
    
    def _test_stability_classification(self, reducer, scenario, results):
        """Test stability classification accuracy."""
        # Test multiple clearing times around expected CCT
        base_cct = scenario['fault_duration']
        test_times = [0.8 * base_cct, 0.9 * base_cct, base_cct, 
                     1.1 * base_cct, 1.2 * base_cct]
        
        for t_clear in test_times:
            # Full model prediction
            stable_full = self._simulate_fault_full(
                scenario['initial_state'], 
                scenario['fault_bus'], 
                t_clear, 
                dt=0.001
            )
            
            # Reduced model prediction
            stable_reduced = self._simulate_fault_reduced(
                reducer,
                scenario['initial_state'], 
                scenario['fault_bus'], 
                t_clear, 
                dt=0.001
            )
            
            # Record result
            results['stability_predictions'].append({
                'true': stable_full,
                'predicted': stable_reduced,
                'correct': stable_full == stable_reduced
            })
    
    def _test_trajectory_tracking(self, reducer, scenario, results):
        """Test trajectory tracking accuracy."""
        x0 = scenario['initial_state']
        t_clear = scenario['fault_duration']
        dt = 0.01
        t_sim = 3.0
        
        # Simulate with full model
        traj_full = self._simulate_trajectory_full(x0, scenario['fault_bus'], t_clear, dt, t_sim)
        
        # Simulate with reduced model
        traj_reduced = self._simulate_trajectory_reduced(reducer, x0, scenario['fault_bus'], 
                                                        t_clear, dt, t_sim)
        
        # Compute errors
        if hasattr(self.sys, 'N_NODES'):
            # Angle errors (convert to degrees)
            angle_errors = (traj_full[:, :self.sys.N_NODES-1] - 
                          traj_reduced[:, :self.sys.N_NODES-1]) * 180 / np.pi
            
            angle_rmse = angle_errors.pow(2).mean().sqrt().item()
            angle_max = angle_errors.abs().max().item()
            
            results['angle_errors'].append({
                'rmse': angle_rmse,
                'max': angle_max
            })
            
            # Frequency errors (convert to Hz)
            freq_errors = (traj_full[:, self.sys.N_NODES-1:] - 
                         traj_reduced[:, self.sys.N_NODES-1:]) * 60 / (2 * np.pi)
            
            freq_rmse = freq_errors.pow(2).mean().sqrt().item()
            freq_max = freq_errors.abs().max().item()
            
            results['frequency_errors'].append({
                'rmse': freq_rmse,
                'max': freq_max
            })
    
    def _simulate_trajectory_full(self, x0, fault_bus, t_clear, dt, t_sim):
        """Simulate full trajectory with fault."""
        x = x0.clone().unsqueeze(0) if x0.dim() == 1 else x0
        trajectory = []
        
        t = 0
        while t < t_sim:
            trajectory.append(x.clone())
            
            # Compute dynamics
            f = self.sys._f(x, self.sys.nominal_params)
            
            # Apply fault if within fault duration
            if 0.1 <= t <= 0.1 + t_clear:
                if hasattr(self.sys, 'N_NODES') and fault_bus < self.sys.N_NODES:
                    f[:, self.sys.N_NODES - 1 + fault_bus, 0] *= 0.1
            
            # Update state
            x = x + dt * f.squeeze(-1)
            t += dt
        
        return torch.cat(trajectory, dim=0)
    
    def _simulate_trajectory_reduced(self, reducer, x0, fault_bus, t_clear, dt, t_sim):
        """Simulate trajectory with reduced model."""
        x = x0.clone().unsqueeze(0) if x0.dim() == 1 else x0
        z = reducer.forward(x)
        trajectory = []
        
        t = 0
        while t < t_sim:
            x_full = reducer.inverse(z)
            trajectory.append(x_full.clone())
            
            # Compute dynamics
            f_full = self.sys._f(x_full, self.sys.nominal_params)
            
            # Apply fault
            if 0.1 <= t <= 0.1 + t_clear:
                if hasattr(self.sys, 'N_NODES') and fault_bus < self.sys.N_NODES:
                    f_full[:, self.sys.N_NODES - 1 + fault_bus, 0] *= 0.1
            
            # Project dynamics
            J = reducer.jacobian(x_full)
            f_reduced = torch.bmm(J, f_full).squeeze(-1)
            
            # Update reduced state
            z = z + dt * f_reduced
            t += dt
        
        return torch.cat(trajectory, dim=0)
    
    def _test_energy_conservation(self, reducer, scenario, results):
        """Test energy conservation during transients."""
        x0 = scenario['initial_state']
        
        # Short trajectory after disturbance
        dt = 0.01
        n_steps = 100
        
        # Full model
        x_full = x0.clone().unsqueeze(0)
        energy_full = []
        
        for _ in range(n_steps):
            energy_full.append(self.sys.energy_function(x_full).item())
            f = self.sys._f(x_full, self.sys.nominal_params)
            x_full = x_full + dt * f.squeeze(-1)
        
        # Reduced model
        x_red = x0.clone().unsqueeze(0)
        z = reducer.forward(x_red)
        energy_reduced = []
        
        for _ in range(n_steps):
            x_red = reducer.inverse(z)
            energy_reduced.append(self.sys.energy_function(x_red).item())
            
            f_full = self.sys._f(x_red, self.sys.nominal_params)
            J = reducer.jacobian(x_red)
            f_reduced = torch.bmm(J, f_full).squeeze(-1)
            z = z + dt * f_reduced
        
        # Compute error
        energy_full = np.array(energy_full)
        energy_reduced = np.array(energy_reduced)
        
        rel_error = np.abs(energy_reduced - energy_full) / (np.abs(energy_full) + 1e-6)
        results['energy_errors'].append(rel_error.mean())
    
    def _test_perturbation_sensitivity(self, reducer, scenario, results):
        """Test preservation of perturbation response."""
        x0 = scenario['initial_state']
        
        # Create small perturbation
        perturbation = 0.01 * torch.randn_like(x0)
        x_pert = x0 + perturbation
        
        # Project both states
        z0 = reducer.forward(x0.unsqueeze(0))
        z_pert = reducer.forward(x_pert.unsqueeze(0))
        
        # Compute preservation ratio
        dx_norm = perturbation.norm()
        dz_norm = (z_pert - z0).norm()
        
        if dx_norm > 1e-6:
            ratio = (dz_norm / dx_norm).item()
            results['perturbation_ratios'].append(ratio)
    
    def _test_computational_performance(self, reducer, scenario, results):
        """Test computational efficiency."""
        x0 = scenario['initial_state']
        n_steps = 1000
        dt = 0.001
        
        # Time full model
        x_full = x0.clone().unsqueeze(0)
        t_start = time.time()
        
        for _ in range(n_steps):
            f = self.sys._f(x_full, self.sys.nominal_params)
            x_full = x_full + dt * f.squeeze(-1)
        
        t_full = time.time() - t_start
        
        # Time reduced model
        x_red = x0.clone().unsqueeze(0)
        z = reducer.forward(x_red)
        t_start = time.time()
        
        for _ in range(n_steps):
            x_red = reducer.inverse(z)
            f_full = self.sys._f(x_red, self.sys.nominal_params)
            J = reducer.jacobian(x_red)
            f_reduced = torch.bmm(J, f_full).squeeze(-1)
            z = z + dt * f_reduced
        
        t_reduced = time.time() - t_start
        
        results['computation_times'].append(t_reduced / n_steps)
        results['speedup_factors'].append(t_full / t_reduced)
    
    def _compute_aggregate_metrics(self, results, verbose):
        """Compute aggregate metrics from all tests."""
        # CCT accuracy
        cct_accuracy = np.mean(results['cct_errors']) if results['cct_errors'] else 1.0
        
        # Stability classification
        predictions = results['stability_predictions']
        if predictions:
            correct = sum(p['correct'] for p in predictions)
            total = len(predictions)
            stability_accuracy = correct / total
            
            # Count false positives/negatives
            false_stable = sum(1 for p in predictions 
                             if not p['true'] and p['predicted'])
            false_unstable = sum(1 for p in predictions 
                               if p['true'] and not p['predicted'])
        else:
            stability_accuracy = 0
            false_stable = 0
            false_unstable = 0
        
        # Angle/frequency tracking
        if results['angle_errors']:
            angle_rmse = np.mean([e['rmse'] for e in results['angle_errors']])
            angle_max = np.max([e['max'] for e in results['angle_errors']])
        else:
            angle_rmse = float('inf')
            angle_max = float('inf')
        
        if results['frequency_errors']:
            freq_rmse = np.mean([e['rmse'] for e in results['frequency_errors']])
            freq_max = np.max([e['max'] for e in results['frequency_errors']])
        else:
            freq_rmse = float('inf')
            freq_max = float('inf')
        
        # Other metrics
        energy_error = np.mean(results['energy_errors']) if results['energy_errors'] else 1.0
        perturbation = np.mean(results['perturbation_ratios']) if results['perturbation_ratios'] else 0
        comp_time = np.mean(results['computation_times']) if results['computation_times'] else 1.0
        speedup = np.mean(results['speedup_factors']) if results['speedup_factors'] else 1.0
        
        # Check pass/fail criteria
        passed = (
            cct_accuracy <= self.thresholds['cct_error'] and
            stability_accuracy >= self.thresholds['stability_accuracy'] and
            angle_rmse <= self.thresholds['angle_rmse'] and
            freq_rmse <= self.thresholds['frequency_rmse'] and
            energy_error <= self.thresholds['energy_error'] and
            perturbation >= self.thresholds['min_perturbation'] and
            comp_time <= self.thresholds['max_computation_time']
        )
        
        # Compute overall score (weighted sum of normalized errors)
        score = (
            10 * cct_accuracy +  # CCT is most important
            5 * (1 - stability_accuracy) +
            angle_rmse +
            5 * freq_rmse +
            10 * energy_error +
            2 * (1 - perturbation)
        )
        
        metrics = TransientStabilityMetrics(
            cct_accuracy=cct_accuracy,
            cct_values=results['cct_values'],
            stability_accuracy=stability_accuracy,
            false_stable=false_stable,
            false_unstable=false_unstable,
            angle_rmse=angle_rmse,
            angle_max_error=angle_max,
            frequency_rmse=freq_rmse,
            frequency_max_error=freq_max,
            energy_error=energy_error,
            rocof_error=0.0,  # Not implemented in this version
            perturbation_preservation=perturbation,
            computation_time=comp_time,
            speedup_factor=speedup,
            passed=passed,
            score=score
        )
        
        if verbose:
            self._print_results(metrics)
        
        return metrics
    
    def _print_results(self, metrics: TransientStabilityMetrics):
        """Print validation results."""
        print("\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)
        
        # CCT Results
        print(f"\n1. Critical Clearing Time:")
        print(f"   Average Error: {metrics.cct_accuracy:.1%}")
        print(f"   Acceptable: < {self.thresholds['cct_error']:.1%}")
        print(f"   {'✓ PASSED' if metrics.cct_accuracy <= self.thresholds['cct_error'] else '✗ FAILED'}")
        
        # Stability Classification
        print(f"\n2. Stability Classification:")
        print(f"   Accuracy: {metrics.stability_accuracy:.1%}")
        print(f"   False Stable: {metrics.false_stable}")
        print(f"   False Unstable: {metrics.false_unstable}")
        print(f"   Acceptable: > {self.thresholds['stability_accuracy']:.1%}")
        print(f"   {'✓ PASSED' if metrics.stability_accuracy >= self.thresholds['stability_accuracy'] else '✗ FAILED'}")
        
        # State Tracking
        print(f"\n3. State Tracking:")
        print(f"   Angle RMSE: {metrics.angle_rmse:.2f}°")
        print(f"   Angle Max Error: {metrics.angle_max_error:.2f}°")
        print(f"   Frequency RMSE: {metrics.frequency_rmse:.3f} Hz")
        print(f"   Frequency Max Error: {metrics.frequency_max_error:.3f} Hz")
        print(f"   Acceptable: < {self.thresholds['angle_rmse']}° angle, < {self.thresholds['frequency_rmse']} Hz freq")
        print(f"   {'✓ PASSED' if metrics.angle_rmse <= self.thresholds['angle_rmse'] and metrics.frequency_rmse <= self.thresholds['frequency_rmse'] else '✗ FAILED'}")
        
        # Physical Quantities
        print(f"\n4. Physical Conservation:")
        print(f"   Energy Error: {metrics.energy_error:.1%}")
        print(f"   Acceptable: < {self.thresholds['energy_error']:.1%}")
        print(f"   {'✓ PASSED' if metrics.energy_error <= self.thresholds['energy_error'] else '✗ FAILED'}")
        
        # Perturbation Sensitivity
        print(f"\n5. Perturbation Sensitivity:")
        print(f"   Preservation: {metrics.perturbation_preservation:.1%}")
        print(f"   Acceptable: > {self.thresholds['min_perturbation']:.1%}")
        print(f"   {'✓ PASSED' if metrics.perturbation_preservation >= self.thresholds['min_perturbation'] else '✗ FAILED'}")
        
        # Computational Performance
        print(f"\n6. Computational Performance:")
        print(f"   Time per step: {metrics.computation_time*1000:.1f} ms")
        print(f"   Speedup factor: {metrics.speedup_factor:.1f}x")
        print(f"   Acceptable: < {self.thresholds['max_computation_time']*1000:.0f} ms")
        print(f"   {'✓ PASSED' if metrics.computation_time <= self.thresholds['max_computation_time'] else '✗ FAILED'}")
        
        # Overall Assessment
        print(f"\n{'='*60}")
        print(f"OVERALL: {'✓ PASSED' if metrics.passed else '✗ FAILED'}")
        print(f"Score: {metrics.score:.2f} (lower is better)")
        print(f"{'='*60}\n")


def create_validation_plots(sys, reducers: Dict, test_scenario: Dict):
    """
    Create comprehensive validation plots for different reducers.
    
    Args:
        sys: Power system object
        reducers: Dict of reducer_name -> reducer object
        test_scenario: Single test scenario for visualization
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    # Setup
    x0 = test_scenario['initial_state']
    fault_bus = test_scenario['fault_bus']
    t_clear = test_scenario['fault_duration']
    dt = 0.01
    t_sim = 3.0
    
    # Simulate with full model
    print("Simulating full model for plots...")
    validator = TransientStabilityValidator(sys)
    traj_full = validator._simulate_trajectory_full(x0, fault_bus, t_clear, dt, t_sim)
    time_steps = np.arange(len(traj_full)) * dt
    
    # Store trajectories for all reducers
    trajectories = {'Full': traj_full}
    
    for name, reducer in reducers.items():
        print(f"Simulating {name}...")
        traj = validator._simulate_trajectory_reduced(reducer, x0, fault_bus, t_clear, dt, t_sim)
        trajectories[name] = traj
    
    # Plot 1: Angle of first generator
    ax = axes[0]
    for name, traj in trajectories.items():
        if hasattr(sys, 'N_NODES'):
            angle = traj[:, 0] * 180 / np.pi
            ax.plot(time_steps, angle, label=name, 
                   linewidth=2 if name == 'Full' else 1.5)
    ax.axvline(0.1, color='r', linestyle='--', alpha=0.5, label='Fault on')
    ax.axvline(0.1 + t_clear, color='g', linestyle='--', alpha=0.5, label='Fault cleared')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Generator 1-2 Angle Difference')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Frequency of first generator
    ax = axes[1]
    for name, traj in trajectories.items():
        if hasattr(sys, 'N_NODES'):
            freq = traj[:, sys.N_NODES-1] * 60 / (2 * np.pi)
            ax.plot(time_steps, freq, label=name,
                   linewidth=2 if name == 'Full' else 1.5)
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency Deviation (Hz)')
    ax.set_title('Generator 1 Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: System Energy
    ax = axes[2]
    for name, traj in trajectories.items():
        energy = []
        for i in range(len(traj)):
            e = sys.energy_function(traj[i:i+1])
            energy.append(e.item())
        ax.plot(time_steps, energy, label=name,
               linewidth=2 if name == 'Full' else 1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Total Energy')
    ax.set_title('System Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Angle Error
    ax = axes[3]
    for name, traj in trajectories.items():
        if name != 'Full' and hasattr(sys, 'N_NODES'):
            angle_error = (traj[:, :sys.N_NODES-1] - 
                         trajectories['Full'][:, :sys.N_NODES-1]) * 180 / np.pi
            rmse = angle_error.pow(2).mean(dim=1).sqrt()
            ax.plot(time_steps, rmse, label=f'{name} RMSE')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle RMSE (degrees)')
    ax.set_title('Angle Reconstruction Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Frequency Error
    ax = axes[4]
    for name, traj in trajectories.items():
        if name != 'Full' and hasattr(sys, 'N_NODES'):
            freq_error = (traj[:, sys.N_NODES-1:] - 
                        trajectories['Full'][:, sys.N_NODES-1:]) * 60 / (2 * np.pi)
            rmse = freq_error.pow(2).mean(dim=1).sqrt()
            ax.plot(time_steps, rmse, label=f'{name} RMSE')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency RMSE (Hz)')
    ax.set_title('Frequency Reconstruction Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Energy Error
    ax = axes[5]
    for name, traj in trajectories.items():
        if name != 'Full':
            energy_full = []
            energy_red = []
            for i in range(len(traj)):
                energy_full.append(sys.energy_function(trajectories['Full'][i:i+1]).item())
                energy_red.append(sys.energy_function(traj[i:i+1]).item())
            
            rel_error = 100 * np.abs(np.array(energy_red) - np.array(energy_full)) / \
                        (np.array(energy_full) + 1e-6)
            ax.plot(time_steps, rel_error, label=name)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy Error (%)')
    ax.set_title('Energy Conservation Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 7: Phase Portrait (Angle vs Frequency for one generator)
    ax = axes[6]
    for name, traj in trajectories.items():
        if hasattr(sys, 'N_NODES'):
            angle = traj[:, 0] * 180 / np.pi
            freq = traj[:, sys.N_NODES-1] * 60 / (2 * np.pi)
            ax.plot(angle, freq, label=name,
                   linewidth=2 if name == 'Full' else 1.5)
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Phase Portrait - Generator 1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 8: RoCoF
    ax = axes[7]
    for name, traj in trajectories.items():
        if hasattr(sys, 'N_NODES'):
            freq = traj[:, sys.N_NODES-1] * 60 / (2 * np.pi)
            rocof = np.gradient(freq, dt)
            ax.plot(time_steps[1:], rocof[1:], label=name,
                   linewidth=2 if name == 'Full' else 1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RoCoF (Hz/s)')
    ax.set_title('Rate of Change of Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 9: Reduction Quality Metrics
    ax = axes[8]
    metrics_names = []
    angle_errors = []
    freq_errors = []
    energy_errors = []
    
    for name, traj in trajectories.items():
        if name != 'Full':
            # Compute average errors
            if hasattr(sys, 'N_NODES'):
                angle_err = ((traj[:, :sys.N_NODES-1] - 
                            trajectories['Full'][:, :sys.N_NODES-1]) * 180/np.pi).abs().mean().item()
                freq_err = ((traj[:, sys.N_NODES-1:] - 
                           trajectories['Full'][:, sys.N_NODES-1:]) * 60/(2*np.pi)).abs().mean().item()
            else:
                angle_err = 0
                freq_err = 0
            
            energy_full = sys.energy_function(trajectories['Full'])
            energy_red = sys.energy_function(traj)
            energy_err = ((energy_red - energy_full) / energy_full).abs().mean().item()
            
            metrics_names.append(name)
            angle_errors.append(angle_err)
            freq_errors.append(freq_err)
            energy_errors.append(energy_err * 100)
    
    x = np.arange(len(metrics_names))
    width = 0.25
    
    ax.bar(x - width, angle_errors, width, label='Angle (deg)')
    ax.bar(x, freq_errors, width, label='Freq (Hz)')
    ax.bar(x + width, energy_errors, width, label='Energy (%)')
    
    ax.set_xlabel('Reducer')
    ax.set_ylabel('Average Error')
    ax.set_title('Reduction Quality Summary')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('transient_stability_validation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


# Example usage function
def run_comprehensive_validation(sys, reducers: Dict, n_scenarios: int = 10):
    """
    Run comprehensive validation on multiple reducers.
    
    Args:
        sys: Power system object
        reducers: Dict of name -> reducer object
        n_scenarios: Number of test scenarios
    """
    validator = TransientStabilityValidator(sys)
    
    # Run validation for each reducer
    results = {}
    for name, reducer in reducers.items():
        print(f"\nValidating {name}...")
        metrics = validator.validate_reducer(reducer, 
                                           test_scenarios=None,  # Use default scenarios
                                           verbose=True)
        results[name] = metrics
    
    # Summary comparison
    print("\n" + "="*80)
    print("COMPARATIVE SUMMARY")
    print("="*80)
    print(f"\n{'Reducer':<15} {'CCT Err':<10} {'Stab Acc':<10} {'Angle RMSE':<12} "
          f"{'Freq RMSE':<12} {'Energy Err':<12} {'Score':<10} {'Status':<10}")
    print("-"*109)
    
    for name, metrics in results.items():
        print(f"{name:<15} {metrics.cct_accuracy:<10.1%} {metrics.stability_accuracy:<10.1%} "
              f"{metrics.angle_rmse:<12.2f} {metrics.frequency_rmse:<12.3f} "
              f"{metrics.energy_error:<12.1%} {metrics.score:<10.2f} "
              f"{'PASS' if metrics.passed else 'FAIL':<10}")
    
    # Create visualization
    test_scenario = {
        'initial_state': sys.goal_point + 0.01 * torch.randn_like(sys.goal_point),
        'fault_bus': 0,
        'fault_duration': 0.15,
        'name': 'Visualization Test'
    }
    
    create_validation_plots(sys, reducers, test_scenario)
    
    return results