from warnings import warn

from .control_affine_system import ControlAffineSystem
from .observable_system import ObservableSystem

from .SwingEquationSystem import IEEE39HybridSystem

__all__ = [
    "ControlAffineSystem",
    "ObservableSystem",
    "PlanarLidarSystem",
    "InvertedPendulum",
    "Quad2D",
    "Quad3D",
    "NeuralLander",
    "KSCar",
    "STCar",
    "TurtleBot",
    "TurtleBot2D",
    "Segway",
    "LinearSatellite",
    "SingleIntegrator2D",
    "AutoRally",
     "IEEE39HybridSystem"
]

try:
    from .f16 import F16  # noqa

    __all__.append("F16")
except ImportError:
    warn("Could not import F16 module; is AeroBench installed")
