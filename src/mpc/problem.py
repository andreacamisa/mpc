from dataclasses import dataclass

from mpc.cost import Cost
from mpc.system import System
from nptyping import Float, NDArray, Shape


@dataclass(frozen=True)
class OptimalControlProblem:
    """Finite-horizon optimal control problem composed of cost function,
    system dynamics and initial state."""

    horizon: int
    """Prediction horizon of optimal control problem."""

    cost: Cost
    """Cost function appearing of optimal control problem."""

    system: System
    """System dynamics."""

    initial_state: NDArray[Shape["N"], Float]
    """Initial state of the system."""
