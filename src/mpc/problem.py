from dataclasses import dataclass

from mpc.cost import Cost
from mpc.system import System
from nptyping import Float, NDArray, Shape


@dataclass(frozen=True)
class OptimalControlProblem:
    horizon: int
    cost: Cost
    system: System
    initial_state: NDArray[Shape["N"], Float]
