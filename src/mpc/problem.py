from dataclasses import dataclass

from mpc.cost import Cost
from mpc.system import Dynamics
from nptyping import Float, NDArray, Shape


@dataclass(frozen=True)
class OptimalControlProblem:
    horizon: int
    cost: Cost
    dynamics: Dynamics
    initial_state: NDArray[Shape["*"], Float]
