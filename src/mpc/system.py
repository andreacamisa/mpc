"""
TODO (acamisa): classes with similar structure as Cost but allow to get system matrices

(perhaps this is a hint that we should think of something more generic, which can be either
time invariant or time-varying and allows you to get the information at different time instants?)
"""


from abc import abstractmethod
from typing import Tuple

from nptyping import Float, NDArray, Shape


class Dynamics:
    @abstractmethod
    def get_matrices(
        self, time: int
    ) -> Tuple[NDArray[Shape["x, x"], Float], NDArray[Shape["x, u"], Float]]:
        pass


class TimeInvariantDynamics(Dynamics):
    def __init__(
        self,
        state_matrix: NDArray[Shape["x, x"], Float],
        input_matrix: NDArray[Shape["x, u"], Float],
    ) -> None:
        super().__init__()


class TimeVaryingDynamics(Dynamics):
    pass
