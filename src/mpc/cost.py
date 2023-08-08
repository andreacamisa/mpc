import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Iterator

from nptyping import Float, NDArray, Shape


@dataclass
class QuadraticStageCost:
    # TODO (acamisa): make S, q, r optional

    Q: NDArray[Shape["N, N"], Float]
    """Matrix for state quadratic term `x'Qx`."""

    R: NDArray[Shape["M, M"], Float]
    """Matrix for input quadratic term `u'Ru`."""

    S: NDArray[Shape["M, N"], Float]
    """Matrix for mixed quadratic term `u'Sx`."""

    q: NDArray[Shape["N"], Float]
    """Vector for state linear term `q'x`."""

    r: NDArray[Shape["M"], Float]
    """Vector for input linear term `r'u`."""


@dataclass
class QuadraticTerminalCost:
    Q: NDArray[Shape["N, N"], Float]
    """Matrix for quadratic term `x'Qx`."""

    q: NDArray[Shape["N"], Float]
    """Vector for linear term `q'x`."""


class Cost(ABC):
    r"""
    Cost function written as the sum of several stage costs, plus a terminal cost.

    This class models a cost function which is the sum of several stage costs at different
    time instants, plus an (optional) terminal cost:

    `l_0(x_0, u_0) + ... + l_T(x_{T-1}, u_{T-1}) + r(x_T)`

    where `l_t` is the t-th stage cost and `r` is the terminal cost.
    """

    @abstractmethod
    def get_stage_cost(self) -> Iterator[QuadraticStageCost]:
        """Get iterator for stage cost quadratic approximation.

        Returns an iterator where each element represents the quadratic approximation of the
        cost at each time instant, starting from time 0.
        """
        pass

    @abstractmethod
    def get_terminal_cost(self) -> QuadraticTerminalCost:
        """Get the terminal cost."""
        pass


class TimeInvariantCost(Cost):
    def __init__(
        self,
        stage_cost: QuadraticStageCost,
        terminal_cost: QuadraticTerminalCost,
    ) -> None:
        self._stage_cost = stage_cost
        self._terminal_cost = terminal_cost

    def get_stage_cost(self) -> Iterator[QuadraticStageCost]:
        return itertools.repeat(self._stage_cost)

    def get_terminal_cost(self) -> QuadraticTerminalCost:
        return self._terminal_cost


class TimeVaryingCost(Cost):
    def __init__(
        self,
        stage_costs: Iterable[QuadraticStageCost],
        terminal_cost: QuadraticTerminalCost,
    ) -> None:
        self._stage_costs = stage_costs
        self._terminal_cost = terminal_cost

    def get_stage_cost(self) -> Iterator[QuadraticStageCost]:
        return iter(self._stage_costs)

    def get_terminal_cost(self) -> QuadraticTerminalCost:
        return self._terminal_cost
