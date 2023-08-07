from abc import abstractmethod

from nptyping import Float, NDArray, Shape


@dataclass
class StageCostMatrices:
    Q: NDArray
    R: NDArray
    S: NDArray
    q: NDArray
    r: NDArray


@dataclass
class TerminalCostMatrices:
    Q: NDArray
    q: NDArray


class StageCost:
    # TODO (acamisa): single item (x'Qx + u'Ru) of cost function at a time instant
    def __init__(
        self,
        state_matrix: NDArray[Shape["x, x"], Float],
        input_matrix: NDArray[Shape["u, u"], Float],
    ) -> None:
        pass


class Cost:
    @abstractmethod
    def get_stage_cost(self, time: int) -> StageCostMatrices:
        pass

    @abstractmethod
    def get_terminal_cost(self) -> TerminalCostMatrices:
        pass


class TimeInvariantCost(Cost):
    # TODO(acamisa): always use the same StageCost to return values
    def __init__(self, stage_cost: StageCost) -> None:
        super().__init__()


class TimeVaryingCost(Cost):
    pass  # TODO(acamisa): use different StageCosts to return values


class LinearizedSystemLQCost(Cost):
    def __init__(self, nonlinear_dynamics) -> None:
        super().__init__()

    pass
