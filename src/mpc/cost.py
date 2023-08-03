from abc import abstractmethod

from nptyping import Float, NDArray, Shape


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
    def get_state_cost(self, time: int) -> NDArray[Shape["x, x"], Float]:
        pass

    @abstractmethod
    def get_input_cost(self, time: int) -> NDArray[Shape["u, u"], Float]:
        pass

    @abstractmethod
    def get_terminal_cost(self) -> NDArray[Shape["x, x"], Float]:
        pass


class TimeInvariantCost(Cost):
    # TODO(acamisa): always use the same StageCost to return values
    def __init__(self, stage_cost: StageCost) -> None:
        super().__init__()


class TimeVaryingCost(Cost):
    pass  # TODO(acamisa): use different StageCosts to return values
