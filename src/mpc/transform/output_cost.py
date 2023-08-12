import dataclasses
from functools import partial
from typing import Tuple

from mpc.cost import QuadraticStageCost, QuadraticTerminalCost, TransformedCost
from mpc.problem import OptimalControlProblem
from mpc.transform.transform import IdentitySolutionTransform, ProblemTransform
from nptyping import Float, NDArray, Shape


class OutputCostTransform(ProblemTransform):
    """Problem transformation that represents having the output in the cost function.

    This transformation enables having the system output in the cost function rather than
    the system state. That is, if y is the system output, then applying this transformation
    is equivalent to assuming that the cost function terms are e.g. of the form y'Qy instead of
    x'Qx.
    """

    # TODO IVANO: ho aggiunto la D, bisogna riadattare i calcoli nelle funzioni _change_..
    # TODO ANDREA: aggiungere la C e la D alla dinamica

    def __init__(
        self,
        output_matrix: NDArray[Shape["P, N"], Float],
        output_feedthrough: NDArray[Shape["P, P"], Float],
    ) -> None:
        self._C = output_matrix
        self._D = output_feedthrough

    def apply(
        self, problem: OptimalControlProblem
    ) -> Tuple[OptimalControlProblem, IdentitySolutionTransform]:
        new_problem = dataclasses.replace(
            problem,
            cost=TransformedCost(
                problem.cost,
                partial(self._change_stage_cost, C=self._C, D=self._D),
                partial(self._change_terminal_cost, C=self._C, D=self._D),
            ),
        )
        return new_problem, IdentitySolutionTransform()

    @staticmethod
    def _change_stage_cost(
        cost: QuadraticStageCost,
        C: NDArray[Shape["P, N"], Float],
        D: NDArray[Shape["P, N"], Float],
    ) -> QuadraticStageCost:
        return QuadraticStageCost(
            Q=C.T @ cost.Q @ C,
            R=cost.R,
            S=cost.S @ C,
            q=cost.q @ C,
            r=cost.r,
        )

    @staticmethod
    def _change_terminal_cost(
        cost: QuadraticTerminalCost,
        C: NDArray[Shape["P, N"], Float],
        D: NDArray[Shape["P, N"], Float],
    ) -> QuadraticTerminalCost:
        return QuadraticTerminalCost(Q=C.T @ cost.Q @ C, q=cost.q @ C)
