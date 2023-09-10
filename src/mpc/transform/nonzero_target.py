import dataclasses
from typing import Tuple

from mpc.cost import StageCost, TerminalCost, TransformedCost
from mpc.problem import OptimalControlProblem
from mpc.transform.transform import IdentitySolutionTransform, ProblemTransform
from nptyping import Float, NDArray, Shape


class NonZeroTargetTransform(ProblemTransform):
    """Problem transformation to stabilize a state which is not the origin.

    This transformation represents stabilization of a state different than zero. That is,
    the resulting optimal control problem will make sure that a non-zero equilibrium
    is stabilized instead of the origin.
    """

    # TODO in futuro si può generalizzare questa classe ad una "TrackingTransform" per
    # stabilizzare traiettorie invece di equilibri

    def __init__(
        self,
        target_state: NDArray[Shape["*"], Float],
        target_input: NDArray[Shape["*"], Float],
    ) -> None:
        self._xref = target_state
        self._uref = target_input

    def apply(
        self, problem: OptimalControlProblem
    ) -> Tuple[OptimalControlProblem, IdentitySolutionTransform]:
        new_problem = dataclasses.replace(
            problem,
            cost=TransformedCost(
                problem.cost, self._change_stage_cost, self._change_terminal_cost
            ),
        )
        return new_problem, IdentitySolutionTransform()

    def _change_stage_cost(
        self,
        cost: StageCost,
    ) -> StageCost:
        q_aug = cost.q - 2 * cost.Q @ self._xref - 2 * cost.S.T @ self._uref
        r_aug = cost.r - 2 * cost.R @ self._uref - 2 * cost.S @ self._xref

        return StageCost(Q=cost.Q, R=cost.R, S=cost.S, q=q_aug, r=r_aug)

    def _change_terminal_cost(
        self,
        cost: TerminalCost,
    ) -> TerminalCost:
        q_aug = cost.q - 2 * cost.Q @ self._xref

        return TerminalCost(cost.Q, q=q_aug)
