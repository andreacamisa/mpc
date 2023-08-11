from mpc.cost import QuadraticStageCost, QuadraticTerminalCost, TransformedCost
from mpc.problem import OptimalControlProblem
from mpc.solver import ProblemSolution
from mpc.transform.transform import ProblemTransform
from nptyping import Float, NDArray, Shape


class NonZeroTargetTransform(ProblemTransform):
    """Transformation to stabilize a state which is not the origin.

    This transformation represents stabilization of a state different than zero. That is,
    the resulting optimal control problem will make sure that a non-zero equilibrium
    is stabilized instead of the origin.
    """

    # TODO in futuro si può generalizzare questa classe ad una "TrackingTransform" per
    # stabilizzare traiettorie invece di equilibri

    def __init__(self, target_state: NDArray[Shape["N"], Float]) -> None:
        self._target_state = target_state

    def transform_problem(
        self, problem: OptimalControlProblem
    ) -> OptimalControlProblem:
        return OptimalControlProblem(
            horizon=problem.horizon,
            cost=TransformedCost(
                problem.cost, self._change_stage_cost, self._change_terminal_cost
            ),
            system=problem.system,
            initial_state=problem.initial_state,
        )

    def _change_stage_cost(
        self, cost: QuadraticStageCost, xref: Float, uref: Float
    ) -> QuadraticStageCost:
        # IVANO: necessario passare anche (xref,uref)
        q_aug = cost.q - 2 * cost.Q @ xref - 2 * cost.S.T @ uref
        r_aug = cost.r - 2 * cost.R @ uref - 2 * cost.S @ xref

        return QuadraticStageCost(Q=cost.Q, R=cost.R, S=cost.S, q=q_aug, r=r_aug)

    def _change_terminal_cost(
        self, cost: QuadraticTerminalCost, xref: Float
    ) -> QuadraticTerminalCost:
        # IVANO: necessario passare anche xref
        q_aug = cost.q - 2 * cost.Q @ xref

        return QuadraticTerminalCost(cost.Q, q=q_aug)

    def inverse_transform_solution(self, solution: ProblemSolution) -> ProblemSolution:
        return solution
