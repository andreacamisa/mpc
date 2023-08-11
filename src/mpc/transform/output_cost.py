from mpc.cost import QuadraticStageCost, QuadraticTerminalCost, TransformedCost
from mpc.problem import OptimalControlProblem
from mpc.solver import ProblemSolution
from mpc.transform.transform import ProblemTransform
from nptyping import Float, NDArray, Shape


class OutputCostTransform(ProblemTransform):
    """Transformation that represents having the output in the cost function.

    This transformation enables having the system output in the cost function rather than
    the system state. That is, if y is the system output, then applying this transformation
    is equivalent to assuming that the cost function terms are e.g. of the form y'Qy instead of
    x'Qx.
    """

    def __init__(self, output_matrix: NDArray[Shape["P, N"], Float]) -> None:
        self._C = output_matrix

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

    def _change_stage_cost(self, cost: QuadraticStageCost) -> QuadraticStageCost:
        # Cosi?
        # (la funzione viene chiamata separatamente per ogni istante di tempo t)

        # qui bisogna immaginare e.g. che cost.Q sia la matrice applicata alla y invece che alla x
        return QuadraticStageCost(
            Q=self._C.T @ cost.Q @ self._C,
            R=cost.R,
            S=cost.S @ self._C,
            q=cost.q @ self._C,
            r=cost.r,
        )

    def _change_terminal_cost(
        self, cost: QuadraticTerminalCost
    ) -> QuadraticTerminalCost:
        # TODO IVANO: qui si fanno le dovute manipolazioni alle matrici del terminal cost prima di darle al solver
        return QuadraticStageCost(Q=self._C.T @ cost.Q @ self._C, q=cost.q @ self._C)

    def inverse_transform_solution(self, solution: ProblemSolution) -> ProblemSolution:
        return solution
