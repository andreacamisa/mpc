from mpc.cost import QuadraticStageCost, QuadraticTerminalCost, TransformedCost
from mpc.problem import OptimalControlProblem
from mpc.solver import ProblemSolution
from mpc.system import LinearDynamics, TransformedSystem
from mpc.transform.transform import ProblemTransform
from nptyping import Float, NDArray, Shape


class IntegralActionTransform(ProblemTransform):
    """Transformation that adds an integral action.

    This transformation adds an integral action to an optimal control problem. An integral
    action consists of augmenting the state vector with new states that accumulate the output
    error over time. This quantity is then.... (che si fa con la z? si aggiunge alla u?)

    TODO spiegazione matematica piu' dettagliata
    """

    # TODO qui servirebbe la matrice C, per ora assumiamo C = I

    def transform_problem(
        self, problem: OptimalControlProblem
    ) -> OptimalControlProblem:
        return OptimalControlProblem(
            horizon=problem.horizon,
            cost=TransformedCost(
                problem.cost, self._change_stage_cost, self._change_terminal_cost
            ),
            system=TransformedSystem(problem.system, self._change_dynamics),
            initial_state=self._change_initial_state(problem.initial_state),
        )

    def _change_initial_state(
        self, state: NDArray[Shape["N"], Float]
    ) -> NDArray[Shape["N"], Float]:
        # TODO IVANO: qui si fanno le dovute manipolazioni allo stato iniziale prima di darlo al solver
        raise NotImplementedError()

    def _change_dynamics(self, dynamics: LinearDynamics) -> LinearDynamics:
        # TODO IVANO: qui si fanno le dovute manipolazioni alle matrici della dinamica prima di darle al solver
        # (la funzione viene chiamata separatamente per ogni istante di tempo t)
        raise NotImplementedError()

    def _change_stage_cost(self, cost: QuadraticStageCost) -> QuadraticStageCost:
        # TODO IVANO: qui si fanno le dovute manipolazioni alle matrici dello stage cost prima di darle al solver
        # (la funzione viene chiamata separatamente per ogni istante di tempo t)
        raise NotImplementedError()

    def _change_terminal_cost(
        self, cost: QuadraticTerminalCost
    ) -> QuadraticTerminalCost:
        # TODO IVANO: qui si fanno le dovute manipolazioni alle matrici del terminal cost prima di darle al solver
        raise NotImplementedError()

    def inverse_transform_solution(self, solution: ProblemSolution) -> ProblemSolution:
        # TODO IVANO: qui bisogna riadattare la traiettoria della soluzione in termini del problem originale
        raise NotImplementedError()
