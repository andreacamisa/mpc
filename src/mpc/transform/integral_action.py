import numpy as np
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
        # raise NotImplementedError()
        nx = state.shape[0]
        return np.concatenate(state, np.zeros((nx)))

    def _change_dynamics(self, dynamics: LinearDynamics) -> LinearDynamics:
        # TODO IVANO: qui si fanno le dovute manipolazioni alle matrici della dinamica prima di darle al solver
        # (la funzione viene chiamata separatamente per ogni istante di tempo t)
        # raise NotImplementedError()
        nx = dynamics.A.shape[0]
        nu = dynamics.B.shape[0]

        A_aug = np.block([[dynamics.A, np.zeros((nx, nx))], [np.eye(nx), np.eye(nx)]])
        B_aug = np.vstack((dynamics.B, np.zeros((nx, nu))))
        c_aug = np.vstack((dynamics.c, np.zeros((nx))))

        return LinearDynamics(A=A_aug, B=B_aug, c=c_aug)

    def _change_stage_cost(self, cost: QuadraticStageCost) -> QuadraticStageCost:
        # raise NotImplementedError()
        #
        # do NOT weight the integrator state
        nx = cost.Q.shape[0]
        nu = cost.R.shape[0]

        Q_aug = np.block(
            [[cost.Q, np.zeros((nx, nx))], [np.zeros((nx, nx)), np.zeros((nx, nx))]]
        )

        S_aug = np.block([[cost.S, np.zeros((nu, nx))]])

        q_aug = np.concatenate(cost.q, np.zeros((nx)))

        return QuadraticStageCost(Q=Q_aug, R=cost.R, S=S_aug, q=q_aug, r=cost.r)

    def _change_terminal_cost(
        self, cost: QuadraticTerminalCost
    ) -> QuadraticTerminalCost:
        # raise NotImplementedError()
        nx = cost.Q.shape[0]

        Q_aug = np.block(
            [[cost.Q, np.zeros((nx, nx))], [np.zeros((nx, nx)), np.zeros((nx, nx))]]
        )
        q_aug = np.concatenate(cost.q, np.zeros((nx)))

        return QuadraticTerminalCost(Q=Q_aug, q=q_aug)

    def inverse_transform_solution(
        self, solution: ProblemSolution, nx: int
    ) -> ProblemSolution:
        # IVANO: serve la dimensione di x originale
        state_traj = solution.state_traj[:nx]

        return ProblemSolution(state_traj=state_traj, input_traj=solution.input_traj)
