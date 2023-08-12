from functools import partial
from typing import Tuple

import numpy as np
from mpc.cost import QuadraticStageCost, QuadraticTerminalCost, TransformedCost
from mpc.problem import OptimalControlProblem
from mpc.system import LinearDynamics, TransformedSystem
from mpc.transform.transform import PartialStateSolutionTransform, ProblemTransform
from nptyping import Float, NDArray, Shape


class IntegralActionTransform(ProblemTransform):
    """Transformation that adds an integral action.

    This transformation adds an integral action to an optimal control problem. An integral
    action consists of augmenting the state vector with new states that accumulate the output
    error over time. This quantity is then.... (che si fa con la z? si aggiunge alla u?)

    TODO spiegazione matematica piu' dettagliata
    """

    # TODO IVANO: ho aggiunto la C e la D, bisogna riadattare i calcoli nelle funzioni _change_..
    # TODO ANDREA: aggiungere la C e la D alla dinamica
    # TODO ANDREA: passare yref da usare nel calcolo dell'errore sull'output

    def __init__(
        self,
        output_matrix: NDArray[Shape["P, N"], Float],
        output_feedthrough: NDArray[Shape["P, P"], Float],
    ) -> None:
        self._C = output_matrix
        self._D = output_feedthrough

    def apply(
        self, problem: OptimalControlProblem
    ) -> Tuple[OptimalControlProblem, PartialStateSolutionTransform]:
        output_dim = self._C.shape[0]
        new_state_dim = problem.system.state_dim + output_dim
        new_problem = OptimalControlProblem(
            horizon=problem.horizon,
            cost=TransformedCost(
                problem.cost,
                partial(self._change_stage_cost, output_dim=output_dim),
                partial(self._change_terminal_cost, output_dim=output_dim),
            ),
            system=TransformedSystem(
                problem.system,
                partial(self._change_dynamics, C=self._C, D=self._D),
                new_state_dim,
            ),
            initial_state=self._change_initial_state(problem.initial_state, output_dim),
        )
        solution_transform = PartialStateSolutionTransform(problem.system.state_dim)
        return new_problem, solution_transform

    @staticmethod
    def _change_initial_state(
        state: NDArray[Shape["*"], Float], output_dim: int
    ) -> NDArray[Shape["*"], Float]:
        return np.concatenate(state, np.zeros(output_dim))

    @staticmethod
    def _change_dynamics(
        dynamics: LinearDynamics,
        C: NDArray[Shape["P, N"], Float],
        D: NDArray[Shape["P, N"], Float],
    ) -> LinearDynamics:
        nx = dynamics.A.shape[0]
        nu = dynamics.B.shape[0]

        A_aug = np.block([[dynamics.A, np.zeros((nx, nx))], [np.eye(nx), np.eye(nx)]])
        B_aug = np.vstack((dynamics.B, np.zeros((nx, nu))))
        c_aug = np.vstack((dynamics.c, np.zeros((nx))))

        return LinearDynamics(A=A_aug, B=B_aug, c=c_aug)

    @staticmethod
    def _change_stage_cost(
        cost: QuadraticStageCost, output_dim: int
    ) -> QuadraticStageCost:
        # do NOT weigh the integrator state
        nx = cost.Q.shape[0]
        nu = cost.R.shape[0]

        Q_aug = np.block(
            [[cost.Q, np.zeros((nx, nx))], [np.zeros((nx, nx)), np.zeros((nx, nx))]]
        )

        S_aug = np.block([[cost.S, np.zeros((nu, nx))]])

        q_aug = np.concatenate(cost.q, np.zeros((nx)))

        return QuadraticStageCost(Q=Q_aug, R=cost.R, S=S_aug, q=q_aug, r=cost.r)

    @staticmethod
    def _change_terminal_cost(
        cost: QuadraticTerminalCost, output_dim: int
    ) -> QuadraticTerminalCost:
        nx = cost.Q.shape[0]

        Q_aug = np.block(
            [[cost.Q, np.zeros((nx, nx))], [np.zeros((nx, nx)), np.zeros((nx, nx))]]
        )
        q_aug = np.concatenate(cost.q, np.zeros((nx)))

        return QuadraticTerminalCost(Q=Q_aug, q=q_aug)
