from abc import abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
from mpc.problem import OptimalControlProblem
from nptyping import Float, NDArray, Shape
from typing_extensions import TypeAlias

Trajectory: TypeAlias = NDArray[Shape["Time, State"], Float]
"""
TODO (acamisa): turn Trajectory into a fully-featured class that allows e.g. for:
- trajectory[t] or trajectory.get(t) <- return state vector at time t
- trajectory[t, k] or trajectory.get(t, k) <- return k-th entry of state at time t
- trajectory.get_entry(k) <- return trajectory vector of k-th entry of state
"""


@dataclass(frozen=True)
class ProblemSolution:
    state: Trajectory
    input: Trajectory


class Solver:
    @abstractmethod
    def solve(self, problem: OptimalControlProblem) -> ProblemSolution:
        pass


class RiccatiSolver(Solver):
    def solve(self, problem: OptimalControlProblem) -> ProblemSolution:
        # NOTE (acamisa): this is just a draft implementation

        P_list: List[NDArray[Shape["x, x"], Float]] = []
        K_list: List[NDArray[Shape["x, u"], Float]] = []

        # backward pass by prepending elements to P
        P_list.append(problem.cost.get_terminal_cost())
        for k in reversed(range(problem.horizon - 1)):
            A_k, B_k = problem.dynamics.get_matrices(k)
            Q_k = problem.cost.get_state_cost(k)
            R_k = problem.cost.get_input_cost(k)

            P_kp = P_list[0]  # because we prepend, P[0] is always the last P computed
            tmp = A_k.T @ P_kp @ B_k
            P_k = (
                Q_k
                + A_k.T @ P_kp @ A_k
                - tmp @ np.linalg.inv(R_k + B_k.T @ P_kp @ B_k) @ tmp.T
            )
            K_k = -np.linalg.inv(R_k + B_k.T @ P_k @ B_k) @ (
                B_k.T @ P_k @ A_k
            )  # TODO (acamisa): DOUBLE CHECK - I think P[k+1] was a typo so I corrected to P[k]

            # prepend to lists
            P_list.insert(0, P_k)
            K_list.insert(0, K_k)

        # NOTE (acamisa): here, P_list has N elements, while K_list has N-1 elements

        x_traj = np.zeros((problem.horizon[0], A_k.shape[0]))
        x_traj[0] = problem.initial_state
        u_traj = np.zeros((problem.horizon[0], B_k.shape[1]))

        # forward pass
        for k in range(problem.horizon - 1):
            A_k, B_k = problem.dynamics.get_matrices(k)
            u_traj[k] = K_list[k] @ x_traj[k]
            if k != problem.horizon - 2:  # avoid storing final state
                x_traj[k + 1] = A_k @ x_traj[k] + B_k @ u_traj[k]

        return ProblemSolution(x_traj, u_traj)
