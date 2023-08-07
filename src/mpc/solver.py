import itertools
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
        P_list: List[NDArray[Shape["N, N"], Float]] = []
        K_list: List[NDArray[Shape["N, M"], Float]] = []
        dynamics = list(
            itertools.islice(problem.system.get_dynamics(), problem.horizon)
        )
        cost = list(itertools.islice(problem.cost.get_stage_cost(), problem.horizon))

        # backward pass by prepending elements to P
        P_list.append(problem.cost.get_terminal_cost().Q)
        for k in reversed(range(problem.horizon - 1)):
            A_k, B_k = dynamics[k].A, dynamics[k].B
            Q_k, R_k = cost[k].Q, cost[k].R

            P_kp = P_list[0]  # because we prepend, P[0] is always the last P computed
            tmp = A_k.T @ P_kp @ B_k
            P_k = (
                Q_k
                + A_k.T @ P_kp @ A_k
                - tmp @ np.linalg.inv(R_k + B_k.T @ P_kp @ B_k) @ tmp.T
            )
            K_k = -np.linalg.inv(R_k + B_k.T @ P_kp @ B_k) @ (B_k.T @ P_kp @ A_k)

            # prepend to lists
            P_list.insert(0, P_k)
            K_list.insert(0, K_k)

        x_traj = np.zeros((problem.horizon, A_k.shape[0]))
        x_traj[0] = problem.initial_state
        u_traj = np.zeros((problem.horizon, B_k.shape[1]))

        # forward pass
        for k in range(problem.horizon - 1):
            A_k, B_k = dynamics[k].A, dynamics[k].B
            u_traj[k] = K_list[k] @ x_traj[k]
            x_traj[k + 1] = A_k @ x_traj[k] + B_k @ u_traj[k]

        return ProblemSolution(x_traj, u_traj)
