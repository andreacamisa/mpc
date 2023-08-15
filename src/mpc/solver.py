import itertools
from abc import abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
from mpc.problem import OptimalControlProblem
from nptyping import Float, NDArray, Shape


@dataclass(frozen=True)
class ProblemSolution:
    """Solution to an optimal control problem."""

    state_traj: NDArray[Shape["T, N"], Float]
    """State trajectory (numpy array with time on rows and state entries on columns)."""

    input_traj: NDArray[Shape["T, M"], Float]
    """Input trajectory (numpy array with time on rows and input entries on columns)."""


"""
TODO (acamisa): turn state and input trajectories into a fully-featured class that allows e.g. for:
- trajectory[t] or trajectory.get(t) <- return state vector at time t
- trajectory[t, k] or trajectory.get(t, k) <- return k-th entry of state at time t
- trajectory.get_entry(k) <- return trajectory vector of k-th entry of state
"""


class Solver:
    """Generic interface for optimal control problem solvers."""

    @abstractmethod
    def solve(self, problem: OptimalControlProblem) -> ProblemSolution:
        """Solve an optimal control problem."""
        pass


class RiccatiSolver(Solver):
    def solve(self, problem: OptimalControlProblem) -> ProblemSolution:
        # TODO check that problem is unconstrained LQ

        P_list: List[NDArray[Shape["N, N"], Float]] = []
        p_list: List[NDArray[Shape["*"], Float]] = []
        K_list: List[NDArray[Shape["N, M"], Float]] = []
        sigma_list: List[NDArray[Shape["*"], Float]] = []
        dynamics = list(
            itertools.islice(problem.system.get_dynamics(), problem.horizon)
        )
        cost = list(itertools.islice(problem.cost.get_stage_cost(), problem.horizon))
        terminal_cost = problem.cost.get_terminal_cost()

        # backward pass by prepending elements
        P_kp = terminal_cost.Q
        p_kp = terminal_cost.q
        P_list.append(P_kp)
        p_list.append(p_kp)
        for k in reversed(range(problem.horizon - 1)):
            A_k, B_k, c_k = dynamics[k].A, dynamics[k].B, dynamics[k].c
            Q_k, R_k, S_k, q_k, r_k = (
                cost[k].Q,
                cost[k].R,
                cost[k].S,
                cost[k].q,
                cost[k].r,
            )

            # update P
            tmp = np.linalg.inv(R_k + B_k.T @ P_kp @ B_k)
            K_k = -tmp @ (S_k + B_k.T @ P_kp @ A_k)
            P_k = Q_k + A_k.T @ P_kp @ A_k + K_k.T @ (R_k + B_k.T @ P_kp @ B_k) @ K_k

            # update p
            sigma_k = -tmp @ (r_k + B_k.T @ (p_kp + P_kp @ c_k))
            p_k = (
                q_k
                + A_k.T @ p_kp
                + A_k.T @ P_kp @ c_k
                + K_k.T @ (R_k + B_k.T @ P_kp @ B_k) @ sigma_k
            )

            # add to lists
            P_list.insert(0, P_k)
            p_list.insert(0, p_k)
            K_list.insert(0, K_k)
            sigma_list.insert(0, sigma_k)
            P_kp = P_k
            p_kp = p_k

        x_traj = np.zeros((problem.horizon, A_k.shape[0]))
        x_traj[0] = problem.initial_state
        u_traj = np.zeros((problem.horizon, B_k.shape[1]))

        # forward pass
        for k in range(problem.horizon - 1):
            A_k, B_k, c_k = dynamics[k].A, dynamics[k].B, dynamics[k].c
            u_traj[k] = K_list[k] @ x_traj[k] + sigma_list[k]
            x_traj[k + 1] = A_k @ x_traj[k] + B_k @ u_traj[k] + c_k

        return ProblemSolution(x_traj, u_traj)
