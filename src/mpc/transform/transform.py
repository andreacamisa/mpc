from abc import ABC, abstractmethod
from typing import Tuple

from mpc.problem import OptimalControlProblem
from mpc.solver import ProblemSolution


class ProblemTransform(ABC):
    """Transformation applied to an optimal control problem.

    This class represents a generic transformation of an optimal control problem.
    Use the method `apply()` to get the transformed optimal control problem and an associated
    `SolutionTransform` that can be applied to obtain the solution to the original problem.
    """

    @abstractmethod
    def apply(
        self, problem: OptimalControlProblem
    ) -> Tuple[OptimalControlProblem, "SolutionTransform"]:
        """Apply transformation to optimal control problem and obtain the new problem formulation.

        Returns:
            A tuple containing, respectively, the transformed optimal control problem
            and a transformation that can be applied to its solution to get a solution to
            the original problem.
        """
        pass


class SolutionTransform(ABC):
    """Transformation applied to an optimal control problem solution.

    Represents a generic transformation of a solution of an optimal control problem.
    Use the method `apply()` to get the transformed optimal control problem solution.
    This class is typically returned by instances of `ProblemTransform` and is not meant
    to be instantiated directly.
    """

    @abstractmethod
    def apply(self, solution: ProblemSolution) -> ProblemSolution:
        """Apply transformation to problem solution and obtain the transformed solution."""
        pass


class IdentitySolutionTransform(SolutionTransform):
    """Solution transformation that returns the given solution without any change."""

    def apply(self, solution: ProblemSolution) -> ProblemSolution:
        return solution


class PartialStateSolutionTransform(SolutionTransform):
    def __init__(self, n_keep: int) -> None:
        """Solution transform that keeps the first "n" components of the state vector.

        This transform returns a new `ProblemSolution` object with
        - unchanged input trajectory
        - state trajectory that keeps only the first "n" components of state vector

        Args:
            n_keep: number of components to keep from the state vector, starting from the the first one
        """
        self._n_keep = n_keep

    def apply(self, solution: ProblemSolution) -> ProblemSolution:
        state_traj = solution.state_traj[: self._n_keep]
        return ProblemSolution(state_traj=state_traj, input_traj=solution.input_traj)
