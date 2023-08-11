from abc import ABC, abstractmethod

from mpc.problem import OptimalControlProblem
from mpc.solver import ProblemSolution


class ProblemTransform(ABC):
    """Transformation applied to an optimal control problem and to its solution.

    This class represents a generic transformation applied to an optimal control problem.
    An `OptimalControlProblem` instance can be transformed to another `OptimalControlProblem`
    instance through the method `transform_problem()`. The resulting problem will likely have
    a changed structure with e.g. altered cost, system dynamics, augmented state vector etc.
    Once the solution to the new problem is found, in the form of a `ProblemSolution` object,
    the solution to the original problem can be obtained through the method
    `inverse_transform_solution()`.
    """

    @abstractmethod
    def transform_problem(
        self, problem: OptimalControlProblem
    ) -> OptimalControlProblem:
        """Apply transformation to optimal control problem and obtain a new problem formulation."""
        pass

    @abstractmethod
    def inverse_transform_solution(self, solution: ProblemSolution) -> ProblemSolution:
        """Apply the inverse transformation to problem solution and obtain solution
        to untransformed problem."""
        pass
