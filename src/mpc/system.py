"""
TODO (acamisa): classes with similar structure as Cost but allow to get system matrices

(perhaps this is a hint that we should think of something more generic, which can be either
time invariant or time-varying and allows you to get the information at different time instants?)
"""


import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Tuple

import numpy as np
from nptyping import Float, NDArray, Shape
from typing_extensions import TypeAlias

State: TypeAlias = NDArray[Shape["*"], Float]
Input: TypeAlias = NDArray[Shape["*"], Float]
StateJacobian: TypeAlias = NDArray[Shape["N, N"], Float]
InputJacobian: TypeAlias = NDArray[Shape["N, M"], Float]


@dataclass
class LinearDynamics:
    """Linear dynamics of the form

    `x(t+1) = Ax(t) + Bu(t)`
    """

    A: NDArray[Shape["N, N"], Float]
    """State matrix multiplying `x`."""

    B: NDArray[Shape["N, M"], Float]
    """Input matrix multiplying `u`."""


@dataclass
class CompleteLinearDynamics(LinearDynamics):
    """Linear dynamics, with output equation, of the form

    `x(t+1) = Ax(t) + Bu(t)` (state equation)

    `y(t) = Cx(t) + Du(t)` (output equation)
    """

    C: NDArray[Shape["P, N"], Float]
    """Output matrix multiplying `x`."""

    D: NDArray[Shape["P, M"], Float]
    """Direct input-output feedthrough matrix multiplying `u`."""


class System(ABC):
    """Linear, discrete-time dynamical system of the form `x(t+1) = A(t) x(t) + B u(t)`."""

    @abstractmethod
    def get_dynamics(self) -> Iterator[LinearDynamics]:
        """Get iterator for dynamics of system.

        Returns an iterator where each element represents the dynamics of the
        system at each time instant, starting from time 0.
        """
        pass

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of the state vector."""
        pass


class TimeInvariantSystem(System):
    def __init__(self, dynamics: LinearDynamics) -> None:
        self._dynamics = dynamics

    def get_dynamics(self) -> Iterator[LinearDynamics]:
        return itertools.repeat(self._dynamics)

    @property
    def state_dim(self) -> int:
        return self._dynamics.A.shape[0]


class TimeVaryingSystem(System):
    def __init__(self, dynamics: Iterable[LinearDynamics], state_dim: int) -> None:
        self._dynamics = dynamics
        self._state_dim = state_dim

    def get_dynamics(self) -> Iterator[LinearDynamics]:
        return iter(self._dynamics)

    @property
    def state_dim(self) -> int:
        return self._state_dim


class LinearizedSystem(System):
    def __init__(
        self,
        dynamics: Callable[[State, Input], Tuple[State, StateJacobian, InputJacobian]],
        x0: State,
        inputs: Iterable[Input],
    ) -> None:
        self._dynamics = dynamics
        self._x0 = x0
        self._inputs = inputs

    def get_dynamics(self) -> Iterator[LinearDynamics]:
        x = self._x0
        for u in iter(self._inputs):
            x, A_jac, B_jac = self._dynamics(x, u)
            yield LinearDynamics(A_jac, B_jac, np.zeros(x.shape))

    @property
    def state_dim(self) -> int:
        return self._x0.shape[0]
