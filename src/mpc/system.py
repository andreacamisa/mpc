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
    """Linear dynamics (with an affine term) of the form

    `x(t+1) = Ax(t) + Bu(t) + c`
    """

    A: NDArray[Shape["N, N"], Float]
    """State matrix multiplying `x`."""

    B: NDArray[Shape["N, M"], Float]
    """Input matrix multiplying `u`."""

    c: NDArray[Shape["N"], Float]  # TODO (acamisa): make field optional
    """Constant term added to state equation."""


class System(ABC):
    """Discrete-time dynamical system of the form `x(t+1) = f(x(t), u(t), t)`."""

    @abstractmethod
    def get_dynamics(self) -> Iterator[LinearDynamics]:
        """Get iterator for linearized dynamics of system.

        Returns an iterator where each element represents the linearized dynamics of the
        system at each time instant, starting from time 0.
        """
        # TODO (acamisa): find better wording here, it's not correct to say linearized dynamics
        # and then return also an affine term
        pass


class TimeInvariantSystem(System):
    def __init__(self, dynamics: LinearDynamics) -> None:
        self._dynamics = dynamics

    def get_dynamics(self) -> Iterator[LinearDynamics]:
        return itertools.repeat(self._dynamics)


class TimeVaryingSystem(System):
    def __init__(self, dynamics: Iterable[LinearDynamics]) -> None:
        self._dynamics = dynamics

    def get_dynamics(self) -> Iterator[LinearDynamics]:
        return iter(self._dynamics)


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
