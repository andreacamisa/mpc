import pytest
from common import (
    C_VEC,
    HORIZON,
    NO_Q_VEC,
    NO_R_VEC,
    NO_S,
    Q_VEC,
    R_VEC,
    X0,
    A,
    B,
    Q,
    R,
    S,
)
from mpc.cost import QuadraticStageCost, QuadraticTerminalCost, TimeInvariantCost
from mpc.problem import OptimalControlProblem
from mpc.system import LinearDynamics, TimeInvariantSystem


@pytest.fixture
def regulation_problem() -> OptimalControlProblem:
    return OptimalControlProblem(
        HORIZON,
        TimeInvariantCost(
            QuadraticStageCost(Q, R, NO_S, NO_Q_VEC, NO_R_VEC),
            QuadraticTerminalCost(Q, NO_Q_VEC),
        ),
        TimeInvariantSystem(LinearDynamics(A, B, C_VEC)),
        X0,
    )


@pytest.fixture
def tracking_problem() -> OptimalControlProblem:
    return OptimalControlProblem(
        HORIZON,
        TimeInvariantCost(
            QuadraticStageCost(Q, R, S, Q_VEC, R_VEC), QuadraticTerminalCost(Q, Q_VEC)
        ),
        TimeInvariantSystem(LinearDynamics(A, B, C_VEC)),
        X0,
    )
