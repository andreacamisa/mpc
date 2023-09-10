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
from mpc.cost import StageCost, TerminalCost, TimeInvariantCost
from mpc.problem import OptimalControlProblem
from mpc.system import LinearDynamics, TimeInvariantSystem


@pytest.fixture
def regulation_problem() -> OptimalControlProblem:
    return OptimalControlProblem(
        HORIZON,
        TimeInvariantCost(
            StageCost(Q, R, NO_S, NO_Q_VEC, NO_R_VEC),
            TerminalCost(Q, NO_Q_VEC),
        ),
        TimeInvariantSystem(LinearDynamics(A, B, C_VEC)),
        X0,
    )


@pytest.fixture
def tracking_problem() -> OptimalControlProblem:
    return OptimalControlProblem(
        HORIZON,
        TimeInvariantCost(StageCost(Q, R, S, Q_VEC, R_VEC), TerminalCost(Q, Q_VEC)),
        TimeInvariantSystem(LinearDynamics(A, B, C_VEC)),
        X0,
    )
