import numpy as np
import pytest
from mpc.solver import ProblemSolution
from mpc.transform.transform import (
    IdentitySolutionTransform,
    PartialStateSolutionTransform,
)

STATE_DIM = 4
INPUT_DIM = 2
TRAJECTORY_LENGTH = 5


@pytest.fixture
def solution() -> ProblemSolution:
    return ProblemSolution(
        np.arange(STATE_DIM * TRAJECTORY_LENGTH).reshape(
            (STATE_DIM, TRAJECTORY_LENGTH)
        ),
        np.arange(INPUT_DIM * TRAJECTORY_LENGTH).reshape(
            (INPUT_DIM, TRAJECTORY_LENGTH)
        ),
    )


def test__identity_solution_transform__returns_same_solution(
    solution: ProblemSolution,
) -> None:
    transform = IdentitySolutionTransform()
    new_solution = transform.apply(solution)

    assert np.allclose(new_solution.input_traj, solution.input_traj)
    assert np.allclose(new_solution.state_traj, solution.state_traj)


def test__partial_state_solution_transform__returns_only_first_state_components(
    solution: ProblemSolution,
) -> None:
    n_keep = 2
    transform = PartialStateSolutionTransform(n_keep)
    new_solution = transform.apply(solution)

    assert np.allclose(new_solution.input_traj, solution.input_traj)
    assert np.allclose(new_solution.state_traj, solution.input_traj[:n_keep, :])
