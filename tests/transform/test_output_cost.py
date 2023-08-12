import numpy as np
import pytest
from common import HORIZON, NO_D, NO_Q_VEC, NO_R_VEC, NO_S, X0, C, D, Q, R
from mpc.problem import OptimalControlProblem
from mpc.transform.output_cost import OutputCostTransform
from nptyping import Float, NDArray, Shape


@pytest.mark.parametrize("D_matrix", [D, NO_D])
def test__output_cost_transform__regulation_problem(
    regulation_problem: OptimalControlProblem, D_matrix: NDArray[Shape["P, P"], Float]
) -> None:
    transform = OutputCostTransform(C, D_matrix)
    new_problem, _ = transform.apply(regulation_problem)

    assert new_problem.horizon == HORIZON
    assert np.allclose(new_problem.initial_state, X0)

    # test stage cost
    for cost in new_problem.cost.get_stage_cost():
        assert np.allclose(cost.Q, Q)  # TODO IVANO
        assert np.allclose(cost.R, R)  # TODO IVANO

        # regulation problem has no S, q, r
        assert np.allclose(cost.S, NO_S)
        assert np.allclose(cost.q, NO_Q_VEC)
        assert np.allclose(cost.r, NO_R_VEC)

    # test terminal cost
    terminal_cost = new_problem.cost.get_terminal_cost()
    assert np.allclose(terminal_cost.Q, Q)  # TODO IVANO
    assert np.allclose(terminal_cost.q, NO_Q_VEC)


# TODO IVANO test simile sostituendo regulation_problem con tracking_problem
