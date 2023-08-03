import numpy as np
from mpc.cost import StageCost, TimeInvariantCost
from mpc.problem import OptimalControlProblem
from mpc.solver import RiccatiSolver
from mpc.system import TimeInvariantDynamics

Q = np.eye(2)
R = np.eye(2)

A = np.ones((2, 2))
B = np.ones((2, 2))

horizon = 5
x0 = np.ones(2)

problem = OptimalControlProblem(
    horizon, TimeInvariantCost(StageCost(Q, R)), TimeInvariantDynamics(A, B), x0
)
solution = RiccatiSolver().solve(problem)

print("Solution:")

for k in range(horizon):
    print(f"time {k}")
    print(f"\tx = {np.array2string(solution.state[k], precision=4, floatmode='fixed')}")
    if k != horizon - 1:
        print(
            f"\tu = {np.array2string(solution.input[k], precision=4, floatmode='fixed')}"
        )
