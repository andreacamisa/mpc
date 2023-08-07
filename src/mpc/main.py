import numpy as np
from mpc.cost import QuadraticStageCost, QuadraticTerminalCost, TimeInvariantCost
from mpc.problem import OptimalControlProblem
from mpc.solver import RiccatiSolver
from mpc.system import LinearDynamics, TimeInvariantSystem

# cost
Q = np.eye(2)
R = np.eye(2)
S = np.zeros((2, 2))
q = np.zeros(2)
r = np.zeros(2)

# dynamics
A = np.ones((2, 2))
B = np.ones((2, 2))
c = np.zeros(2)

horizon = 10
x0 = np.ones(2)

problem = OptimalControlProblem(
    horizon,
    TimeInvariantCost(QuadraticStageCost(Q, R, S, q, r), QuadraticTerminalCost(Q, q)),
    TimeInvariantSystem(LinearDynamics(A, B, c)),
    x0,
)

# TODO add util function to generate stage cost randomly
# StageCost.generate_random()

# TODO I want to change the cost matrix
# problem.cost.Q = []

solution = RiccatiSolver().solve(problem)

# TODO we actually would like the package to understand itself what is the best solver to use
# we will have a class that knows all the solvers and their capabilities and queries
# problem object to understand what are its requirements (e.g. linear dynamics, quadratic cost, constraints)
# ProblemSolver.solve(problem)

# TODO high-level interface to build problems easily
# problem = (
#     ProblemBuilder()
#     .add_quadratic_cost(Q, R)
#     .setpoint(...)
#     .add_linear_dynamics(A, B)
#     .add_integral_action()
#     .initial_state(x0)
#     .add_deltau_penalty()
#     .build()
# )

# TODO allow the user to create variations of problems
# problems2 = problem.add_deltau_penalty()
# or: problems2 = problem.get_builder().add_deltau_penalty().build()

print("Solution:")

for k in range(horizon):
    print(f"time {k}")
    print(f"\tx = {np.array2string(solution.state[k], precision=4, floatmode='fixed')}")
    if k != horizon - 1:
        print(
            f"\tu = {np.array2string(solution.input[k], precision=4, floatmode='fixed')}"
        )
