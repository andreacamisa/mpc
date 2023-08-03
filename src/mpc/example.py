import numpy as np
from numpy.typing import NDArray

np.random.seed(0)


def gen_symm_pos_def_matrix(dim: int) -> NDArray[np.float64]:
    while True:
        x = np.random.uniform(size=(dim, dim))
        x = x.T @ x
        if np.all(np.linalg.eigvals(x) > 0):
            return x


nx = 3
nu = 2
N = 20  # prediction horizon

# system matrices
A = np.random.uniform(size=(N, nx, nx))
B = np.random.uniform(size=(N, nx, nu))

# initial state
x0 = np.ones(nx)

# cost matrices
Q = gen_symm_pos_def_matrix(nx)
R = gen_symm_pos_def_matrix(nu)

# =============================
# begin algorithm

P = np.empty((N, nx, nx))
x = np.empty((N, nx))
u = np.empty((N, nu))
x[0] = x0

# backward pass
P[-1] = Q
for k in reversed(range(N - 1)):
    tmp = A[k].T @ P[k + 1] @ B[k]
    P[k] = (
        Q
        + A[k].T @ P[k + 1] @ A[k]
        - tmp @ np.linalg.inv(R + B[k].T @ P[k + 1] @ B[k]) @ tmp.T
    )

# forward pass
for k in range(N - 1):
    K = -np.linalg.inv(R + B[k].T @ P[k + 1] @ B[k]) @ (B[k].T @ P[k + 1] @ A[k])
    u[k] = K @ x[k]
    x[k + 1] = A[k] @ x[k] + B[k] @ u[k]

print("Solution:")

for k in range(N):
    print(f"time {k}")
    print(f"\tx = {np.array2string(x[k], precision=4, floatmode='fixed')}")
    if k != N - 1:
        print(f"\tu = {np.array2string(u[k], precision=4, floatmode='fixed')}")
