from typing import cast

import numpy as np
from numpy.typing import NDArray

np.random.seed(0)


def gen_symm_pos_def_matrix(dim: int, max_eig: float = 10.0) -> NDArray[np.float64]:
    orthMatrix, _ = np.linalg.qr(np.random.normal(size=(dim, dim)))
    x = orthMatrix.T @ np.diag(np.random.uniform(size=(dim), high=max_eig)) @ orthMatrix
    return cast(NDArray[np.float64], x)


nx = 3
nu = 2
N = 20  # prediction horizon

# system matrices
A = np.random.uniform(size=(nx, nx))
B = np.random.uniform(size=(nx, nu))

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
    tmp = A.T @ P[k + 1] @ B
    P[k] = Q + A.T @ P[k + 1] @ A - tmp @ np.linalg.inv(R + B.T @ P[k + 1] @ B) @ tmp.T

# forward pass
for k in range(N - 1):
    K = -np.linalg.inv(R + B.T @ P[k + 1] @ B) @ (B.T @ P[k + 1] @ A)
    u[k] = K @ x[k]
    x[k + 1] = A @ x[k] + B @ u[k]

print("Solution:")

for k in range(N):
    print(f"time {k}")
    print(f"\tx = {np.array2string(x[k], precision=4, floatmode='fixed')}")
    if k != N - 1:
        print(f"\tu = {np.array2string(u[k], precision=4, floatmode='fixed')}")
