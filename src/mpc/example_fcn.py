from typing import Tuple

import numpy as np
from numpy.typing import NDArray

np.random.seed(0)


def gen_symm_pos_def_matrix_old(dim: int) -> NDArray[np.float64]:
    while True:
        x = np.random.uniform(size=(dim, dim))
        x = x.T @ x
        if np.all(np.linalg.eigvals(x) > 0):
            return x


def gen_symm_pos_def_matrix(dim: int, max_eig: float = 10.0) -> NDArray[np.float64]:
    Q, _ = np.linalg.qr(np.random.normal(size=(dim, dim)))
    Ttemp = Q @ Q.T
    x = Ttemp.T @ np.diag(np.random.uniform(size=(dim), high=max_eig)) @ Ttemp
    return x


def solve_LQ(
    Q: NDArray[np.float64],
    R: NDArray[np.float64],
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    N: int,
    x0: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Solve canonical LQ: min Sum_k x'Qx + u'Ru
                                    st x' = Ax+Bu, x0 = xzero
    """
    nx = Q.shape[0]
    nu = R.shape[0]

    P = np.empty((N, nx, nx))
    u = np.empty((N, nu))
    x = np.empty((N, nx))
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

    return x, u


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

x, u = solve_LQ(Q, R, A, B, N, x0)
print("Solution:")

for k in range(N):
    print(f"time {k}")
    print(f"\tx = {np.array2string(x[k], precision=4, floatmode='fixed')}")
    if k != N - 1:
        print(f"\tu = {np.array2string(u[k], precision=4, floatmode='fixed')}")
