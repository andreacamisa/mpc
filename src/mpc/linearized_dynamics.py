import numpy as np
from numpy.typing import NDArray


def dynamics(x: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
    xp = x + u
    df_dx = np.array[[1]]
    df_du = np.array[[1]]
    return xp, df_dx, df_du


def shooting(
    N: int, nx: int, nu: int, x0: NDArray[np.float64], dynamics
) -> NDArray[np.float64]:
    utraj = np.zeros((N, nu))
    xtraj = np.empty((N, nx))
    xtraj[0] = x0

    for k in range(N):
        xtraj[k + 1], _, _ = dynamics(xtraj[k], utraj[k])

    return xtraj, utraj


def compute_linearized_dynamics(
    N: int, xtraj: NDArray[np.float64], utraj: NDArray[np.float64], dynamics
) -> NDArray[np.float64]:
    nx = xtraj[0].shape[0]
    nu = utraj[0].shape[0]

    A = np.empty((N, nx, nx))
    B = np.empty((N, nx, nu))
    for k in range(N):
        _, _, A[k], B[k] = dynamics(xtraj[k], utraj[k])

    return A, B


N = 10
nx = 3
nu = 2
x0 = np.ones((nx))
xtraj, utraj = shooting(N, nx, nu, x0, dynamics)
A, B = compute_linearized_dynamics(N, xtraj, utraj, dynamics)
