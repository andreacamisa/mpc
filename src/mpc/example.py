import numpy as np

print("Example")


def RiccatiBW() -> float:
    return 0


nx = 3
nu = 2
N = 20

A = np.random.rand((nx, nx))
B = np.random.rand((nx, nu))
P = np.nan((N, nx, nx))
K = np.nan((N, nu, nx))
x = np.nan((N, nx))
u = np.nan((N, nu))
print(A)

P[-1] = np.random.rand()
for k in reversed(range(N)):
    P[k] = A.T @ P[k + 1] @ A
    K[k] = A @ x[k] + B @ u[k]

for k in range(N):
    u[k] = K[k] @ x[k]
    x[k + 1] = A @ x[k] + B @ u[k]
