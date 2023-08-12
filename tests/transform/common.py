import numpy as np

STATE_DIM = 4
INPUT_DIM = 2
OUTPUT_DIM = 3
HORIZON = 5
A = 1 * np.eye(STATE_DIM)
B = 2 * np.eye(STATE_DIM, INPUT_DIM)
C_VEC = np.zeros(STATE_DIM)
C = 3 * np.eye(OUTPUT_DIM, STATE_DIM)
D = 4 * np.eye(OUTPUT_DIM)
X0 = np.arange(STATE_DIM)
Q = 5 * np.eye(STATE_DIM)
R = 6 * np.eye(INPUT_DIM)
S = 7 * np.eye(INPUT_DIM, STATE_DIM)
Q_VEC = 8 * np.ones(STATE_DIM)
R_VEC = 9 * np.ones(INPUT_DIM)

# zero vectors and matrices
NO_D = 0 * D
NO_S = 0 * S
NO_Q_VEC = 0 * Q_VEC
NO_R_VEC = 0 * R_VEC
