import numpy as np


def test_numpy_one_index_refers_to_row() -> None:
    x = np.zeros((2, 2))
    x[1] = np.asarray([1, 2])

    assert np.all(x[1, :] == np.asarray([1, 2]))
