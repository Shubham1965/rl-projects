from math import floor

import numpy as np


def row_col_to_seq(row_col: np.ndarray, num_cols: int) -> int:
    return row_col[:, 0] * num_cols + row_col[:, 1]


def seq_to_col_row(seq, num_cols):
    r = floor(seq / num_cols)
    c = seq - r * num_cols
    return np.array([[r, c]])
