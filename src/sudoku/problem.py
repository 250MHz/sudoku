import numpy as np
from pymoo.core.problem import Problem


class SudokuProblem(Problem):
    def __init__(self, initial_board: np.ndarray):
        """Defines metadata for a Sudoku problem.

        Args:
            initial_board: 2D array of board of size n x n
                where sqrt(n) is a positive integer
        """
        self.initial_board = initial_board
        self.N = len(initial_board)
        self.sqrt_N = int(np.sqrt(self.N))
        assert np.sqrt(self.N) == self.sqrt_N

        # If [i][j] == 1, then given by board. If 0, then open.
        self.associated_matrix = initial_board != 0

        super().__init__(n_var=self.N**2, n_obj=1, xl=1, xu=self.N, vtype=np.int32)

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        pop_size = len(x)
        major_matrices = x.reshape((pop_size, self.N, self.N))
        col_conflicts = np.zeros(pop_size, dtype=np.int32)
        subblock_conflicts = np.zeros(pop_size, dtype=np.int32)

        # This fitness function is different than Wang et al.'s
        # We count the number of values that are missing from
        # the columns and sub-blocks,
        # rather than doing a binary value for whether constraint is met.

        for i in range(self.N):
            columns = major_matrices[:, :, i]
            # In every column i, check if a value in [1, N] is missing.
            # If so, increment num violations for that individual.
            for val in range(1, self.N + 1):
                is_missing = ~np.any(columns == val, axis=1)
                col_conflicts += is_missing.astype(np.int32)

        for r_offset in range(0, self.N, self.sqrt_N):
            for c_offset in range(0, self.N, self.sqrt_N):
                subblocks = major_matrices[
                    :,
                    r_offset : r_offset + self.sqrt_N,
                    c_offset : c_offset + self.sqrt_N,
                ]
                subblocks_flat = subblocks.reshape((pop_size, self.N))
                for val in range(1, self.N + 1):
                    is_missing = ~np.any(subblocks_flat == val, axis=1)
                    subblock_conflicts += is_missing.astype(np.int32)

        out["F"] = (col_conflicts + subblock_conflicts).reshape(-1, 1).astype(float)
