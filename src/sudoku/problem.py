import numpy as np
from numba import njit
from pymoo.core.problem import Problem


@njit
def evaluate_population_binary(X_pop, N, sqrt_N):
    pop_size = X_pop.shape[0]
    out = np.zeros((pop_size, 1))

    for i in range(pop_size):
        grid = X_pop[i].reshape((N, N))
        fitness = 0
        for c in range(N):
            seen = 0
            unique_count = 0
            for r in range(N):
                val = grid[r, c]
                if not (seen & (1 << val)):
                    seen |= 1 << val
                    unique_count += 1
            if unique_count < N:
                fitness += 1
        for br in range(sqrt_N):
            for bc in range(sqrt_N):
                seen = 0
                unique_count = 0
                for r in range(br * sqrt_N, (br + 1) * sqrt_N):
                    for c in range(bc * sqrt_N, (bc + 1) * sqrt_N):
                        val = grid[r, c]
                        if not (seen & (1 << val)):
                            seen |= 1 << val
                            unique_count += 1
                if unique_count < N:
                    fitness += 1
        out[i, 0] = fitness
    return out


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
        out["F"] = evaluate_population_binary(x, self.N, self.sqrt_N)
