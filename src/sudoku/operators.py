from typing import TYPE_CHECKING

import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.core.survival import Survival
from pymoo.core.termination import Termination

if TYPE_CHECKING:
    from sudoku.problem import SudokuProblem


class MySampling(Sampling):
    def _do(
        self,
        problem: "SudokuProblem",
        n_samples: int,
        random_state: np.random.Generator = None,
        **kwargs,
    ):
        N = problem.N
        A = problem.associated_matrix
        X = np.zeros((n_samples, N**2), dtype=np.int32)
        full_range = np.arange(1, N + 1)
        for i in range(n_samples):
            grid = problem.initial_board.copy()
            for r in range(N):
                given = grid[r, A[r]]
                missing = np.setdiff1d(full_range, given, assume_unique=True)
                shuffled_missing = random_state.permutation(missing)
                grid[r, ~A[r]] = shuffled_missing
            X[i, :] = grid.ravel()
        return X


class RowCrossover(Crossover):
    """Crossover operation described in algorithm 1 of Wang et al."""

    def __init__(self, row_cross_rate=0.1, **kwargs):
        """
        Args:
            prob: probability of executing crossover.
                Wang et al. call this PC1.
            row_cross_rate: row crossover rate; probability used
                to determine whether to swap each row.
                Wang et al. call this PC2.
        """
        super().__init__(2, 2, **kwargs)
        self.row_cross_rate = row_cross_rate

    def _do(
        self,
        problem: "SudokuProblem",
        X: np.ndarray,
        random_state: np.random.Generator = None,
        **kwargs,
    ):
        # https://pymoo.org/customization/custom.html
        # X is (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape
        N = problem.N
        Y = X.copy()
        for i in range(n_matings):
            p1 = Y[0, i].reshape((N, N))
            p2 = Y[1, i].reshape((N, N))
            for r in range(N):
                if random_state.random() < self.row_cross_rate:
                    p1[r, :], p2[r, :] = p2[r, :].copy(), p1[r, :].copy()
            Y[0, i] = p1.flatten()
            Y[1, i] = p2.flatten()
        return Y


class SwapReinitMutation(Mutation):
    """Swap mutation and reinitialization mutation
    described in algorithm 2 of Wang et al.
    """

    def __init__(self, swap_rate=0.3, reinit_rate=0.05):
        """
        Args:
            swap_rate: swap mutation rate.
                Wang et al. call this PM1.
            reinit_rate: reinitialization mutation rate.
                Wang et al. call this PM2.
        """
        super().__init__()
        self.swap_rate = swap_rate
        self.reinit_rate = reinit_rate

    def _do(
        self,
        problem: "SudokuProblem",
        X: np.ndarray,
        random_state: np.random.Generator = None,
        **kwargs,
    ):
        Y = X.copy()
        N = problem.N
        A = problem.associated_matrix
        for i in range(len(Y)):
            grid = Y[i].reshape((N, N))
            for r in range(N):
                non_given_indices = np.where(~A[r])[0]
                if len(non_given_indices) < 2:
                    continue
                if random_state.random() < self.swap_rate:
                    a, b = random_state.choice(non_given_indices, size=2, replace=False)
                    grid[r, a], grid[r, b] = grid[r, b], grid[r, a]
                if random_state.random() < self.reinit_rate:
                    given_numbers = grid[r, A[r]]
                    missing_numbers = np.setdiff1d(np.arange(1, N + 1), given_numbers)
                    random_state.shuffle(missing_numbers)
                    grid[r, ~A[r]] = missing_numbers
            Y[i] = grid.flatten()
        return Y


class LocalSearchRepair(Repair):
    def _do(
        self,
        problem: "SudokuProblem",
        X: np.ndarray,
        random_state: np.random.Generator = None,
        **kwargs,
    ):
        N = problem.N
        A = problem.associated_matrix
        for i in range(len(X)):
            grid = X[i].reshape((N, N))
            grid = self.column_local_search(grid, N, A, random_state)
            grid = self.subblock_local_search(grid, N, A, random_state)
            X[i] = grid.flatten()
        return X

    def column_local_search(
        self, grid: np.ndarray, N: int, A: np.ndarray, random_state: np.random.Generator
    ):
        illegal_columns = [j for j in range(N) if len(np.unique(grid[:, j])) < N]

        if len(illegal_columns) < 2:
            return grid

        for j in illegal_columns:
            others = [c for c in illegal_columns if c != j]
            k = random_state.choice(others)

            vals_j, counts_j = np.unique(grid[:, j], return_counts=True)
            repeated_j = vals_j[counts_j > 1]
            vals_k, counts_k = np.unique(grid[:, k], return_counts=True)
            repeated_k = vals_k[counts_k > 1]

            # Non-given rows in col j that contain repeated numbers
            rows_j_guilty = np.where(np.isin(grid[:, j], repeated_j) & ~A[:, j])[0]
            # Non-given rows in col k that contain repeated numbers
            rows_k_guilty = np.where(np.isin(grid[:, k], repeated_k) & ~A[:, k])[0]
            rows_both_guilty = np.intersect1d(rows_j_guilty, rows_k_guilty)

            for r in rows_both_guilty:
                val_at_j = grid[r, j]
                val_at_k = grid[r, k]
                # Since we swap below and iterate again, rows_both_guilty may be wrong.
                # Count again I guess?
                if (
                    (np.count_nonzero(grid[:, j] == val_at_j) > 1)
                    and (np.count_nonzero(grid[:, k] == val_at_k) > 1)
                    and (val_at_k not in grid[:, j])
                    and (val_at_j not in grid[:, k])
                ):
                    grid[r, j], grid[r, k] = val_at_k, val_at_j

        return grid

    def subblock_local_search(
        self, grid: np.ndarray, N: int, A: np.ndarray, random_state: np.random.Generator
    ):
        # Slightly different from Wang et al.'s
        # Don't bother putting all sub-blocks into the same set,
        # the only ones we can even try to swap need to be in the same row.
        sqrt_N = int(np.sqrt(N))
        # (block_row, block_col, row_in_block, row_in_col)
        blocks = grid.reshape(sqrt_N, sqrt_N, sqrt_N, sqrt_N).swapaxes(1, 2)
        A_blocks = A.reshape(sqrt_N, sqrt_N, sqrt_N, sqrt_N).swapaxes(1, 2)

        for block_row_index in range(sqrt_N):
            block_row = blocks[block_row_index]  # (block_col, row_in_block, row_in_col)
            A_block_row = A_blocks[block_row_index]

            illegal_block_col_indices = [
                b for b in range(sqrt_N) if len(np.unique(block_row[b])) < N
            ]

            if len(illegal_block_col_indices) < 2:
                continue

            for b1 in illegal_block_col_indices:
                others = [b for b in illegal_block_col_indices if b != b1]
                b2 = random_state.choice(others)

                vals_1, counts_1 = np.unique(block_row[b1], return_counts=True)
                repeated_1 = vals_1[counts_1 > 1]
                vals_2, counts_2 = np.unique(block_row[b2], return_counts=True)
                repeated_2 = vals_2[counts_2 > 1]

                # 2D mask, cell is True if nongiven and is a duplicate
                nongiven_repeated_b1 = (
                    np.isin(block_row[b1], repeated_1) & ~A_block_row[b1]
                )
                nongiven_repeated_b2 = (
                    np.isin(block_row[b2], repeated_2) & ~A_block_row[b2]
                )

                rows_b1_guilty_mask = nongiven_repeated_b1.any(axis=1)
                rows_b2_guilty_mask = nongiven_repeated_b2.any(axis=1)
                rows_both_guilty = np.where(rows_b1_guilty_mask & rows_b2_guilty_mask)[
                    0
                ]

                for r in rows_both_guilty:
                    cand_j = np.where(nongiven_repeated_b1[r])[0]
                    cand_k = np.where(nongiven_repeated_b2[r])[0]
                    if cand_j.size > 0 and cand_k.size > 0:
                        j, k = cand_j[0], cand_k[0]
                        val_j = block_row[b1, r, j]
                        val_k = block_row[b2, r, k]
                        # Since we swap below and iterate again,
                        # rows_both_guilty may be wrong.
                        # Count again I guess?
                        if (
                            (np.count_nonzero(block_row[b1] == val_j) > 1)
                            and (np.count_nonzero(block_row[b2] == val_k) > 1)
                            and (val_k not in block_row[b1])
                            and (val_j not in block_row[b2])
                        ):
                            block_row[b1, r, j], block_row[b2, r, k] = val_k, val_j

        return grid


class EPLSurvival(Survival):
    def __init__(self, n_elite: int, sampling: Sampling):
        super().__init__(filter_infeasible=False)
        self.n_elite = n_elite
        self.sampling = sampling

    def _do(
        self,
        problem,
        pop,
        n_survive=None,
        random_state: np.random.Generator = None,
        **kwargs,
    ):
        F = pop.get("F")[:, 0]
        idx_sorted = np.argsort(F)

        elites = pop[idx_sorted[: self.n_elite]]
        candidates = pop[idx_sorted[self.n_elite : n_survive]]

        new_pop_list = list(elites)
        for ind in candidates:
            random_elite = random_state.choice(elites)
            f_bad = ind.F[0]
            f_elite = random_elite.F[0]
            p_b = (f_bad - f_elite) / f_bad if f_bad > 0 else 0
            if random_state.random() < p_b:
                new_pop_list.append(random_elite.copy())
            else:
                new_x = self.sampling._do(
                    problem, n_samples=1, random_state=random_state
                )[0]
                new_ind = ind.copy()
                new_ind.X = new_x
                new_ind.F = None
                new_pop_list.append(new_ind)

        new_pop = Population.create(*new_pop_list)
        is_evaluated = np.array([ind.F is not None for ind in new_pop])
        if not np.all(is_evaluated):
            to_be_evaluated = new_pop[~is_evaluated]
            X_to_eval = to_be_evaluated.get("X")
            out = problem.evaluate(X_to_eval, return_values_of=["F"])
            for i, ind in enumerate(to_be_evaluated):
                ind.F = out[i]
        return new_pop


class ZeroFunctionValueTermination(Termination):
    def __init__(self) -> None:
        super().__init__()

    def _update(self, algorithm):
        opt = algorithm.opt
        return 1.0 if opt.get("F").min() <= 0 else 0.0
