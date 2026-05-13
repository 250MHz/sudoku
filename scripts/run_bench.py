import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.termination.collection import TerminationCollection
from tqdm import tqdm

from sudoku.operators import (
    EPLSurvival,
    LocalSearchRepairV2,
    MySampling,
    RowCrossover,
    SwapReinitMutation,
    ZeroFunctionValueTermination,
)
from sudoku.problem import SudokuProblem

puzzles = {
    "easy-1": np.array(
        [
            [0, 0, 9, 0, 0, 0, 1, 0, 0],
            [2, 1, 7, 0, 0, 0, 3, 6, 8],
            [0, 0, 0, 2, 0, 7, 0, 0, 0],
            [0, 6, 4, 1, 0, 3, 5, 8, 0],
            [0, 7, 0, 0, 0, 0, 0, 3, 0],
            [1, 5, 0, 4, 2, 8, 0, 7, 9],
            [0, 0, 0, 5, 8, 9, 0, 0, 0],
            [4, 8, 5, 0, 0, 0, 2, 9, 3],
            [0, 0, 6, 3, 0, 2, 8, 0, 0],
        ]
    ),
    "easy-11": np.array(
        [
            [2, 9, 0, 7, 0, 1, 0, 0, 0],
            [5, 3, 0, 0, 6, 0, 1, 0, 0],
            [0, 0, 6, 3, 0, 0, 0, 4, 0],
            [0, 0, 0, 5, 9, 0, 0, 0, 4],
            [0, 1, 5, 0, 0, 4, 6, 8, 9],
            [0, 0, 0, 1, 8, 0, 0, 0, 3],
            [0, 0, 2, 6, 0, 0, 0, 9, 0],
            [3, 6, 0, 0, 4, 0, 7, 0, 0],
            [9, 4, 0, 8, 0, 5, 0, 0, 0],
        ]
    ),
    "medium-27": np.array(
        [
            [0, 1, 0, 5, 0, 6, 0, 2, 0],
            [3, 0, 0, 0, 0, 0, 0, 0, 6],
            [0, 0, 9, 1, 0, 4, 5, 0, 0],
            [0, 9, 0, 0, 1, 0, 0, 4, 0],
            [0, 7, 0, 3, 0, 2, 0, 5, 0],
            [0, 3, 0, 0, 8, 0, 0, 6, 0],
            [0, 0, 3, 2, 0, 7, 1, 0, 0],
            [9, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 5, 0, 6, 0, 1, 0, 8, 0],
        ]
    ),
    "medium-29": np.array(
        [
            [0, 0, 1, 0, 8, 0, 0, 0, 0],
            [0, 0, 0, 3, 0, 4, 7, 5, 0],
            [0, 6, 0, 0, 5, 0, 0, 0, 0],
            [8, 0, 6, 0, 0, 2, 3, 4, 9],
            [0, 0, 9, 0, 0, 0, 0, 0, 0],
            [3, 0, 4, 0, 0, 8, 1, 7, 2],
            [0, 3, 0, 0, 7, 0, 0, 0, 0],
            [0, 0, 0, 8, 0, 1, 5, 6, 0],
            [0, 0, 2, 0, 3, 0, 0, 0, 0],
        ]
    ),
    "hard-77": np.array(
        [
            [5, 0, 0, 0, 0, 0, 0, 0, 9],
            [9, 0, 0, 8, 0, 5, 0, 0, 6],
            [3, 0, 0, 9, 0, 7, 0, 0, 5],
            [0, 0, 0, 0, 9, 0, 0, 0, 0],
            [0, 9, 0, 0, 1, 0, 0, 2, 0],
            [0, 3, 8, 0, 0, 0, 9, 4, 0],
            [4, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 3, 5, 0, 9, 6, 0, 0],
            [0, 0, 2, 4, 0, 1, 3, 0, 0],
        ]
    ),
    "hard-106": np.array(
        [
            [0, 0, 0, 4, 0, 7, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 7, 0, 0],
            [4, 0, 0, 0, 0, 0, 0, 0, 3],
            [0, 2, 0, 3, 0, 9, 0, 4, 0],
            [0, 4, 0, 0, 1, 0, 0, 9, 0],
            [0, 0, 6, 0, 0, 0, 8, 0, 0],
            [5, 0, 0, 0, 0, 0, 0, 0, 8],
            [0, 8, 4, 0, 6, 0, 5, 3, 0],
            [3, 0, 0, 0, 0, 0, 0, 0, 2],
        ]
    ),
    "sd1": np.array(
        [
            [7, 9, 0, 0, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 0, 0, 0, 6, 0],
            [8, 0, 1, 0, 0, 4, 0, 0, 2],
            [0, 0, 5, 0, 0, 0, 0, 0, 0],
            [3, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 4, 0, 0, 0, 6, 2, 0, 9],
            [2, 0, 0, 0, 3, 0, 0, 0, 6],
            [0, 3, 0, 6, 0, 5, 4, 2, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    ),
    "sd2": np.array(
        [
            [1, 0, 0, 0, 0, 7, 0, 9, 0],
            [0, 3, 0, 0, 2, 0, 0, 0, 8],
            [0, 0, 9, 6, 0, 0, 5, 0, 0],
            [0, 0, 5, 3, 0, 0, 9, 0, 0],
            [0, 1, 0, 0, 8, 0, 0, 0, 2],
            [6, 0, 0, 0, 0, 4, 0, 0, 0],
            [3, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 4, 0, 0, 0, 0, 0, 0, 7],
            [0, 0, 7, 0, 0, 0, 3, 0, 0],
        ]
    ),
    "sd3": np.array(
        [
            [0, 0, 0, 0, 0, 3, 0, 1, 7],
            [0, 1, 5, 0, 0, 9, 0, 0, 8],
            [0, 6, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 7, 0, 0, 0],
            [0, 0, 9, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 5, 0, 0, 0, 0, 4],
            [0, 0, 0, 0, 0, 0, 0, 2, 0],
            [5, 0, 0, 6, 0, 0, 3, 4, 0],
            [3, 4, 0, 2, 0, 0, 0, 0, 0],
        ]
    ),
}


def run_ga(
    seed,
    puzzle_name: str,
    pop_size=150,
    row_crossover_rate=0.1,
    crossover_prob=0.2,
    swap_mutation_rate=0.3,
    reinit_mutation_rate=0.05,
    ls_greedier=False,
    ls_exhaustive=False,
    ls_strict=True,
    num_elite=50,
    epl_nudge=False,
    epl_force_unique=False,
    term_max_gen=10000,
):
    puzzle_board = puzzles[puzzle_name]
    problem = SudokuProblem(initial_board=puzzle_board)
    algorithm = GA(
        pop_size=pop_size,
        sampling=MySampling(),
        crossover=RowCrossover(row_cross_rate=row_crossover_rate, prob=crossover_prob),
        mutation=SwapReinitMutation(
            swap_rate=swap_mutation_rate, reinit_rate=reinit_mutation_rate
        ),
        repair=LocalSearchRepairV2(
            greedier=ls_greedier, exhaustive=ls_exhaustive, strict=ls_strict
        ),
        survival=EPLSurvival(
            n_elite=num_elite,
            sampling=MySampling(),
            swap_rate=swap_mutation_rate,
            reinit_rate=reinit_mutation_rate,
            nudge=epl_nudge,
            force_unique=epl_force_unique,
        ),
        eliminate_duplicates=True,
    )

    termination = TerminationCollection(
        get_termination("n_gen", term_max_gen), ZeroFunctionValueTermination()
    )

    start_time = time.perf_counter()
    res = minimize(problem, algorithm, termination, seed=seed, verbose=False)
    end_time = time.perf_counter()
    final_pop_f = res.pop.get("F")[:, 0]
    unique_f_count = len(np.unique(final_pop_f))
    return {
        "seed": seed,
        "puzzle": puzzle_name,
        "n_gen": res.algorithm.n_gen,
        "n_evals": res.algorithm.evaluator.n_eval,
        "success": int(res.F[0]) == 0,
        "time": end_time - start_time,
        "unique_f_in_pop": unique_f_count,
        "min_f": int(res.F[0]),
        "mean_f": np.mean(final_pop_f),
        "median_f": np.median(final_pop_f),
        "max_f": np.max(final_pop_f),
        "greedier": ls_greedier,
        "exhaustive": ls_exhaustive,
        "strict": ls_strict,
        "nudge": epl_nudge,
        "force_unique": epl_force_unique,
    }


def do_n_runs(
    num_runs: int,
    **kwargs,
) -> list[dict[str, float]]:
    seed_sequence = np.random.SeedSequence(42)
    run_seeds = seed_sequence.generate_state(num_runs)
    puzzle_name = kwargs.pop("puzzle_name")
    tasks = (delayed(run_ga)(seed, puzzle_name, **kwargs) for seed in run_seeds)
    results = []
    for res in tqdm(
        Parallel(n_jobs=14, return_as="generator_unordered")(tasks), total=num_runs
    ):
        results.append(res)
    return results


def produce_configs(original: bool, puzzle_name: str) -> dict:
    return {
        "puzzle_name": puzzle_name,
        "pop_size": 150,
        "row_crossover_rate": 0.1,
        "crossover_prob": 0.2,
        "swap_mutation_rate": 0.3,
        "reinit_mutation_rate": 0.05,
        "ls_greedier": not original,
        "ls_exhaustive": not original,
        "ls_strict": original,
        "num_elite": 50,
        "epl_nudge": not original,
        "epl_force_unique": not original,
        "term_max_gen": 10000 if puzzle_name == "sd3" else 2000,
        # Expect most results to just be burning CPU cycles
        # I highly doubt bumping up to 10000 will help much
        # sd3 is the only one i'm willing to go higher on since paper
        # didn't have great results either
    }


def main():
    output_file = "ga_benchmark_results.csv"
    all_results = []

    if os.path.exists(output_file):
        print(f"Found existing {output_file}, loading")
        all_results = pd.read_csv(output_file).to_dict("records")

    for puzzle_name in list(puzzles.keys()):
        for original in (True, False):
            variant_label = "paper" if original else "modified"
            key = f"{puzzle_name}-{'paper' if original else 'modified'}"
            if any(r.get("variant") == key for r in all_results):
                print(f"Skip {puzzle_name}")
                continue

            print(f"\nRunning {puzzle_name} - {variant_label}")
            cfg = produce_configs(original, puzzle_name)
            batch = do_n_runs(100, **cfg)

            for r in batch:
                r["variant"] = key
                r["variant_type"] = variant_label
            all_results.extend(batch)

            df_temp = pd.DataFrame(all_results)
            temp_file = f"{output_file}.tmp"
            df_temp.to_csv(temp_file, index=False)
            fd = os.open(temp_file, os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
            os.replace(temp_file, output_file)


if __name__ == "__main__":
    main()
