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

from sudoku.data.sudoku_explainer import (
    se_advance_puzzles,
    se_easy_puzzles,
    se_fiendish_puzzles,
    se_hard_puzzles,
    se_medium_puzzles,
    se_super_puzzles,
    se_superior_puzzles,
)
from sudoku.data.websudoku import (
    easy_puzzles,
    evil_puzzles,
    hard_puzzles,
    medium_puzzles,
)
from sudoku.operators import (
    EPLSurvival,
    LocalSearchRepairV2,
    MySampling,
    RowCrossover,
    SwapReinitMutation,
    ZeroFunctionValueTermination,
)
from sudoku.problem import SudokuProblem


def run_ga(
    seed,
    puzzle_name: str,
    puzzle_board: np.ndarray,
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


def worker_wrapper(task):
    res = run_ga(
        seed=task["seed"],
        puzzle_name=task["puzzle_name"],
        puzzle_board=task["puzzle_board"],
        **task["cfg"],
    )
    res["source"] = task["source"]
    res["difficulty"] = task["difficulty"]
    res["variant_type"] = "modified"
    return res


def main():
    output_file = "ga_supplement_benchmark_results.csv"

    all_tasks = []
    seed_sequence = np.random.SeedSequence(42)
    sources = {
        "websudoku": {
            "easy": easy_puzzles,
            "medium": medium_puzzles,
            "hard": hard_puzzles,
            "evil": evil_puzzles,
        },
        "sudoku_explainer": {
            "easy": se_easy_puzzles,
            "medium": se_medium_puzzles,
            "hard": se_hard_puzzles,
            "superior": se_superior_puzzles,
            "fiendish": se_fiendish_puzzles,
            "super": se_super_puzzles,
            "advance": se_advance_puzzles,
        },
    }

    for source_name, difficulty_dict in sources.items():
        for diff, puzzle_dict in difficulty_dict.items():
            for puzzle_name, board in puzzle_dict.items():
                max_gen = 1500
                # Most easy / medium stuff finishes below 500 gens
                # Some easy / medium stuff is tricky.
                # Keep at 1500 to get the results I used in report.
                # Otherwise, should finetune it for better perf.
                # Also consider changing other params besides term_max_gen.
                # if diff in ("evil", "fiendish", "super", "advance"):
                #     max_gen = 1500
                # elif diff in ("medium", "hard", "superior"):
                #     max_gen = 750
                # else:
                #     max_gen = 500

                run_seeds = seed_sequence.generate_state(10)
                for seed in run_seeds:
                    cfg = {
                        "pop_size": 150,
                        "row_crossover_rate": 0.1,
                        "crossover_prob": 0.2,
                        "swap_mutation_rate": 0.3,
                        "reinit_mutation_rate": 0.05,
                        "ls_greedier": True,
                        "ls_exhaustive": True,
                        "ls_strict": False,
                        "num_elite": 50,
                        "epl_nudge": True,
                        "epl_force_unique": True,
                        "term_max_gen": max_gen,
                    }
                    all_tasks.append(
                        {
                            "seed": int(seed),
                            "puzzle_name": puzzle_name,
                            "puzzle_board": board,
                            "source": source_name,
                            "difficulty": diff,
                            "cfg": cfg,
                        }
                    )

    finished_seeds = set()
    if os.path.exists(output_file):
        print(f"Found existing {output_file}, loading")
        df_existing = pd.read_csv(output_file)
        finished_seeds = set(
            zip(df_existing["puzzle"], df_existing["seed"], df_existing["source"])
        )

    tasks_to_run = [
        t
        for t in all_tasks
        if (t["puzzle_name"], t["seed"], t["source"]) not in finished_seeds
    ]
    print(f"Remaining: {len(tasks_to_run)}")

    if tasks_to_run:
        results = Parallel(n_jobs=14)(
            delayed(worker_wrapper)(t) for t in tqdm(tasks_to_run)
        )
        df_new = pd.DataFrame(results)
        if os.path.exists(output_file):
            df_old = pd.read_csv(output_file)
            df_final = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_final = df_new
        temp_file = f"{output_file}.tmp"
        df_final.to_csv(temp_file, index=False)
        fd = os.open(temp_file, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
        os.replace(temp_file, output_file)


if __name__ == "__main__":
    main()
