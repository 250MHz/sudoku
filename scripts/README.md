`run_bench.py` does runs over the 9 puzzles in Section IV.A from [Wang et al.][1]
`run_supplement.py` does runs over the 190 puzzles mentioend in Section IV.B
and Section IV.C.

Note that the scripts run with 14 parallel jobs.
Lower the job count if your hardware doesn't support that amount.

Run these scripts with [`nohup(1)`][2] to avoid crashing on `run_bench.py`
since can take a long time:

```sh
nohup uv run run_bench.py &
nohup uv run run_supplement_bench.py &
```

`run_bench.py` outputs a `ga_benchmark_results.csv` file with results.
`run_supplement_bench.py` outputs a `ga_supplement_benchmark_results.csv`
with results.
The results should be identical to `.csv` files already in the repo.

[1]: http://doi.org/10.1109/TG.2023.3236490
[2]: https://manned.org/man/nohup
