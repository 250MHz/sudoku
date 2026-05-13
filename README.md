# Evolutionary sudoku solver

## Getting started

Install [uv][1] and set up the project with `uv sync`.

Benchmarks used in report come from `scripts/run_bench.py`
and `scripts/run_supplement_bench.py`.
See `scripts/README.md` for more information.

## Developing

Store Jupyter notebooks in the `notebooks/` directory. Follow the [naming
convention][2] for notebooks used by Cookiecutter Data Science.

You should add a cell at the top of notebooks with the following:
```
%load_ext autoreload
%autoreload 2
```
This should make code from the `sudoku` module importable.

Before commiting, use `ruff` to format your Python code:
```sh
uv run ruff format
uv run ruff check --select I --fix
```

[1]: https://docs.astral.sh/uv/getting-started/installation/
[2]: https://cookiecutter-data-science.drivendata.org/using-the-template/
