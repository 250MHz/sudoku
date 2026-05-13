import numpy as np


def validate_set(puzzles_dict, n=9):
    """Checks for:
    * At least 17 given numbers
    * No duplicates in rows
    * No dulicates in columns
    * No duplicates in sub-blocks.
    """
    sqrt_n = int(np.sqrt(n))
    assert sqrt_n == np.sqrt(n)
    res = {}
    for puzzle_name, grid in puzzles_dict.items():
        if grid is None:
            res[puzzle_name] = "Skipped (None)"
            continue
        errors = []
        num_given = np.count_nonzero(grid)
        if num_given < 17:
            errors.append(f"Invalid: Only {num_given} given numbers")
        for i in range(n):
            row = grid[i, :]
            row_given = row[row != 0]
            if len(row_given) != len(set(row_given)):
                errors.append(f"Row {i + 1} has duplicates")
            col = grid[:, i]
            col_given = col[col != 0]
            if len(col_given) != len(set(col_given)):
                errors.append(f"Column {i + 1} has duplicates")
        for r_offset in range(0, n, sqrt_n):
            for c_offset in range(0, n, sqrt_n):
                block = grid[
                    r_offset : r_offset + sqrt_n, c_offset : c_offset + sqrt_n
                ].flatten()
                block_given = block[block != 0]
                if len(block_given) != len(set(block_given)):
                    errors.append(
                        f"Block ({r_offset // sqrt_n + 1}, {c_offset // sqrt_n + 1}) has duplicates"
                    )
        if not errors:
            res[puzzle_name] = "valid"
        else:
            res[puzzle_name] = f"invalid: {'; '.join(errors)}"
    return res
