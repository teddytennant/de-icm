# de_icm

Numpy implementation of DE-ICM, near-exact Independent Chip Model placement probabilities ([arXiv:2610.06028](https://arxiv.org/abs/2610.06028)).

## Install

```bash
pip install -e ".[test]"
```

## Run

```bash
python -m pytest -q
python demo.py
```

## What does not match the paper

The math is Algorithm 1 with the Section C grid, `c = 20` and `[x_L, x_R) = [-3.75, 27.5)`. This is single-threaded. It does not block nodes by 32 or use the parallel reduction from the paper. Fields of a few thousand players are supported by the same recurrence, but the tests check exact enumeration only up to six players.
